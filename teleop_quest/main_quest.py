#!/usr/bin/env python3
"""
Main control program for UR3e Robot via Meta Quest 3/3S (WebXR)
Modularized components:
  - bootstrap: Environment initialization, Docker URSim, ADB watcher, config & SSL
  - camera_pipeline: Video streaming and OSD management for Viser & Quest
  - quest_server: WebSocket & HTTP Server cho Quest
  - viser_ui: 3D Visualization and GUI controls
  - home_planner: Safe trajectory planning to Home configuration
  - delta_pose_source: VR relative pose & rotation processing
  - data_logger: Expert HDF5 data collection for ACT / Diffusion Policy
"""

import os
import sys
import time
import signal
import argparse
import traceback
import threading
import math
from typing import Optional, Dict, Any, List
import numpy as np
import torch

from core.pyroki_ursim_control import PyrokiURSim, StandbyRobotController
from core.selectable_recorder import SelectableDataRecorder
from teleop.pyroki_solver import UR3E_LIMITS
from teleop.teleoperation import (
    PyrokiIKSolverWrapper, SafetyGate, ServoJExecutor, TeleoperationController
)
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.quest_server import QuestServer
from teleop_quest.network_utils import get_local_ip, setup_ssl
from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop_quest.pose_sources import WidgetPoseSource, get_robot_fk_as_ur
from teleop_quest.timeout_watchdog import TimeoutWatchdog
from teleop_quest.viser_ui import TeleopViserUI
from teleop_quest.home_planner import HomePlanner
from teleop_quest.grasp_simulator import GraspSimulator
from teleop_quest.camera_driver import create_wrist_camera
from teleop_quest.camera_pipeline import CameraStreamPipeline
from teleop_quest.profiler import SystemProfiler
from teleop_quest.bootstrap import (
    load_singularity_config,
    ensure_ursim_container,
    start_adb_watcher,
    print_startup_banner,
)

POLL_SLEEP = 0.05
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_CONFIG = os.path.join(BASE_DIR, "assets", "obstacle_scene.yml")
SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]

def normalize_and_clip_joints(q_in, limits=UR3E_LIMITS, margin=2e-3) -> np.ndarray:
    """
    Chuẩn hóa góc khớp thực tế vào dải UR3E_LIMITS:
    1. Hóa giải hiện tượng cuộn góc ±2pi đưa góc về dải cho phép.
    2. Kẹp nhẹ [lower + margin, upper - margin] để triệt tiêu lỗi làm tròn chạm biên.
    """
    q_out = []
    for i, (lo, hi) in enumerate(limits):
        val = float(q_in[i])
        center = (lo + hi) / 2.0
        k_ideal = round((center - val) / (2.0 * math.pi))
        candidates = [val + (k_ideal + dk) * 2.0 * math.pi for dk in (-1, 0, 1)]
        in_range = [c for c in candidates if (lo - 0.1) <= c <= (hi + 0.1)]
        if in_range:
            chosen = min(in_range, key=lambda c: abs(c - center))
        else:
            chosen = val
        clipped = float(np.clip(chosen, lo + margin, hi - margin))
        q_out.append(clipped)
    return np.array(q_out, dtype=float)

def register_ui_callbacks(
    ui: TeleopViserUI,
    grasp_sim: GraspSimulator,
    logger: Any,
    teleop: TeleoperationController,
    executor: ServoJExecutor,
    ctrl: PyrokiURSim,
    widget_pose_source: WidgetPoseSource,
    home_planner: HomePlanner,
    quest_server: QuestServer,
    is_navigating_home: threading.Event,
    pose_source: DeltaPoseSource = None,
    trigger_go_home = None,
    gizmo_interaction: list = None,
):
    """Register event callbacks for Viser UI controls."""
    placed_at_start = [0]

    @ui.cb_teleop.on_update
    def _on_teleop_update(_):
        t0 = time.perf_counter()
        is_enabled = bool(ui.cb_teleop.value)
        teleop.enabled = is_enabled
        quest_server.set_drive_mode(is_enabled)
        if is_enabled:
            print(f"[Viser UI Event] User ENABLED Teleop from UI.")
            try:
                q_now = ctrl.joints()
                if q_now is not None and len(q_now) == 6:
                    q_norm = normalize_and_clip_joints(q_now, UR3E_LIMITS, margin=2e-3).tolist()
                    fk_now = teleop._solver.fk(q_norm)
                    if hasattr(ui.target_widget, "update_pose"):
                        ui.target_widget.update_pose(pos=fk_now["pos"], wxyz=fk_now["quat"], suppress=True)
                    else:
                        ui.target_widget.position = tuple(fk_now["pos"])
                        ui.target_widget.wxyz = tuple(fk_now["quat"])
                    widget_pose_source.sync_with_robot(pos=fk_now["pos"], quat=fk_now["quat"])
                    teleop.sync_with_robot(q_norm)
            except Exception as e:
                print(f"[Viser UI Event] Notice syncing teleop on enable: {e}")
                widget_pose_source.sync_with_robot()
            executor.connect()
            ui.set_status("OK")
        else:
            print(f"[Viser UI Event] User DISABLED Teleop from UI.")
            try:
                executor.stop(10.0)
            except Exception:
                pass
            try:
                teleop.stop(10.0)
            except Exception:
                pass
            ui.set_status("OK")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Viser UI Event] Teleop state updated: {is_enabled} (Latency: {latency_ms:.2f}ms)")

    @ui.cb_gripper.on_update
    def _on_gripper_update(_):
        t0 = time.perf_counter()
        is_closed = bool(ui.cb_gripper.value)
        quest_server.send_haptic(intensity=0.9 if is_closed else 0.4, duration=150)
        state_str = "GRIP (ON)" if is_closed else "RELEASE (OFF)"
        print(f"[Viser UI Event] User clicked Gripper: {state_str}")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Viser UI Event] Gripper state updated: {is_closed} (Latency: {latency_ms:.2f}ms)")


    if hasattr(ui, "ui_status") and hasattr(ui.ui_status, "on_update"):
        @ui.ui_status.on_update
        def _on_status_update(_):
            t0 = time.perf_counter()
            st_val = str(ui.ui_status.value)
            if quest_server:
                cur_pos = None
                cur_quat = None
                try:
                    q_c = ctrl.joints()
                    fk_c = teleop._solver.fk(q_c)
                    cur_pos = fk_c.get("pos")
                    cur_quat = fk_c.get("quat")
                except Exception:
                    pass
                quest_server.broadcast_robot_telemetry(pos=cur_pos, quat=cur_quat, status=st_val)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Viser UI Event] Robot status updated: '{st_val}' (Latency: {latency_ms:.2f}ms)")

    if hasattr(ui, "_raw_gizmo"):
        @ui._raw_gizmo.on_update
        def _on_gizmo_update(_):
            # 1. Ignore if programmatically suppressed or absorbing WebSocket echo
            if hasattr(ui, "target_widget"):
                if getattr(ui.target_widget, "is_suppressed", False):
                    return
                if getattr(ui.target_widget, "is_suppression_active", None) and ui.target_widget.is_suppression_active():
                    return

            # 2. Ignore if VR controller is actively engaged or Cockpit is jogging
            if quest_server is not None:
                if quest_server.is_cockpit_jogging():
                    return
                ctrl_frame = quest_server.get_frame() if hasattr(quest_server, "get_frame") else (quest_server.get_controller_frame() if hasattr(quest_server, "get_controller_frame") else None)
                if ctrl_frame and (ctrl_frame.get("trigger", False) or ctrl_frame.get("grip", False)):
                    return

            t0 = time.perf_counter()
            if gizmo_interaction is not None:
                gizmo_interaction[0] = time.time()
            pos = ui._raw_gizmo.position
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if not hasattr(_on_gizmo_update, "_last_log") or (time.time() - _on_gizmo_update._last_log > 1.0):
                _on_gizmo_update._last_log = time.time()
                print(f"[Viser UI Event] Gizmo moved -> Target: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] (Latency: {latency_ms:.2f}ms)")

    def toggle_recording_session(source_name: str = "Viser UI"):
        """Toggle start or stop & save HDF5 recording session."""
        if not logger.is_recording:
            placed_at_start[0] = grasp_sim.placed_count
            init_meta = {
                "tomato_init_pos": list(grasp_sim.tomato_pos),
                "basket_pos": list(grasp_sim.place_basket_pos),
            }
            active_streams = ui.get_recorder_config() if hasattr(ui, "get_recorder_config") else None
            filepath = logger.start_episode(
                task_description="Pick and Place Tomato Teleop",
                metadata=init_meta,
                streams=active_streams
            )
            ui.set_recording_state(True, logger.current_episode_idx)
            ui.set_status("OK")
            print(f"\n[AI Recorder] >>> START RECORDING Episode {logger.current_episode_idx} ({source_name}) -> {filepath}")
            if hasattr(quest_server, "send_haptic"):
                quest_server.send_haptic(0.9, 200)
        else:
            ep_num = logger.current_episode_idx
            ep_success = bool(grasp_sim.placed_count > placed_at_start[0] and not grasp_sim.is_grasped)
            saved_file = logger.end_episode(success=ep_success)
            ui.set_recording_state(False, ep_num + 1)
            if saved_file:
                ui.set_episodes(logger.current_episode_idx)
                status_str = "Success" if ep_success else "Incomplete"
                ui.set_status("OK")
                print(f"\n[AI Recorder] >>> SUCCESSFULLY SAVED Episode {ep_num} ({status_str}) ({source_name}) -> {saved_file}")
            else:
                ui.set_status("OK")
                print(f"\n[AI Recorder] >>> Stopped recording (No frames saved)")
            if hasattr(quest_server, "send_haptic"):
                quest_server.send_haptic(0.5, 100)

    def discard_recording_session(source_name: str = "Viser UI"):
        """Discard current recording session."""
        if logger.is_recording:
            logger.discard_episode()
            ui.set_recording_state(False, logger.current_episode_idx)
            ui.set_status("OK")
            print(f"\n[AI Recorder] >>> DISCARDED current episode ({source_name})!")
            if hasattr(quest_server, "send_haptic"):
                quest_server.send_haptic(0.7, 300)

    if hasattr(ui, "btn_record"):
        @ui.btn_record.on_click
        def _on_btn_record_click(_):
            toggle_recording_session(source_name="Viser UI 'Start Recording' Button")

    if hasattr(ui, "btn_discard"):
        @ui.btn_discard.on_click
        def _on_btn_discard_click(_):
            discard_recording_session(source_name="Viser UI 'Discard' Button")

    @ui.cb_record.on_update
    def _on_record_update(_):
        if ui.cb_record.value != logger.is_recording:
            toggle_recording_session(source_name="Checkbox Viser 'Record AI'")

    @ui.cb_discard.on_update
    def _on_cb_discard_update(_):
        if ui.cb_discard.value:
            discard_recording_session(source_name="Checkbox Viser 'Discard'")
            time.sleep(0.2)
            ui.cb_discard.value = False

    @ui.btn_home.on_click
    def _on_btn_home_click(_):
        if trigger_go_home is not None:
            trigger_go_home(source_name="Viser UI Home Button")
        else:
            if is_navigating_home.is_set():
                print("[HomePlanner] Moving to Home, please wait...")
                return
            is_navigating_home.set()
            teleop.enabled = False
            ui.cb_teleop.value = False
            ui.update_controller_visual(is_home=True)
            threading.Thread(
                target=home_planner.go_home,
                args=(
                    teleop, executor, ctrl, ui.target_widget,
                    widget_pose_source, ui.ui_status, ui.cb_teleop
                ),
                kwargs={
                    "is_navigating_home": is_navigating_home,
                    "poll_sleep": POLL_SLEEP,
                    "quest_server": quest_server
                },
                daemon=True
            ).start()

    if hasattr(ui, "slider_sensitivity") and ui.slider_sensitivity is not None:
        @ui.slider_sensitivity.on_update
        def _on_sensitivity_update(_):
            t0 = time.perf_counter()
            new_scale = float(ui.slider_sensitivity.value)
            if pose_source is not None and abs(pose_source.scale - new_scale) > 1e-3:
                pose_source.scale = new_scale
                if quest_server is not None:
                    quest_server.broadcast_robot_telemetry(scale=new_scale)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Viser UI Event] Sensitivity adjusted (Scale): {new_scale:.2f}x (Latency: {latency_ms:.2f}ms)")

    return {
        "toggle_recording": toggle_recording_session,
        "discard_recording": discard_recording_session,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="UR3e WebXR Teleoperation")
    parser.add_argument("--robot", type=str, choices=["real", "sim"], default=None, help="Robot mode: real (UR3e hardware) or sim (URSim simulation)")
    parser.add_argument("--robot-ip", type=str, default=None, help="UR3e robot IP address (default: 127.0.0.1 for URSim)")
    parser.add_argument("--camera", type=str, choices=["synthetic", "real", "realsense", "d455", "d435"], default="synthetic", help="Camera source: synthetic, real or realsense (D455/D435)")
    parser.add_argument("--cam-source", type=str, choices=["usb", "eth", "ws", "websocket"], default="usb", help="Camera connection method: usb, eth, or ws")
    parser.add_argument("--cam-url", type=str, default=os.environ.get("CAM_URL", "tcp://192.168.1.100:5555"), help="Camera Ethernet URL (e.g. tcp://192.168.1.100:5555)")
    parser.add_argument("--camera-device", type=int, default=0, help="Device ID for OpenCV camera (default: 0)")
    parser.add_argument("--camera-serial", type=str, default=None, help="RealSense camera serial number (default: auto-detect)")
    parser.add_argument("--disable-self-collision", action="store_true", default=False, help="Disable self-collision checking")
    parser.add_argument("--auto", action="store_true", default=False, help="Auto-discover UR3e Robot IP on local subnet via DHCP")
    args, _ = parser.parse_known_args()
    if args.auto:
        args.robot = "real"
        args.robot_ip = "auto"

    if args.robot is None:
        env_mode = os.environ.get("ROBOT_MODE", "").strip().lower()
        if env_mode in ("sim", "real"):
            args.robot = env_mode

    # Nếu chỉ định rõ --robot sim mà không gõ --robot-ip, luôn ưu tiên 127.0.0.1
    if args.robot == "sim" and args.robot_ip is None:
        args.robot_ip = "127.0.0.1"
    elif args.robot_ip is None:
        args.robot_ip = os.environ.get("ROBOT_IP", "127.0.0.1")

    return args

class TelemetryUIWorker:
    """
    Independent worker updating Viser UI & WebXR Telemetry:
    - Runs at independent rate (15-20Hz), decoupling 3D rendering from IK solver loop
    - Receives latest state snapshot from Main Control Loop via lock-free swap
    - Ensures Main Control Loop is never blocked by 3D rendering or WebSocket I/O
    """
    def __init__(self, ui: TeleopViserUI, quest_server: QuestServer, camera_pipeline: CameraStreamPipeline, ctrl: PyrokiURSim, grasp_sim: GraspSimulator):
        self.ui = ui
        self.quest_server = quest_server
        self.camera_pipeline = camera_pipeline
        self.ctrl = ctrl
        self.grasp_sim = grasp_sim
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_state: Optional[dict] = None
        self._cond = threading.Condition(self._lock)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TelemetryUIWorker")
        self._thread.start()

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def post_state(self, state_dict: dict):
        """Atomic snapshot update (< 0.005ms) from Main Control Loop."""
        with self._lock:
            self._latest_state = state_dict
            self._cond.notify()

    def _worker_loop(self):
        while self._running:
            with self._cond:
                if self._latest_state is None:
                    self._cond.wait(timeout=0.04)
                st = self._latest_state
                self._latest_state = None

            if st is None or not self._running:
                continue

            try:
                q_real = st.get("q_real")
                fk_pos = st.get("fk_pos")
                fk_quat = st.get("fk_quat")
                status = st.get("status", "")
                gripper_active = st.get("gripper_active", False)
                scale = st.get("scale", 1.0)
                is_recording = st.get("is_recording", False)
                is_holding = st.get("is_holding", False)
                is_connected = st.get("is_connected", False)
                hud_args = st.get("hud_args")
                sync_widget = st.get("sync_widget")

                # 1. Update connection state and grasp status
                self.ui.set_connected(is_connected)
                if status:
                    self.ui.set_status(status)
                self.ui.set_grasp_state(is_holding)

                # 2. Update Gizmo on Viser if VR is active (with deadband, sign continuity & suppression)
                if sync_widget and "pos" in sync_widget and "quat" in sync_widget:
                    if hasattr(self.ui.target_widget, "update_pose"):
                        self.ui.target_widget.update_pose(
                            pos=sync_widget["pos"],
                            wxyz=sync_widget["quat"],
                            suppress=True
                        )
                    else:
                        self.ui.target_widget.position = sync_widget["pos"]
                        self.ui.target_widget.wxyz = sync_widget["quat"]

                # 3. Update actual robot configuration on Viser 3D
                if q_real is not None:
                    self.ui.update_robot_state(q_real)

                # 4. Update 3D Controller HUD
                if hud_args:
                    self.ui.update_controller_visual(**hud_args)

                # 5. Broadcast Telemetry cho WebXR Meta Quest
                if fk_pos is not None and fk_quat is not None:
                    tcp_cur = None
                    try:
                        tcp_cur = self.ctrl.tcp()
                    except Exception:
                        pass

                    self.quest_server.broadcast_robot_telemetry(
                        pos=fk_pos,
                        quat=fk_quat,
                        status=status,
                        gripper=gripper_active,
                        joints=q_real,
                        tcp_pose=tcp_cur,
                        scale=scale,
                        recording=is_recording
                    )

                    # 6. Update tool pose for Camera Pipeline
                    self.camera_pipeline.update_state(
                        tool_pos=fk_pos,
                        tool_quat=fk_quat,
                        tomato_pos=self.grasp_sim.tomato_pos,
                        basket_pos=self.grasp_sim.place_basket_pos,
                        gripper_closed=gripper_active
                    )

            except Exception:
                pass


def main():
    print("=" * 60)
    print(" STARTING VR TELEOP (META QUEST 3/3S) ")
    print("=" * 60)

    args = parse_args()
    if args.disable_self_collision:
        os.environ["DISABLE_SELF_COLLISION"] = "1"
        print("[Safety] Flag --disable-self-collision active: Self-collision check DISABLED!")

    # Auto-Detect thông minh giữa Robot thật và URSim:
    # Nếu IP hiện tại trỏ tới một địa chỉ ngoài và không tìm thấy, tự động kiểm tra URSim tại 127.0.0.1
    is_real_robot = (args.robot == "real") or (args.robot is None and args.robot_ip not in ("127.0.0.1", "localhost", "192.168.56.101"))

    # Kiểm tra mạng Robot thật (Ping, RTDE 30004, Secondary 30002, Dashboard 29999)
    if is_real_robot and args.robot_ip not in ("auto", "dhcp", "DHCP", "127.0.0.1", "localhost"):
        def _quick_probe(ip, port, timeout=0.8):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((ip, port))
                s.close()
                return True
            except Exception:
                return False

        ping_ok = subprocess.run(["ping", "-c", "1", "-W", "1", args.robot_ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        p_rtde = _quick_probe(args.robot_ip, 30004)
        p_sec = _quick_probe(args.robot_ip, 30002)
        p_dash = _quick_probe(args.robot_ip, 29999)

        print("\n" + "=" * 68)
        print(f"  📡 KIỂM TRA ĐƯỜNG TRUYỀN MẠNG ROBOT ({args.robot_ip})")
        print("=" * 68)
        print(f"  [+] Ping ICMP ({args.robot_ip}):           {'✅ THÀNH CÔNG' if ping_ok else '❌ KHÔNG PHẢN HỒI (Ping timeout)'}")
        print(f"  [+] Cổng RTDE 30004 (Điều khiển 125Hz):  {'✅ ĐÃ MỞ (Sẵn sàng)' if p_rtde else '❌ ĐÓNG / BỊ CHẶN'}")
        print(f"  [+] Cổng Secondary 30002 (URScript):     {'✅ ĐÃ MỞ (Sẵn sàng)' if p_sec else '❌ ĐÓNG / BỊ CHẶN'}")
        print(f"  [+] Cổng Dashboard 29999 (PolyScope):    {'✅ ĐÃ MỞ (Sẵn sàng)' if p_dash else '❌ ĐÓNG / BỊ CHẶN'}")

        if not (ping_ok or p_rtde or p_sec or p_dash):
            # Kiểm tra card mạng máy tính có dải IP tương ứng không (vd: 10.42.0.x)
            try:
                ip_prefix = ".".join(args.robot_ip.split(".")[:3])
                ip_out = subprocess.check_output(["ip", "-4", "addr", "show"], stderr=subprocess.DEVNULL).decode()
                has_matching_subnet = ip_prefix in ip_out
                if not has_matching_subnet:
                    print("\n  ⚠️  CHẨN ĐOÁN NGUYÊN NHÂN MẤT KẾT NỐI:")
                    print(f"     Card mạng máy tính CHƯA CÓ IP cùng dải mạng '{ip_prefix}.x' với Robot!")
                    print(f"     👉 Hãy gán IP tĩnh cho card mạng Ethernet trên máy tính (ví dụ: {ip_prefix}.1):")
                    print(f"        sudo ip addr add {ip_prefix}.1/24 dev <tên_card_mạng_ethernet>")
                    print("        (Hoặc vào Settings -> Network -> Wired -> IPv4: chọn Manual, gán IP tương ứng).")
            except Exception:
                pass

        if not (p_rtde or p_sec or p_dash) and args.robot != "real":
            # Nếu người dùng không chỉ định cứng 'real', tự động kiểm tra xem có URSim local không
            if _quick_probe("127.0.0.1", 30004, timeout=0.3) or _quick_probe("127.0.0.1", 29999, timeout=0.3):
                print(f"  [Auto-Detect] Phát hiện URSim đang chạy tại 127.0.0.1! Tự động chuyển sang URSim.")
                args.robot_ip = "127.0.0.1"
                os.environ["ROBOT_IP"] = "127.0.0.1"
                args.robot = "sim"
                is_real_robot = False
        print("=" * 68 + "\n")

    # Auto-discover UR3e IP via DHCP if auto mode selected
    if is_real_robot and (args.robot_ip in ("auto", "dhcp", "DHCP") or (args.robot == "real" and args.robot_ip in ("127.0.0.1", "localhost"))):
        from core.robot_finder import discover_robot_ip
        discovered_ip = discover_robot_ip(timeout=12.0, verbose=True)
        if discovered_ip:
            args.robot_ip = discovered_ip
            os.environ["ROBOT_IP"] = discovered_ip
        else:
            print("\n\033[91;1m" + "!" * 72 + "\033[0m")
            print("\033[91;1m[AUTO-DISCOVERY ERROR] No UR3e Robot found on local network!\033[0m")
            print("  1. Check network cable from PC to UR Control Box.")
            print("  2. On Teach Pendant (PolyScope): Open Settings -> System -> Network.")
            print("  3. Select DHCP on PolyScope and apply to receive IP.")
            print("\033[91;1m" + "!" * 72 + "\033[0m\n")
            sys.exit(1)

    os.environ["ROBOT_IP"] = args.robot_ip
    mode_text = "REAL HARDWARE (UR3e)" if is_real_robot else "SIMULATION (URSim Docker)"
    print(f"[Bootstrap] Operating Mode: {mode_text} | Robot IP: {args.robot_ip}")

    # 1. Initialize networking, SSL certificate, and Quest WebXR server
    local_ip = get_local_ip()
    cert_path, key_path = setup_ssl(BASE_DIR, local_ip)
    quest_server = QuestServer(cert_path, key_path)
    quest_server.start()
    start_adb_watcher()

    # 2. Check and start Docker URSim if in simulation mode
    if not is_real_robot and args.robot_ip in ("127.0.0.1", "localhost"):
        ensure_ursim_container(args.robot_ip)

    # 3. Start Robot Controller, SafetyGate & Data Logger
    print(f"[System] Connecting PyRoki (CPU JIT) and UR3e at IP: {args.robot_ip}...")
    ctrl = None
    is_standby = False
    try:
        ctrl = PyrokiURSim(scene_file=SCENE_CONFIG, robot_ip=args.robot_ip, is_real=is_real_robot)
    except Exception as e:
        # Nếu cấu hình trỏ tới IP khác và lỗi, chỉ thử fallback sang URSim nếu người dùng KHÔNG chỉ định cứng robot real
        if args.robot_ip not in ("127.0.0.1", "localhost") and args.robot != "real":
            try:
                print(f"[Auto-Detect Fallback] Robot {args.robot_ip} lỗi ({e}). Đang thử kết nối URSim tại 127.0.0.1...")
                ensure_ursim_container("127.0.0.1")
                ctrl = PyrokiURSim(scene_file=SCENE_CONFIG, robot_ip="127.0.0.1", is_real=False)
                args.robot_ip = "127.0.0.1"
                os.environ["ROBOT_IP"] = "127.0.0.1"
                is_real_robot = False
                print(f"[Auto-Detect Fallback] ✅ Kết nối URSim 127.0.0.1 THÀNH CÔNG!")
            except Exception:
                ctrl = None
        if ctrl is None:
            print("\n" + "\033[93m" + "!" * 72 + "\033[0m")
            print(f"\033[91;1m[LỖI KẾT NỐI PHẦN CỨNG] Chưa thể kết nối tới Robot tại {args.robot_ip} ({e}).\033[0m")
            print("\033[96m-> HỆ THỐNG TỰ ĐỘNG CHUYỂN SANG CHẾ ĐỘ STANDBY / PREVIEW:\033[0m")
            print("   - Giao diện 3D Viser Dashboard và WebXR Cockpit VẪN MỞ BÌNH THƯỜNG.")
            print("   - Bạn vẫn có thể trải nghiệm giao diện WebXR và kiểm thử camera.")
            print(f"   - Để điều khiển cánh tay robot thật: Hãy đảm bảo cáp Ethernet cắm chắc, PC có IP tĩnh cùng dải 10.42.0.x và PolyScope ở chế độ RUNNING.")
            print("\033[93m" + "!" * 72 + "\033[0m\n")
            ctrl = StandbyRobotController(scene_file=SCENE_CONFIG)
            is_standby = True

    actual_q = None
    if not is_standby:
        for _ in range(50):
            try:
                q_read = ctrl.joints()
                if q_read is not None and len(q_read) == 6:
                    q_arr = np.array(q_read, dtype=float)
                    if np.linalg.norm(q_arr) > 0.01:
                        actual_q = q_arr
                        break
            except Exception:
                pass
            time.sleep(0.1)
    else:
        actual_q = np.array(ctrl.joints(), dtype=float)


    if actual_q is None:
        print("[Warning] Robot joint reading timeout or all zeros, using SAFE_HOME_JOINTS fallback.")
        actual_q = np.array(SAFE_HOME_JOINTS, dtype=float)

    start_joints = normalize_and_clip_joints(actual_q, UR3E_LIMITS, margin=2e-3).tolist()

    config = load_singularity_config(BASE_DIR)
    logger = SelectableDataRecorder(config)
    guard = SingularityGuard(config)

    # Check initial singularity status
    try:
        init_m = guard.metrics_calc.get_metrics(torch.tensor([start_joints], device=guard.metrics_calc.device))
        init_sigma = init_m["sigma_min"].item()
        if init_sigma < guard.danger_threshold:
            print(f"[Warning] Robot is near singularity (sigma={init_sigma:.3f} < {guard.danger_threshold:.3f}). Please return to Home.")
    except Exception:
        pass

    # 4. Compute initial robot pose via PyRoki FK
    pyroki_solver = PyrokiIKSolverWrapper(ctrl.ik_solver)
    fk_init = pyroki_solver.fk(start_joints)
    robot_init_pos = fk_init["pos"]
    robot_init_quat = fk_init["quat"]

    pick_pos = (0.22, 0.22, 0.035)
    place_pos = (0.20, -0.20, 0.02)

    # 5. Initialize Viser 3D UI, Grasp Simulator, and Camera Pipeline
    ui = TeleopViserUI(
        init_pos=robot_init_pos,
        init_quat=robot_init_quat,
        init_joints=start_joints,
        init_tomato_pos=(pick_pos[0], pick_pos[1])
    )
    if hasattr(ui.target_widget, "update_pose"):
        ui.target_widget.update_pose(pos=robot_init_pos, wxyz=robot_init_quat, suppress=True)
    else:
        ui.target_widget.position = tuple(robot_init_pos)
        ui.target_widget.wxyz = tuple(robot_init_quat)
    ui.set_episodes(logger.current_episode_idx)

    grasp_sim = GraspSimulator(ui.server, pick_pos=pick_pos, place_pos=place_pos)
    wrist_cam = create_wrist_camera(
        camera_type=args.camera,
        cam_source=args.cam_source,
        cam_url=args.cam_url,
        device_index=args.camera_device,
        serial_number=args.camera_serial,
        width=640,
        height=480
    )

    camera_pipeline = CameraStreamPipeline(
        wrist_cam=wrist_cam,
        ui=ui,
        quest_server=quest_server,
        init_pos=robot_init_pos,
        init_quat=robot_init_quat,
        init_tomato_pos=pick_pos,
        init_basket_pos=place_pos
    )
    camera_pipeline.start()

    # Start TelemetryUIWorker running independently from Control Loop
    ui_worker = TelemetryUIWorker(
        ui=ui,
        quest_server=quest_server,
        camera_pipeline=camera_pipeline,
        ctrl=ctrl,
        grasp_sim=grasp_sim
    )
    ui_worker.start()

    # 6. Setup VR PoseSource and Teleop Pipeline
    widget_pose_source = WidgetPoseSource(ui.target_widget)
    widget_pose_source.sync_with_robot(pos=robot_init_pos, quat=robot_init_quat)
    if is_standby:
        class DummyExecutor:
            """Bộ thực thi giả lập động học (Kinematic Mock) khi không có robot thật hoặc URSim."""
            def __init__(self, mock_ctrl):
                self.ctrl = mock_ctrl
                self.ROBOT_IP = "127.0.0.1"
            def connect(self):
                print("[DummyExecutor] Chế độ Mô phỏng Động học nội bộ (Kinematic Simulation) đã sẵn sàng!")
            def disconnect(self):
                pass
            def send(self, q):
                if self.ctrl and hasattr(self.ctrl, "servo_j"):
                    self.ctrl.servo_j(q)
            def stop(self, a=10.0):
                pass
        executor = DummyExecutor(ctrl)
    else:
        executor = ServoJExecutor(dt=POLL_SLEEP, lookahead_time=0.11, gain=250, robot_ip=args.robot_ip)
        # Khởi tạo kết nối ServoJ chạy nền trước để kéo Gizmo phản hồi tức thì
        threading.Thread(target=executor.connect, daemon=True).start()

    teleop = TeleoperationController(
        pose_source=widget_pose_source,
        pyroki_solver=pyroki_solver,
        safety_gate=SafetyGate(guard=guard, solver=pyroki_solver),
        executor=executor
    )
    teleop.sync_with_robot(start_joints)
    if is_standby:
        ui.set_status("[STANDBY] ROBOT NOT CONNECTED")
    else:
        ui.set_status("OK")


    pose_source = DeltaPoseSource(
        quest_server, ctrl, scale=1.0,
        anchor_reference_func=lambda: get_robot_fk_as_ur(teleop, ctrl),
        alpha=0.28,
        max_delta_m=0.22
    )

    watchdog = TimeoutWatchdog(quest_server, executor)
    watchdog.start()
    home_planner = HomePlanner(scene_file=SCENE_CONFIG, viz=ui.viz, default_home_q=SAFE_HOME_JOINTS)

    is_navigating_home = threading.Event()

    def trigger_go_home(source_name="VR"):
        t0 = time.perf_counter()
        if is_navigating_home.is_set():
            print(f"[HomePlanner] ({source_name}) Returning to Home, ignoring new commands...")
            return False
        is_navigating_home.set()
        try:
            teleop.enabled = False
            ui.cb_teleop.value = False
            ui.update_controller_visual(is_home=True)
        except Exception as e:
            print(f"[HomePlanner] Notice preparing Home state: {e}")
        threading.Thread(
            target=home_planner.go_home,
            args=(
                teleop, executor, ctrl, ui.target_widget,
                widget_pose_source, ui.ui_status, ui.cb_teleop
            ),
            kwargs={
                "is_navigating_home": is_navigating_home,
                "poll_sleep": POLL_SLEEP,
                "quest_server": quest_server
            },
            daemon=True
        ).start()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[HomePlanner] Triggered Go Home from {source_name} (Latency: {latency_ms:.2f}ms)")
        return True

    gizmo_interaction = [0.0]
    recorder_ctrl = register_ui_callbacks(
        ui, grasp_sim, logger, teleop, executor, ctrl,
        widget_pose_source, home_planner, quest_server, is_navigating_home,
        pose_source=pose_source, trigger_go_home=trigger_go_home,
        gizmo_interaction=gizmo_interaction
    )
    toggle_recording = recorder_ctrl["toggle_recording"]
    discard_recording = recorder_ctrl["discard_recording"]

    # Clean shutdown handler
    def handle_sigint(sig, frame):
        print("\n[System] Shutting down cleanly...")
        ui_worker.stop()
        camera_pipeline.stop()
        teleop.enabled = False
        time.sleep(0.5)
        ctrl.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # 7. Print access instructions
    print_startup_banner(local_ip, quest_server.http_port, 8080)

    _last_teleop_state = False
    _last_quest_drive_mode = quest_server.get_drive_mode()
    _was_navigating_home = False
    _gripper_active = False
    _gripper_continuous_val = 0.0
    _last_viser_cb_val = False
    _last_trig_state = False
    _last_grip_state = False
    _last_scale_log_time = 0.0

    # Timer & State variables for VR buttons (Meta Quest right controller)
    _btn_b_prev = False
    _btn_b_press_start = 0.0
    _btn_b_long_fired = False
    _btn_b_last_pulse_t = 0.0

    _btn_a_prev = False
    _btn_a_press_start = 0.0
    _btn_a_long_fired = False

    _thumbstick_click_prev = False
    _btn_x_prev = False

    # 8. Initialize Performance Profiler & WebXR Watchdog
    profiler = SystemProfiler(window_size=100, print_to_stdout=False)
    _vr_tracking_lost = False
    q_real = None
    fk_res = None

    # 9. Main Control Loop (Phase-Locked 20.0Hz RTDE Cadence)
    target_loop_time = time.perf_counter()
    while True:
        loop_start_time = time.perf_counter()

        # Cập nhật FK và góc khớp thật sớm phục vụ watchdog & auto re-anchor
        try:
            q_real = ctrl.joints()
            fk_res = teleop._solver.fk(q_real)
        except Exception:
            pass

        if is_navigating_home.is_set():
            _was_navigating_home = True
            time.sleep(POLL_SLEEP)
            target_loop_time = time.perf_counter()
            continue

        if _was_navigating_home:
            _was_navigating_home = False
            _last_teleop_state = False
            ui.cb_teleop.value = True
            ui.update_controller_visual(is_home=False)

        # A. Check VR E-Stop
        if quest_server.check_and_clear_estop():
            teleop.enabled = False
            ui.cb_teleop.value = False
            quest_server.set_drive_mode(False)
            _last_teleop_state = False
            _last_quest_drive_mode = False
            try:
                executor._ctrl.stopJ(2.0)
            except Exception:
                pass
            ui.set_status("[HOLD: EMERGENCY_STOP] Triggered from VR")
            quest_server.send_haptic(1.0, 1000)

        # A1. Tự động đồng bộ và khôi phục mốc tham chiếu khi Enter VR / Reconnect mới
        if quest_server.check_and_clear_new_session():
            print("[WebXR] New VR session / reconnect detected -> Auto Re-anchoring state without toggling Teleop.")
            _vr_tracking_lost = False
            pose_source.reset_anchor()
            _last_trig_state = False
            _last_grip_state = False
            _btn_a_prev = False
            _btn_b_prev = False
            _btn_b_long_fired = False
            _btn_b_last_pulse_t = 0.0
            _btn_x_prev = False
            _thumbstick_click_prev = False
            if q_real is not None:
                teleop.notify_clutch_released(q_real)
            if fk_res is not None:
                widget_pose_source.sync_with_robot(pos=fk_res["pos"], quat=fk_res["quat"])
            if ui.cb_teleop.value:
                teleop.enabled = True
                _last_teleop_state = True
                quest_server.set_drive_mode(True)
                executor.connect()
                if q_real is not None:
                    teleop._last_valid_q = list(q_real)

        # A2. WebXR Watchdog (Monitor WebXR tracking loss > 1.0s)
        last_vr_seen = quest_server.last_seen()
        now_vr_time = time.time()
        delta_vr_time = (now_vr_time - last_vr_seen) if last_vr_seen > 0.0 else 0.0

        if last_vr_seen > 0.0 and delta_vr_time > 1.0:
            if not _vr_tracking_lost:
                _vr_tracking_lost = True
                print(f"[WebXR Watchdog] Lost VR tracking (> 1.0s, delta={delta_vr_time:.2f}s). Holding robot safely without terminating WebXR session.")
                # Giữ nguyên robot an toàn tại chỗ mà KHÔNG sập session WebXR và KHÔNG tắt teleop
                pose_source.reset_anchor()
                if q_real is not None:
                    teleop.notify_clutch_released(q_real)
                if fk_res is not None:
                    widget_pose_source.sync_with_robot(pos=fk_res["pos"], quat=fk_res["quat"])
                _btn_a_prev = False
                _btn_b_prev = False
                _btn_x_prev = False
                _thumbstick_click_prev = False
                _last_trig_state = False
                _last_grip_state = False
                # Edge-triggered update for Viser UI
                if (time.time() - gizmo_interaction[0]) >= 0.35 and not is_navigating_home.is_set():
                    ui.set_status("VR LOST")
        elif _vr_tracking_lost and delta_vr_time <= 0.40:
            _vr_tracking_lost = False
            pose_source.reset_anchor()
            if q_real is not None:
                teleop.notify_clutch_released(q_real)
            if fk_res is not None:
                widget_pose_source.sync_with_robot(pos=fk_res["pos"], quat=fk_res["quat"])
            _last_trig_state = False
            _last_grip_state = False
            if ui.cb_teleop.value:
                teleop.enabled = True
                _last_teleop_state = True
                quest_server.set_drive_mode(True)
            print(f"[WebXR Watchdog] Regained VR tracking (delta={delta_vr_time*1000.0:.1f}ms). Resuming safe control.")
            if (time.time() - gizmo_interaction[0]) >= 0.35 and not is_navigating_home.is_set():
                ui.set_status("OK")

        # Fetch latest camera frame (Non-blocking memory buffer)
        profiler.start_segment("Camera_Read")
        latest_cam_frame = camera_pipeline.get_latest_raw_frame()
        profiler.end_segment("Camera_Read")

        # B. Read state from Quest VR headset
        vr_frame = quest_server.get_frame()
        is_frame_tracking_lost = bool(vr_frame.get("tracking_lost", False)) if vr_frame else False
        if _vr_tracking_lost or is_frame_tracking_lost:
            _is_trig = False
            _is_grip = False
            _is_btn_b = False
            _is_btn_a = False
            _is_thumbstick_click = False
            _is_btn_x = False
        else:
            _is_trig = bool(vr_frame.get("trigger", False)) if vr_frame else False
            _is_grip = bool(vr_frame.get("grip", False)) if vr_frame else False
            _is_btn_b = bool(vr_frame.get("button_b", False) or vr_frame.get("button_y", False)) if vr_frame else False
            _is_btn_a = bool(vr_frame.get("button_a", False) or vr_frame.get("record_btn", False)) if vr_frame else False
            _is_thumbstick_click = bool(vr_frame.get("thumbstick_click", False)) if vr_frame else False
            _is_btn_x = bool(vr_frame.get("button_x", False) or vr_frame.get("gripper_raw", False)) if vr_frame else False

        thumbstick_axes = vr_frame.get("thumbstick_axes", [0.0, 0.0]) if vr_frame else [0.0, 0.0]
        stick_x = float(thumbstick_axes[0]) if (isinstance(thumbstick_axes, (list, tuple)) and len(thumbstick_axes) > 0) else float(vr_frame.get("thumbstick_x", 0.0) if vr_frame else 0.0)
        stick_y = float(thumbstick_axes[1]) if (isinstance(thumbstick_axes, (list, tuple)) and len(thumbstick_axes) > 1) else float(vr_frame.get("thumbstick_y", 0.0) if vr_frame else 0.0)
        _is_joy_tilt = bool(abs(stick_x) > 0.15 or abs(stick_y) > 0.15 or vr_frame.get("joystick_tilt", False)) if vr_frame else False
        _is_home = is_navigating_home.is_set()

        now_t = time.perf_counter()

        # B1. Handle Button B (Short press: Toggle Teleop | Long press >= 3s: Go Home / Reset)
        if _is_btn_b and not _btn_b_prev:
            _btn_b_press_start = now_t
            _btn_b_long_fired = False
            _btn_b_last_pulse_t = now_t
            # Rung nhẹ báo hiệu bắt đầu chu trình giữ nút B (0.0s)
            quest_server.send_haptic(intensity=0.3, duration=50)
        elif _is_btn_b:
            if not _btn_b_long_fired:
                hold_dur = now_t - _btn_b_press_start
                if hold_dur >= 3.0:
                    _btn_b_long_fired = True
                    print("[VR Button B] Hold >= 3.0s -> Triggering safe Go Home immediately!")
                    # Rung mạnh xác nhận thực thi thành công Reset to Home (3s)
                    quest_server.send_haptic(intensity=1.0, duration=400)
                    trigger_go_home(source_name="VR Button B (Hold >= 3.0s)")
                else:
                    # Rung nhịp đều đặn trong suốt quá trình đếm ngược 3s (mỗi 0.6s phát một xung rung tăng dần)
                    if (now_t - _btn_b_last_pulse_t) >= 0.6:
                        _btn_b_last_pulse_t = now_t
                        haptic_intensity = min(0.35 + (hold_dur / 3.0) * 0.45, 0.85)
                        quest_server.send_haptic(intensity=haptic_intensity, duration=60)
        elif _btn_b_prev and not _is_btn_b:
            if not _btn_b_long_fired:
                duration_s = now_t - _btn_b_press_start
                new_teleop_state = not ui.cb_teleop.value
                ui.cb_teleop.value = new_teleop_state
                quest_server.set_drive_mode(new_teleop_state)
                _last_quest_drive_mode = new_teleop_state
                state_str = "ENABLED" if new_teleop_state else "DISABLED"
                print(f"[VR Button B] Click ({duration_s:.2f}s < 3.0s) -> Teleop state: {state_str}")
                quest_server.send_haptic(intensity=0.7, duration=150)
            _btn_b_long_fired = False
        _btn_b_prev = _is_btn_b

        # B2. Handle Button A (Short press: Toggle Record | Long press >= 2s: Discard Episode)
        if _is_btn_a and not _btn_a_prev:
            _btn_a_press_start = now_t
            _btn_a_long_fired = False
        elif _is_btn_a:
            if not _btn_a_long_fired and (now_t - _btn_a_press_start) >= 2.0:
                _btn_a_long_fired = True
                print("[VR Button A] Hold >= 2.0s -> Discarding recording episode!")
                if logger.is_recording:
                    discard_recording(source_name="VR Button A (Hold >= 2.0s)")
                else:
                    quest_server.send_haptic(intensity=0.5, duration=150)
                    print("[VR Button A] Hold >= 2.0s but no recording session active.")
        elif _btn_a_prev and not _is_btn_a:
            if not _btn_a_long_fired:
                duration_s = now_t - _btn_a_press_start
                print(f"[VR Button A] Click ({duration_s:.2f}s < 2.0s) -> Toggle Recording")
                toggle_recording(source_name="VR Button A (Click < 2.0s)")
        _btn_a_prev = _is_btn_a

        # B3. Handle Joystick Click / Button X -> Toggle Gripper
        thumbstick_clicked = _is_thumbstick_click and not _thumbstick_click_prev
        _thumbstick_click_prev = _is_thumbstick_click
        btn_x_clicked = _is_btn_x and not _btn_x_prev
        _btn_x_prev = _is_btn_x

        if thumbstick_clicked or btn_x_clicked:
            _gripper_active = not _gripper_active
            _gripper_continuous_val = 1.0 if _gripper_active else 0.0
            ui.cb_gripper.value = _gripper_active
            _last_viser_cb_val = _gripper_active
            quest_server.send_haptic(intensity=0.9 if _gripper_active else 0.4, duration=150)
            src_name = "VR Joystick Click" if thumbstick_clicked else "VR Button X"
            print(f"[{src_name}] Gripper state changed: {'GRIP (ON)' if _gripper_active else 'RELEASE (OFF)'}")

        # B4. Handle Joystick X (Continuous Gripper Control)
        if not _is_trig and not _is_grip and not is_navigating_home.is_set() and abs(stick_x) > 0.35:
            d_grip = stick_x * 1.5 * POLL_SLEEP
            _gripper_continuous_val = float(np.clip(_gripper_continuous_val + d_grip, 0.0, 1.0))
            new_grip_active = (_gripper_continuous_val >= 0.5)
            if new_grip_active != _gripper_active:
                _gripper_active = new_grip_active
                ui.cb_gripper.value = _gripper_active
                _last_viser_cb_val = _gripper_active
                print(f"[VR Teleop] Joystick X ({stick_x:+.2f}) -> Gripper {'GRIP' if _gripper_active else 'RELEASE'} (val={_gripper_continuous_val:.2f})")

        # B5. Handle Joystick Y (Scale / Sensitivity)
        if not _is_trig and not _is_grip and not is_navigating_home.is_set() and abs(stick_y) > 0.35:
            t_stick = time.perf_counter()
            d_scale = -stick_y * 0.4 * POLL_SLEEP
            new_scale = float(np.clip(pose_source.scale + d_scale, 0.25, 3.0))
            pose_source.scale = new_scale
            STEP = 0.05
            snapped_scale = round(round(new_scale / STEP) * STEP, 2)
            if hasattr(ui, "slider_sensitivity") and ui.slider_sensitivity is not None:
                if abs(ui.slider_sensitivity.value - snapped_scale) >= (STEP - 1e-4):
                    ui.slider_sensitivity.value = snapped_scale
                    quest_server.broadcast_robot_telemetry(scale=snapped_scale)
            latency_ms = (time.perf_counter() - t_stick) * 1000.0
            if time.time() - _last_scale_log_time > 0.4:
                _last_scale_log_time = time.time()
                print(f"[VR Teleop] Thumbstick Y ({stick_y:+.2f}) -> Scale updated: {snapped_scale:.2f}x (Latency: {latency_ms:.2f}ms)")

        # D. Update connection status
        ui.set_connected(quest_server.is_connected())

        # E. Synchronize Teleop enable/disable

        if ui.cb_teleop.value != _last_teleop_state:
            _last_teleop_state = ui.cb_teleop.value
            teleop.enabled = ui.cb_teleop.value
            quest_server.set_drive_mode(ui.cb_teleop.value)
            _last_quest_drive_mode = ui.cb_teleop.value

            if teleop.enabled:
                print("[Teleop] CONTROL ENABLED.")
                pose_source.reset_anchor()
                teleop._last_valid_q = list(ctrl.joints())
                executor.connect()
                if quest_server.is_connected():
                    watchdog.arm()
                ui.set_status("OK")
            else:
                print("[Teleop] CONTROL DISABLED (PAUSED).")
                executor.stop(10.0)
                teleop.stop(10.0)
                watchdog.disarm()
                ui.set_status("OK")

        # F. Retrieve robot joint angles and FK pose
        try:
            q_real = ctrl.joints()
            fk_res = teleop._solver.fk(q_real)
        except Exception:
            q_real = None
            fk_res = None

        # G. Handle haptic feedback and safe stopping for Trigger & Grip
        # Rising edge (0 -> 1): Tắt hoàn toàn phản hồi rung khi bóp/giữ Trigger theo yêu cầu
        # if not _last_trig_state and _is_trig:
        #     quest_server.send_haptic(intensity=0.35, duration=40)

        # Falling edge (1 -> 0): Trigger released
        # Chốt vị trí cuối cùng trên Viser Gizmo và giữ robot đứng yên êm dịu mà không gọi servoStop
        if _last_trig_state and not _is_trig:
            pose_source.reset_anchor()
            if q_real is not None:
                teleop.notify_clutch_released(q_real)
            if fk_res is not None:
                widget_pose_source.sync_with_robot(pos=fk_res["pos"], quat=fk_res["quat"])
            try:
                qd = ctrl.rr.getActualQd()
                motor_vel = max(abs(v) for v in qd) if qd else 0.0
            except Exception:
                motor_vel = 0.0
            if motor_vel < 0.001:
                motor_vel = 0.0
            print(f"[Teleop] Trigger released -> Control handed over to Viser Gizmo (motor velocity = {motor_vel:.4f} rad/s)")
        _last_trig_state = _is_trig

        # Rising edge (0 -> 1) for Grip: Tắt hoàn toàn phản hồi rung khi bóp/giữ Grip theo yêu cầu
        # if not _last_grip_state and _is_grip:
        #     quest_server.send_haptic(intensity=0.35, duration=40)

        # Same for Grip release
        if _last_grip_state and not _is_grip:
            pose_source.reset_anchor()
            if q_real is not None:
                teleop.notify_clutch_released(q_real)
            if fk_res is not None:
                widget_pose_source.sync_with_robot(pos=fk_res["pos"], quat=fk_res["quat"])
            print("[Teleop] Grip released -> Control handed over to Viser Gizmo")
        _last_grip_state = _is_grip

        # H. Collect VR controller display state for UI worker
        delta_trans = pose_source.get_delta_translation() if hasattr(pose_source, "get_delta_translation") else np.zeros(3)
        vr_quat = vr_frame.get("quat", None) if vr_frame else None
        hud_args = {
            "is_translating": _is_trig,
            "is_rotating": _is_grip,
            "is_gripper_btn": _gripper_active,
            "is_btn_b": _is_btn_b,
            "is_joystick": (_is_joy_tilt or _is_thumbstick_click or abs(stick_y) > 0.35 or abs(stick_x) > 0.35),
            "is_home": _is_home,
            "delta_trans": delta_trans,
            "vr_quat": vr_quat
        }

        # Synchronize when user clicks Gripper checkbox on Viser
        if ui.cb_gripper.value != _last_viser_cb_val:
            _gripper_active = bool(ui.cb_gripper.value)
            _gripper_continuous_val = 1.0 if _gripper_active else 0.0
            _last_viser_cb_val = _gripper_active
            print(f"[Gripper Toggle] Viser UI click -> State: {'GRIP (ON)' if _gripper_active else 'RELEASE (OFF)'}")

        current_status_display = "OK"
        is_holding = False
        vr_pose = None

        # Auto-enable Teleoperation when user interacts with Cockpit Joint Position
        if quest_server.is_cockpit_jogging():
            if not teleop.enabled:
                ui.cb_teleop.value = True
                teleop.enabled = True
                _last_teleop_state = True
                quest_server.set_drive_mode(True)
                _last_quest_drive_mode = True
                executor.connect()
                if ctrl is not None and hasattr(ctrl, "joints"):
                    teleop._last_valid_q = list(ctrl.joints())

        # Auto-enable Teleoperation when user interacts with Viser 3D Gizmo
        if (time.time() - gizmo_interaction[0]) < 0.35:
            if not teleop.enabled:
                ui.cb_teleop.value = True
                teleop.enabled = True
                _last_teleop_state = True
                quest_server.set_drive_mode(True)
                _last_quest_drive_mode = True
                executor.connect()
                if ctrl is not None and hasattr(ctrl, "joints"):
                    teleop._last_valid_q = list(ctrl.joints())

        if teleop.enabled and fk_res is not None:
            try:
                target_pose = None
                source_type = None

                # Priority 0: Cockpit Joint Control (Sliders / Arrows / Direct Input)
                if quest_server.is_cockpit_jogging():
                    cockpit_q = quest_server.get_cockpit_target_joints()
                    if cockpit_q is not None and len(cockpit_q) == 6:
                        profiler.start_segment("Cockpit_Joint_Control")
                        res = teleop.step_joint_space(cockpit_q, q_real=q_real, max_step=0.06)
                        profiler.end_segment("Cockpit_Joint_Control")
                        current_status = res.get("status", "OK")
                        current_status_display = current_status

                        # FK Synchronization for 3D Gizmo & internal pose anchors
                        fk_now = res.get("fk")
                        if fk_now is not None:
                            fk_res = fk_now
                            if hasattr(ui.target_widget, "update_pose"):
                                ui.target_widget.update_pose(pos=fk_now["pos"], wxyz=fk_now["quat"], suppress=True)
                            else:
                                ui.target_widget.position = tuple(fk_now["pos"])
                                ui.target_widget.wxyz = tuple(fk_now["quat"])
                            widget_pose_source.sync_with_robot(pos=fk_now["pos"], quat=fk_now["quat"])
                            if hasattr(pose_source, "reset_anchor"):
                                pose_source.reset_anchor()
                            if hasattr(pose_source, "reanchor_position"):
                                pose_source.reanchor_position(fk_now["pos"])
                        source_type = "Cockpit Joint"
                else:
                    # Sync real joints to cockpit when not jogging
                    if q_real is not None:
                        quest_server.sync_cockpit_joints(q_real)

                    # 1. Priority 1: Quest VR Controller (Trigger translate / Grip rotate)
                    vr_pose = pose_source.get_pose()
                    if vr_pose is not None:
                        target_pose = vr_pose
                        source_type = "VR Controller"
                    else:
                        # 2. Priority 2: Viser 3D Gizmo (Mặc định khi VR không active)
                        # Đảm bảo người dùng kéo Viser Gizmo bất cứ lúc nào robot cũng lập tức bám theo
                        target_pose = widget_pose_source.get_pose()
                        source_type = "Viser Gizmo"

                if target_pose is not None:
                    # Solve PyRoki IK and send motion command
                    profiler.start_segment("IK_Solver")
                    teleop.step(q_real, pose=target_pose)
                    profiler.end_segment("IK_Solver")
                    if is_standby and hasattr(teleop, "_last_valid_q") and teleop._last_valid_q is not None:
                        ctrl.servo_j(teleop._last_valid_q)

                    current_status = teleop.get_status()
                    status_low = current_status.lower()
                    current_status_display = current_status

                    # Auto-recenter target if trapped in obstacle or singular pose
                    is_twisting = any(err in status_low for err in ["twist", "branch_flip"])
                    is_obstacle = any(err in status_low for err in ["obstacle", "collision", "table"])
                    is_stuck = is_twisting or is_obstacle or any(err in status_low for err in [
                        "out of reach",
                        "singularity", "joint_limit", "branch_flip", "collision"
                    ])

                    snap_threshold = 1.2 if (is_twisting or is_obstacle) else 1.8
                    if is_stuck:
                        pose_source.out_of_reach_time += POLL_SLEEP
                    else:
                        pose_source.out_of_reach_time = 0.0

                    if pose_source.out_of_reach_time >= snap_threshold:
                        if hasattr(pose_source, "_anchor_pos_quest"):
                            pose_source._anchor_pos_quest = None
                            pose_source._anchor_tcp = None
                            pose_source._ema_pos = None
                            pose_source._ema_quat_xyzw = None
                            pose_source._last_button_state = 0
                        widget_pose_source.sync_with_robot(pos=fk_res["pos"], quat=fk_res["quat"])
                        pose_source.out_of_reach_time = 0.0
                        teleop.sync_with_robot(q_real)
                        current_status_display = "OK"

                        if not hasattr(main, "_last_snap_log_time") or time.time() - main._last_snap_log_time > 3.0:
                            main._last_snap_log_time = time.time()
                            print(f"[Auto-Snap] Re-anchoring internal target: {current_status}")
                elif source_type != "Cockpit Joint":
                    # Dự phòng an toàn nếu không có target pose: Giữ nguyên vị trí bằng servoJ, TUYỆT ĐỐI không gọi servoStop
                    if q_real is not None:
                        teleop.notify_clutch_released(q_real)
                        executor.send(q_real)
                    pose_source.out_of_reach_time = 0.0
                    if not _is_trig and not _is_grip:
                        if _vr_tracking_lost:
                            current_status_display = "VR LOST"
                        else:
                            current_status_display = "OK"

                grip_val = 1.0 if _gripper_active else 0.0
                is_holding, grasp_event = grasp_sim.update(fk_res["pos"], _gripper_active)

                if grasp_event == "GRASP_SUCCESS":
                    quest_server.send_haptic(0.9, 150)
                elif grasp_event == "PLACE_SUCCESS":
                    quest_server.send_haptic(0.5, 300)

                # Record HDF5 data for AI training (Non-blocking enqueue)
                if logger.is_recording:
                    q_target_cmd = teleop._last_valid_q if teleop._last_valid_q is not None else q_real

                    # 1. Collect RGB and Depth frames according to UI flags
                    imgs = {}
                    logged_img = latest_cam_frame
                    if logged_img is not None and getattr(ui, "cb_stream_rgb", None) and ui.cb_stream_rgb.value:
                        imgs["wrist_camera"] = logged_img
                    elif logged_img is not None and not hasattr(ui, "cb_stream_rgb"):
                        imgs["wrist_camera"] = logged_img

                    if getattr(ui, "cb_stream_depth", None) and ui.cb_stream_depth.value and hasattr(camera_pipeline, "get_latest_depth_frame"):
                        logged_depth = camera_pipeline.get_latest_depth_frame()
                        if logged_depth is not None:
                            imgs["wrist_camera_depth"] = logged_depth

                    # 2. Collect rich robot telemetry (26 RTDE fields)
                    if hasattr(ctrl, "get_robot_state"):
                        robot_telemetry = ctrl.get_robot_state()
                    else:
                        robot_telemetry = {
                            "actual_q": list(q_real),
                            "actual_qd": list(ctrl.rr.getActualQd()),
                            "actual_tcp_pose": list(ctrl.tcp()),
                            "target_q": list(q_target_cmd),
                            "timestamp_host_ms": int(time.time() * 1000),
                            "timestamp_host_ns": time.perf_counter_ns(),
                            "timestamp_robot": float(ctrl.rr.getTimestamp()),
                        }

                    # 3. Attach Hardware Timestamp from Camera Driver
                    if hasattr(camera_pipeline, "wrist_cam") and hasattr(camera_pipeline.wrist_cam, "get_frame_metadata"):
                        cam_meta = camera_pipeline.wrist_cam.get_frame_metadata()
                        if "timestamp_global_ns" in cam_meta:
                            robot_telemetry["timestamp_camera_ns"] = cam_meta["timestamp_global_ns"]
                            robot_telemetry["camera_frame_number"] = cam_meta.get("frame_number", 0)

                    # 4. Extract VR Controller Pose and IK Target Trajectory
                    vr_controller_pose_7d = [0.0] * 7
                    if vr_frame and "pos" in vr_frame and "quat" in vr_frame:
                        vp = vr_frame["pos"]
                        vq = vr_frame["quat"] # [w, x, y, z] in WebXR standard
                        if vp is not None and vq is not None and len(vp) >= 3 and len(vq) >= 4:
                            vr_controller_pose_7d = [
                                float(vp[0]), float(vp[1]), float(vp[2]),
                                float(vq[1]), float(vq[2]), float(vq[3]), float(vq[0])
                            ]

                    target_tcp_pose_6d = [0.0] * 6
                    if hasattr(ctrl, "rr") and hasattr(ctrl.rr, "getTargetTCPPose"):
                        target_tcp_pose_6d = list(ctrl.rr.getTargetTCPPose())
                    elif target_pose is not None and "pos" in target_pose:
                        p_pos = target_pose["pos"]
                        target_tcp_pose_6d[:3] = [float(p_pos[0]), float(p_pos[1]), float(p_pos[2])]

                    # 5. Non-blocking enqueue (< 0.01ms)
                    logger.record_step(
                        telemetry=robot_telemetry,
                        images=imgs,
                        action=list(q_target_cmd) + [grip_val],
                        gripper_actual=grip_val,
                        gripper_target=grip_val,
                        q_target=q_target_cmd,
                        vr_controller_pose=vr_controller_pose_7d,
                        target_tcp_pose=target_tcp_pose_6d,
                    )

            except Exception as e:
                current_status_display = f"[ERROR: CONTROL_LOOP] {e}"
                print(f"[Control Loop Error] {e}")
                traceback.print_exc()

        elif not teleop.enabled:
            curr_st = str(ui.ui_status.value)
            if not curr_st.startswith("[ERROR") and not curr_st.startswith("[HOLD") and "Error" not in curr_st:
                current_status_display = "OK"
            if fk_res is not None:
                is_holding, grasp_event = grasp_sim.update(fk_res["pos"], _gripper_active)

        # F. Push state to TelemetryUIWorker (< 0.005ms)
        profiler.start_segment("UI_Update")
        if fk_res is not None:
            sync_widget_data = {"pos": vr_pose["pos"], "quat": vr_pose["quat"]} if (vr_pose is not None and source_type == "VR Controller") else None
            ui_worker.post_state({
                "q_real": q_real,
                "fk_pos": fk_res["pos"],
                "fk_quat": fk_res["quat"],
                "status": current_status_display,
                "gripper_active": _gripper_active,
                "scale": pose_source.scale,
                "is_recording": logger.is_recording,
                "is_holding": is_holding,
                "is_connected": quest_server.is_connected(),
                "sync_widget": sync_widget_data,
                "hud_args": hud_args
            })
        profiler.end_segment("UI_Update")

        elapsed = time.perf_counter() - loop_start_time
        profiler.tick_frame(elapsed * 1000.0)

        target_loop_time += POLL_SLEEP
        now_cadence = time.perf_counter()
        sleep_duration = target_loop_time - now_cadence
        if sleep_duration > 0.001:
            time.sleep(sleep_duration)
        else:
            time.sleep(0.002)  # Nhường CPU cho các luồng phụ
            target_loop_time = now_cadence

if __name__ == "__main__":
    main()
