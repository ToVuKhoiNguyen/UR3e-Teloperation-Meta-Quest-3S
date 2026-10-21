#!/usr/bin/env python3
"""
Chương trình chính điều khiển Robot UR3e qua kính Meta Quest 3S (WebXR)
Đã mô-đun hóa:
  - bootstrap: Khởi tạo môi trường, Docker URSim, ADB watcher, config & SSL
  - camera_pipeline: Quản lý luồng video và OSD cho Viser & Quest
  - quest_server: WebSocket & HTTP Server cho Quest
  - viser_ui: Giao diện trực quan 3D & nút bấm Viser
  - home_planner: Quy hoạch chuyển động về vị trí an toàn
  - delta_pose_source: Bộ xử lý biến thiên vị trí & xoay từ VR
  - data_logger: Bộ thu thập dữ liệu chuyên gia chuẩn HDF5 (ACT / Diffusion Policy)
"""

import os
import sys
import time
import signal
import argparse
import traceback
import threading
import numpy as np
import torch

from core.pyroki_ursim_control import PyrokiURSim
from core.data_logger import DataLogger
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
from teleop_quest.bootstrap import (
    load_singularity_config,
    ensure_ursim_container,
    start_adb_watcher,
    print_startup_banner,
)

POLL_SLEEP = 0.05
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_CONFIG = os.path.join(BASE_DIR, "assets", "obstacle_scene.yml")
SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]

def register_ui_callbacks(
    ui: TeleopViserUI,
    grasp_sim: GraspSimulator,
    logger: DataLogger,
    teleop: TeleoperationController,
    executor: ServoJExecutor,
    ctrl: PyrokiURSim,
    widget_pose_source: WidgetPoseSource,
    home_planner: HomePlanner,
    quest_server: QuestServer,
    is_navigating_home: threading.Event,
    pose_source: DeltaPoseSource = None,
    trigger_go_home = None,
):
    """Đăng ký sự kiện các nút bấm trên giao diện Viser UI"""
    placed_at_start = [0]

    @ui.cb_teleop.on_update
    def _on_teleop_update(_):
        t0 = time.perf_counter()
        is_enabled = bool(ui.cb_teleop.value)
        teleop.enabled = is_enabled
        quest_server.set_drive_mode(is_enabled)
        if is_enabled:
            print(f"[Viser UI Event] Người dùng BẬT Teleop từ giao diện.")
            widget_pose_source.sync_with_robot()
            try:
                teleop._last_valid_q = list(ctrl.joints())
            except Exception:
                pass
            executor.connect()
            ui.set_status("Tracking")
        else:
            print(f"[Viser UI Event] Người dùng TẮT Teleop từ giao diện.")
            executor.disconnect()
            ui.set_status("Tắt")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Viser UI Event] Đã cập nhật trạng thái Teleop: {is_enabled} (Độ trễ: {latency_ms:.2f}ms)")

    @ui.cb_gripper.on_update
    def _on_gripper_update(_):
        t0 = time.perf_counter()
        is_closed = bool(ui.cb_gripper.value)
        quest_server.send_haptic(intensity=0.9 if is_closed else 0.4, duration=150)
        state_str = "HÚT/KẸP (ON)" if is_closed else "THẢ/NHẢ (OFF)"
        print(f"[Viser UI Event] Người dùng click Gripper: {state_str}")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Viser UI Event] Đã cập nhật trạng thái Gripper: {is_closed} (Độ trễ: {latency_ms:.2f}ms)")

    if hasattr(ui, "ui_cam_mode"):
        @ui.ui_cam_mode.on_update
        def _on_cam_mode_update(_):
            t0 = time.perf_counter()
            mode_str = ui.get_camera_view_mode()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Viser UI Event] Đổi chế độ xem Camera: {ui.ui_cam_mode.value} -> '{mode_str}' (Độ trễ: {latency_ms:.2f}ms)")

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
            print(f"[Viser UI Event] Trạng thái Robot cập nhật: '{st_val}' (Độ trễ: {latency_ms:.2f}ms)")

    if hasattr(ui, "_raw_gizmo"):
        @ui._raw_gizmo.on_update
        def _on_gizmo_update(_):
            t0 = time.perf_counter()
            _ = ui._raw_gizmo.position
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if not hasattr(_on_gizmo_update, "_last_log") or (time.time() - _on_gizmo_update._last_log > 2.0):
                _on_gizmo_update._last_log = time.time()
                print(f"[Viser UI Event] Tương tác 3D Transform Gizmo (Độ trễ: {latency_ms:.2f}ms)")

    @ui.cb_record.on_update
    def _on_record_update(_):
        if ui.cb_record.value:
            placed_at_start[0] = grasp_sim.placed_count
            init_meta = {
                "tomato_init_pos": list(grasp_sim.tomato_pos),
                "basket_pos": list(grasp_sim.place_basket_pos),
            }
            logger.start_episode(task_description="Pick and Place Tomato Teleop", metadata=init_meta)
            ui.set_status(f"Recording Ep {logger.current_episode_idx}")
        else:
            ep_success = bool(grasp_sim.placed_count > placed_at_start[0] and not grasp_sim.is_grasped)
            saved_file = logger.end_episode(success=ep_success)
            if saved_file:
                ui.set_episodes(logger.current_episode_idx)
                status_str = "Thành công" if ep_success else "Chưa hoàn thành"
                ui.set_status(f"Saved Ep {logger.current_episode_idx - 1} ({status_str})")
            else:
                ui.set_status("Idle")

    @ui.cb_discard.on_update
    def _on_cb_discard_update(_):
        if ui.cb_discard.value:
            if logger.is_recording:
                logger.discard_episode()
                ui.cb_record.value = False
                ui.set_status("Discarded Episode")
                print("[AI Logger] Đã hủy Episode hiện tại theo yêu cầu!")
            time.sleep(0.2)
            ui.cb_discard.value = False

    @ui.btn_home.on_click
    def _on_btn_home_click(_):
        if trigger_go_home is not None:
            trigger_go_home(source_name="Viser UI Nút Home")
        else:
            if is_navigating_home.is_set():
                print("[HomePlanner] Đang di chuyển về Home, vui lòng đợi...")
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
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Viser UI Event] Điều chỉnh độ nhạy (Scale): {new_scale:.2f}x (Độ trễ: {latency_ms:.2f}ms)")

def parse_args():
    parser = argparse.ArgumentParser(description="UR3e WebXR Teleoperation")
    parser.add_argument("--robot-ip", type=str, default=os.environ.get("ROBOT_IP", "127.0.0.1"), help="Địa chỉ IP của Robot UR3e hoặc URSim")
    parser.add_argument("--camera", type=str, choices=["synthetic", "real", "realsense", "d455", "d435"], default="synthetic", help="Nguồn camera: synthetic, real hoặc realsense (D455/D435)")
    parser.add_argument("--camera-device", type=int, default=0, help="Device ID của camera thật (OpenCV)")
    parser.add_argument("--camera-serial", type=str, default=None, help="Serial number của camera RealSense (mặc định: tự động nhận diện)")
    args, _ = parser.parse_known_args()
    return args

def main():
    print("=" * 60)
    print(" KHỞI ĐỘNG HỆ THỐNG VR TELEOP (META QUEST 3S) ")
    print("=" * 60)

    args = parse_args()
    os.environ["ROBOT_IP"] = args.robot_ip

    # 1. Khởi tạo mạng, chứng chỉ SSL và máy chủ Quest WebXR
    local_ip = get_local_ip()
    cert_path, key_path = setup_ssl(BASE_DIR, local_ip)
    quest_server = QuestServer(cert_path, key_path)
    quest_server.start()
    start_adb_watcher()

    # 2. Tự động kiểm tra và khởi động Docker URSim nếu chạy mô phỏng cục bộ
    ensure_ursim_container(args.robot_ip)

    # 3. Khởi động Controller Robot, An toàn & Data Logger
    print(f"[Hệ thống] Đang kết nối PyRoki (CPU JIT) và Robot UR3e tại IP: {args.robot_ip}...")
    ctrl = PyrokiURSim(scene_file=SCENE_CONFIG, robot_ip=args.robot_ip)
    for _ in range(10):
        if np.linalg.norm(ctrl.joints()) > 0.01:
            break
        time.sleep(0.1)
    start_joints = ctrl.joints()

    config = load_singularity_config(BASE_DIR)
    logger = DataLogger(config)
    guard = SingularityGuard(config)

    # Kiểm tra trạng thái kỳ dị ban đầu
    try:
        init_m = guard.metrics_calc.get_metrics(torch.tensor([start_joints], device=guard.metrics_calc.device))
        init_sigma = init_m["sigma_min"].item()
        if init_sigma < guard.danger_threshold:
            print(f"[Cảnh báo] Robot đang ở thế kỳ dị (sigma={init_sigma:.3f} < {guard.danger_threshold:.3f}). Hãy đưa robot về Home.")
    except Exception:
        pass

    # 4. Tính toán vị trí ban đầu của robot qua PyRoki FK
    pyroki_solver = PyrokiIKSolverWrapper(ctrl.ik_solver)
    fk_init = pyroki_solver.fk(start_joints)
    robot_init_pos = fk_init["pos"]
    robot_init_quat = fk_init["quat"]

    pick_pos = (0.22, 0.22, 0.035)
    place_pos = (0.20, -0.20, 0.02)

    # 5. Khởi tạo Giao diện Viser 3D, Grasp Simulator và Camera Pipeline
    ui = TeleopViserUI(
        init_pos=robot_init_pos,
        init_quat=robot_init_quat,
        init_joints=start_joints,
        init_tomato_pos=(pick_pos[0], pick_pos[1])
    )
    ui.set_episodes(logger.current_episode_idx)

    grasp_sim = GraspSimulator(ui.server, pick_pos=pick_pos, place_pos=place_pos)
    wrist_cam = create_wrist_camera(camera_type=args.camera, device_index=args.camera_device, serial_number=args.camera_serial, width=640, height=480)

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

    # 6. Thiết lập nguồn tọa độ VR & Pipeline Teleop
    widget_pose_source = WidgetPoseSource(ui.target_widget)
    executor = ServoJExecutor(dt=POLL_SLEEP, lookahead_time=0.12, gain=250, robot_ip=args.robot_ip)
    teleop = TeleoperationController(
        pose_source=widget_pose_source,
        pyroki_solver=pyroki_solver,
        safety_gate=SafetyGate(guard=guard),
        executor=executor
    )
    teleop._last_valid_q = list(start_joints)

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
            print(f"[HomePlanner] ({source_name}) Đang di chuyển về Home, bỏ qua lệnh mới...")
            return False
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
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[HomePlanner] Kích hoạt Go Home từ {source_name} (Độ trễ khởi tạo: {latency_ms:.2f}ms)")
        return True

    register_ui_callbacks(
        ui, grasp_sim, logger, teleop, executor, ctrl,
        widget_pose_source, home_planner, quest_server, is_navigating_home,
        pose_source=pose_source, trigger_go_home=trigger_go_home
    )

    # Xử lý tắt an toàn
    def handle_sigint(sig, frame):
        print("\n[Hệ thống] Đang tắt an toàn...")
        camera_pipeline.stop()
        teleop.enabled = False
        time.sleep(0.5)
        ctrl.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # 7. In hướng dẫn truy cập
    print_startup_banner(local_ip, quest_server.http_port, 8080)

    _last_teleop_state = False
    _last_quest_drive_mode = quest_server.get_drive_mode()
    _was_navigating_home = False
    _gripper_active = False
    _last_vr_gripper_raw = False
    _last_viser_cb_val = False
    _last_btn_b_state = False
    _last_trig_state = False
    _last_grip_state = False
    _thumbstick_press_start_time = 0.0
    _thumbstick_home_triggered = False
    _last_scale_log_time = 0.0

    # 8. Vòng lặp điều khiển chính (20Hz)
    while True:
        loop_start_time = time.perf_counter()

        if is_navigating_home.is_set():
            _was_navigating_home = True
            time.sleep(POLL_SLEEP)
            continue

        if _was_navigating_home:
            _was_navigating_home = False
            _last_teleop_state = False
            ui.cb_teleop.value = True

        # A. Kiểm tra VR E-Stop
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
            ui.set_status("EMERGENCY STOP (Từ VR)!")
            quest_server.send_haptic(1.0, 1000)

        # B. Đọc trạng thái từ kính Quest
        vr_frame = quest_server.get_frame()
        _is_trig = bool(vr_frame.get("trigger", False)) if vr_frame else False
        _is_grip = bool(vr_frame.get("grip", False)) if vr_frame else False
        _is_btn_b = bool(vr_frame.get("button_b", False) or vr_frame.get("button_y", False)) if vr_frame else False
        _is_joy_tilt = bool(vr_frame.get("joystick_tilt", False)) if vr_frame else False
        _is_thumbstick_click = bool(
            vr_frame.get("thumbstick_click", False) or
            vr_frame.get("home", False)
        ) if vr_frame else False
        _is_home = is_navigating_home.is_set()
        if vr_frame and vr_frame.get("thumbstick", False) and not _is_home and not _is_joy_tilt:
            _is_joy_tilt = True

        # B1. Xử lý sự kiện nhấn cần Joystick (Thumbstick Click) -> Về Home an toàn
        # Ràng buộc an toàn:
        # - Chống bấm nhầm: Yêu cầu nhấn giữ liên tục >= 0.5s
        # - Khóa an toàn: Không kích hoạt khi đang bóp cò (_is_trig) hoặc đang giữ grip (_is_grip)
        # - Chống chạy chồng: Không kích hoạt khi đang thực hiện Go Home (is_navigating_home.is_set())
        if _is_thumbstick_click:
            if _is_trig or _is_grip:
                if _thumbstick_press_start_time > 0.0:
                    print("[VR Teleop] Bỏ qua nhấn Joystick Go Home vì đang bóp cò/grip điều khiển.")
                _thumbstick_press_start_time = 0.0
                _thumbstick_home_triggered = False
            elif is_navigating_home.is_set():
                pass
            else:
                if _thumbstick_press_start_time == 0.0:
                    _thumbstick_press_start_time = time.time()
                elif (time.time() - _thumbstick_press_start_time >= 0.5) and not _thumbstick_home_triggered:
                    _thumbstick_home_triggered = True
                    t_click = time.perf_counter()
                    quest_server.send_haptic(intensity=1.0, duration=300)
                    trigger_go_home(source_name="VR Thumbstick Click (Giữ ≥ 0.5s)")
                    latency_ms = (time.perf_counter() - t_click) * 1000.0
                    print(f"[VR Teleop] Nhấn giữ Joystick đủ 0.5s -> Kích hoạt Go Home an toàn (Độ trễ khởi tạo: {latency_ms:.2f}ms)")
        else:
            _thumbstick_press_start_time = 0.0
            _thumbstick_home_triggered = False

        # B2. Tinh chỉnh độ nhạy (Scale) bằng cần Analog dọc (Thumbstick Y) theo thời gian thực
        thumbstick_axes = vr_frame.get("thumbstick_axes", [0.0, 0.0]) if vr_frame else [0.0, 0.0]
        stick_y = float(thumbstick_axes[1]) if (isinstance(thumbstick_axes, (list, tuple)) and len(thumbstick_axes) > 1) else float(vr_frame.get("thumbstick_y", 0.0) if vr_frame else 0.0)

        if not _is_trig and not _is_grip and not is_navigating_home.is_set() and abs(stick_y) > 0.35:
            t_stick = time.perf_counter()
            # Cần đẩy lên (stick_y < -0.35): Tăng độ nhạy; Kéo xuống (stick_y > 0.35): Giảm độ nhạy
            d_scale = -stick_y * 0.4 * POLL_SLEEP
            new_scale = float(np.clip(pose_source.scale + d_scale, 0.25, 3.0))
            if abs(new_scale - pose_source.scale) > 0.005:
                pose_source.scale = round(new_scale, 2)
                if hasattr(ui, "slider_sensitivity") and ui.slider_sensitivity is not None:
                    ui.slider_sensitivity.value = pose_source.scale
                latency_ms = (time.perf_counter() - t_stick) * 1000.0
                if time.time() - _last_scale_log_time > 0.4:
                    _last_scale_log_time = time.time()
                    print(f"[VR Teleop] Thumbstick Y ({stick_y:+.2f}) -> Cập nhật Scale: {pose_source.scale:.2f}x (Độ trễ: {latency_ms:.2f}ms)")

        # C. Xử lý Nút B (hoặc Y trên tay trái) - Chuẩn stable-pyroki-2026-10-21:
        # Bấm Nút B (đơn lẻ hoặc kết hợp Grip) CHỈ dùng để Hiệu chuẩn hướng bàn máy.
        # TUYỆT ĐỐI KHÔNG tắt Teleop / không ngắt RTDE executor khi bấm Nút B.
        btn_b_triggered = _is_btn_b and not _last_btn_b_state
        _last_btn_b_state = _is_btn_b
        if btn_b_triggered:
            pose_source.calibrate_desk(vr_frame)
            quest_server.send_haptic(intensity=0.8, duration=150)
            ui.set_status("Hiệu chuẩn bàn máy OK")
            print("[VR Teleop] Nút B ấn -> Hiệu chuẩn hướng bàn máy thành công.")

        # D. Cập nhật trạng thái kết nối
        ui.set_connected(quest_server.is_connected())

        # E. Đồng bộ bật/tắt Teleop
        if quest_server.is_connected():
            curr_quest_mode = quest_server.get_drive_mode()
            if curr_quest_mode != _last_quest_drive_mode:
                _last_quest_drive_mode = curr_quest_mode
                ui.cb_teleop.value = curr_quest_mode

        if ui.cb_teleop.value != _last_teleop_state:
            _last_teleop_state = ui.cb_teleop.value
            teleop.enabled = ui.cb_teleop.value
            quest_server.set_drive_mode(ui.cb_teleop.value)
            _last_quest_drive_mode = ui.cb_teleop.value

            if teleop.enabled:
                print("[Teleop] KÍCH HOẠT ĐIỀU KHIỂN.")
                pose_source.reset_anchor()
                teleop._last_valid_q = list(ctrl.joints())
                executor.connect()
                if quest_server.is_connected():
                    watchdog.arm()
                ui.set_status("Tracking")
            else:
                print("[Teleop] HỦY KÍCH HOẠT ĐIỀU KHIỂN.")
                executor.disconnect()
                watchdog.disarm()
                ui.set_status("Tắt")

        # F. Lấy trạng thái góc khớp và FK của robot
        try:
            q_real = ctrl.joints()
            fk_res = teleop._solver.fk(q_real)
        except Exception:
            q_real = None
            fk_res = None

        # G. Phát hiện sự kiện nhả cò súng (Trigger: 1 -> 0):
        # Dừng robot ngay lập tức và in log xác nhận trạng thái cò = 0 và vận tốc motor = 0
        if _last_trig_state and not _is_trig:
            executor.stop(10.0)
            teleop.stop(10.0)
            pose_source.reset_anchor()
            if q_real is not None:
                teleop._last_valid_q = list(q_real)
            try:
                qd = ctrl.rr.getActualQd()
                motor_vel = max(abs(v) for v in qd) if qd else 0.0
            except Exception:
                motor_vel = 0.0
            if motor_vel < 0.001:
                motor_vel = 0.0
            print(f"[Teleop] Cò súng nhả (cò=0) -> Dừng robot ngay lập tức (vận tốc motor = {motor_vel:.4f} rad/s)")
        _last_trig_state = _is_trig

        # Tương tự khi nhả Grip xoay
        if _last_grip_state and not _is_grip:
            executor.stop(10.0)
            teleop.stop(10.0)
            pose_source.reset_anchor()
            if q_real is not None:
                teleop._last_valid_q = list(q_real)
        _last_grip_state = _is_grip

        # H. Cơ chế Toggle Gripper (Bấm nút A/X trên kính Quest hoặc click chuột trên Viser Dashboard)
        toggle_triggered = quest_server.check_and_clear_gripper_toggle()
        raw_vr_gripper = False
        if vr_frame:
            raw_vr_gripper = bool(
                vr_frame.get("gripper_toggle_btn", False) or
                vr_frame.get("gripper_raw", False) or
                vr_frame.get("button_a", False) or
                vr_frame.get("button_x", False)
            )

        if raw_vr_gripper and not _last_vr_gripper_raw:
            toggle_triggered = True
        _last_vr_gripper_raw = raw_vr_gripper

        # Cập nhật màu sắc & phím bấm của mô hình tay cầm VR trên Viser
        ui.update_controller_visual(
            is_translating=_is_trig,
            is_rotating=_is_grip,
            is_gripper_btn=raw_vr_gripper,
            is_btn_b=_is_btn_b,
            is_joystick=(_is_joy_tilt or _is_thumbstick_click or abs(stick_y) > 0.35),
            is_home=_is_home
        )

        if toggle_triggered:
            _gripper_active = not _gripper_active
            ui.cb_gripper.value = _gripper_active
            _last_viser_cb_val = _gripper_active
            quest_server.send_haptic(intensity=0.9 if _gripper_active else 0.4, duration=150)
            print(f"[Gripper Toggle] Phím VR ấn -> Chuyển trạng thái: {'HÚT/KẸP (ON)' if _gripper_active else 'THẢ/NHẢ (OFF)'}")

        # Đồng bộ khi người dùng click chuột vào checkbox trên Viser Dashboard
        if ui.cb_gripper.value != _last_viser_cb_val:
            _gripper_active = bool(ui.cb_gripper.value)
            _last_viser_cb_val = _gripper_active
            print(f"[Gripper Toggle] Viser UI click -> Chuyển trạng thái: {'HÚT/KẸP (ON)' if _gripper_active else 'THẢ/NHẢ (OFF)'}")

        if teleop.enabled and fk_res is not None:
            try:
                # Cập nhật tọa độ VR vào Widget 3D
                vr_pose = pose_source.get_pose()
                if vr_pose is not None:
                    ui.target_widget.position = vr_pose["pos"]
                    ui.target_widget.wxyz = vr_pose["quat"]

                    # Giải IK và gửi lệnh điều khiển tới robot
                    teleop.step(q_real, pose=vr_pose)
                    current_status = teleop.get_status()
                    status_low = current_status.lower()

                    if "thất bại" in status_low or "fail" in status_low:
                        ui.set_status("IK Error")
                    elif "ngoai tam voi" in status_low or "ngoài tầm với" in status_low or "out of reach" in status_low:
                        ui.set_status("Ngoài tầm với (Out of Reach)")
                    elif "mặt bàn" in status_low or "mat ban" in status_low or "table limit" in status_low:
                        ui.set_status("Sát mặt bàn bảo vệ")
                    elif "van xoan" in status_low or "vặn xoắn" in status_low or "tu the ket" in status_low or "tư thế kẹt" in status_low:
                        ui.set_status("Tư thế kẹt (Vặn xoắn)")
                    elif "vat can" in status_low or "vật cản" in status_low:
                        ui.set_status("Vướng vật cản")
                    elif "singularity" in status_low or "kì dị" in status_low:
                        ui.set_status("Singularity")
                    elif "sigma=" in status_low or "pyroki" in status_low:
                        ui.set_status("Tracking")
                    else:
                        ui.set_status(current_status.split("|")[0].strip())

                    # Tự động hồi phục XYZ nếu kẹt tư thế hoặc vướng vật cản kéo dài
                    is_twisting = any(err in status_low for err in ["van xoan", "vặn xoắn", "tu the ket", "tư thế kẹt", "twist"])
                    is_obstacle = any(err in status_low for err in ["vat can", "vật cản", "dung vat can", "đụng vật cản", "obstacle"])
                    is_stuck = is_twisting or is_obstacle or any(err in status_low for err in [
                        "ngoai tam voi", "ngoài tầm với", "out of reach",
                        "singularity", "wrist_clamp", "that bai", "thất bại", "fail", "ik error"
                    ])

                    snap_threshold = 1.2 if (is_twisting or is_obstacle) else 1.8
                    if is_stuck:
                        pose_source.out_of_reach_time += POLL_SLEEP
                        quest_server.send_haptic(intensity=0.8, duration=80)
                    else:
                        pose_source.out_of_reach_time = 0.0

                    if pose_source.out_of_reach_time >= snap_threshold:
                        ui.target_widget.position = fk_res["pos"]
                        ui.target_widget.wxyz = fk_res["quat"]
                        widget_pose_source._ema_pos = None
                        widget_pose_source._ema_quat_xyzw = None
                        if hasattr(pose_source, "_anchor_pos_quest"):
                            pose_source._anchor_pos_quest = None
                            pose_source._anchor_tcp = None
                            pose_source._ema_pos = None
                            pose_source._ema_quat_xyzw = None
                            pose_source._last_button_state = 0
                        pose_source.out_of_reach_time = 0.0
                        teleop._last_valid_q = list(q_real)
                        ui.set_status("Tự động thu hồi XYZ")
                        quest_server.send_haptic(intensity=1.0, duration=200)

                        if not hasattr(main, "_last_snap_log_time") or time.time() - main._last_snap_log_time > 3.0:
                            main._last_snap_log_time = time.time()
                            print(f"[Auto-Snap] Dừng: {current_status}")
                else:
                    # Cò súng không ấn (và không giữ grip) -> Dừng robot ngay lập tức
                    executor.stop(10.0)
                    pose_source.out_of_reach_time = 0.0
                    if not _is_trig and not _is_grip:
                        ui.set_status("Sẵn sàng (Nhả cò)")

                grip_val = 1.0 if _gripper_active else 0.0
                is_holding, grasp_event = grasp_sim.update(fk_res["pos"], _gripper_active)
                ui.set_grasp_state(is_holding)

                if grasp_event == "GRASP_SUCCESS":
                    quest_server.send_haptic(0.9, 150)
                elif grasp_event == "PLACE_SUCCESS":
                    quest_server.send_haptic(0.5, 300)

                # Ghi dữ liệu HDF5 phục vụ huấn luyện AI
                if logger.is_recording:
                    q_target_cmd = teleop._last_valid_q if teleop._last_valid_q is not None else q_real
                    logged_img = camera_pipeline.get_latest_raw_frame()
                    logger.log_step(
                        q_actual=q_real,
                        qd_actual=ctrl.rr.getActualQd(),
                        tcp_actual=ctrl.tcp(),
                        q_target=q_target_cmd,
                        gripper_actual=grip_val,
                        gripper_target=grip_val,
                        images={"wrist_camera": logged_img},
                    )

            except Exception as e:
                ui.set_status(f"Lỗi: {e}")
                print(f"[Lỗi Điều khiển] {e}")
                traceback.print_exc()

        elif not teleop.enabled:
            curr_st = str(ui.ui_status.value)
            if not curr_st.startswith("Lỗi") and not curr_st.startswith("EMERGENCY") and "Error" not in curr_st:
                ui.set_status("Tắt")
            if fk_res is not None:
                is_holding, grasp_event = grasp_sim.update(fk_res["pos"], _gripper_active)
                ui.set_grasp_state(is_holding)

        # F. Cập nhật giao diện 3D và truyền dữ liệu cho Camera Pipeline
        if fk_res is not None:
            ui.update_robot_state(q_real)

            tcp_cur = None
            try:
                tcp_cur = ctrl.tcp()
            except Exception:
                pass

            quest_server.broadcast_robot_telemetry(
                pos=fk_res["pos"],
                quat=fk_res["quat"],
                status=str(ui.ui_status.value),
                gripper=_gripper_active,
                joints=q_real,
                tcp_pose=tcp_cur
            )

            camera_pipeline.update_state(
                tool_pos=fk_res["pos"],
                tool_quat=fk_res["quat"],
                tomato_pos=grasp_sim.tomato_pos,
                basket_pos=grasp_sim.place_basket_pos,
                gripper_closed=_gripper_active
            )

        elapsed = time.perf_counter() - loop_start_time
        sleep_duration = max(0.001, POLL_SLEEP - elapsed)
        time.sleep(sleep_duration)

if __name__ == "__main__":
    main()
