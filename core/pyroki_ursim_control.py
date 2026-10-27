"""
PyRoki -> URSim UR3e Controller
======================================
Pipeline:
  PyRoki CPU (IK) -> best joint solution -> RTDE servoJ / URScript -> URSim
  RTDEReceive (port 30004) -> real-time state monitoring
"""

import os
import math
import socket
import time
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# -- PyRoki CPU Kinematics ----------------------------
from teleop.pyroki_solver import PyrokiIKSolver, normalize_to_current, validate_joints, UR3E_LIMITS

# -- ur-rtde -----------------------------------------------------------------
import rtde_receive

# ===========================================================================
# Configuration
def get_robot_ip() -> str:
    return os.environ.get("ROBOT_IP", "127.0.0.1")

TCP_LINK        = "tool0"          # End-effector link defined in ur3e_custom.urdf

SECONDARY_PORT  = 30002            # URScript secondary interface
DASHBOARD_PORT  = 29999            # Dashboard server

JOINT_ACCEL     = 1.5              # rad/s^2
JOINT_SPEED     = 1.0              # rad/s
BLEND_RADIUS    = 0.0              # 0 = stop at each waypoint

IK_RETURN_SEEDS = 5                # Number of IK solutions to evaluate

# UR3E_LIMITS, normalize_to_current, validate_joints are imported from teleop.pyroki_solver


# ===========================================================================
# URSim helper functions
# ===========================================================================

def setup_ursim(timeout: float = 35.0, is_real: bool = False) -> bool:
    """Kiểm tra trạng thái robot qua Dashboard (Port 29999). Hỗ trợ phân tách Robot thật và Mô phỏng."""
    robot_ip = get_robot_ip()
    target_type = "Robot thật (UR3e Real Hardware)" if is_real else "URSim Mô phỏng (Docker)"
    print(f"[Dashboard] Đang kiểm tra trạng thái {target_type} qua cổng 29999 tại IP: {robot_ip}...")
    if not is_real and robot_ip in ("127.0.0.1", "localhost", "192.168.56.101"):
        print(f"[URSim] Giao diện Teach Pendant PolyScope (noVNC): http://{robot_ip}:6080/vnc.html")

    start_time = time.time()
    while time.time() - start_time < timeout:
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn_timeout = 3.0 if is_real else 4.0
            s.settimeout(conn_timeout)
            s.connect((robot_ip, DASHBOARD_PORT))
            time.sleep(0.3)
            try:
                s.recv(4096)
            except Exception:
                pass

            def _cmd(c, w=0.5):
                s.sendall((c + "\n").encode())
                time.sleep(w)
                try:
                    s.setblocking(False)
                    r = s.recv(4096).decode().strip()
                    s.setblocking(True)
                except Exception:
                    s.setblocking(True)
                    r = ""
                return r

            mode = _cmd("robotmode", 0.3)
            safety = _cmd("safetystatus", 0.3)
            print(f"[{'Robot Thật' if is_real else 'URSim'}] Trạng thái: Mode={mode}, Safety={safety}")

            if "RUNNING" in mode:
                # Dừng script cũ đang chạy trên robot để giải phóng thanh ghi RTDE
                _cmd("stop", 0.2)
                # Kiểm tra chế độ Remote Control
                remote = _cmd("is in remote control", 0.3).lower()
                if "true" not in remote:
                    print("\n" + "\033[93m" + "!" * 72 + "\033[0m")
                    print("\033[93m[CẢNH BÁO REMOTE CONTROL] Robot CHƯA bật chế độ Remote Control!\033[0m")
                    print("\033[91m-> Vui lòng kiểm tra cáp mạng và chuyển Teach Pendant sang chế độ Remote Control.\033[0m")
                    print("   (Chọn biểu tượng Remote ở góc trên bên phải PolyScope màn hình robot).")
                    print("\033[93m" + "!" * 72 + "\033[0m\n")
                s.close()
                if is_real:
                    time.sleep(0.5)
                return True

            # Đối với robot thật, TUYỆT ĐỐI KHÔNG tự động bật nguồn (power on) hay nhả phanh (brake release)
            if is_real:
                print(f"[Robot Thật] Robot đang ở trạng thái {mode}. Vui lòng bật nguồn và nhả phanh trên Teach Pendant.")
                s.close()
                time.sleep(2.0)
                continue

            if "NO_CONTROLLER" in mode or not mode:
                print("[URSim] URControl đang khởi động, đợi 3 giây...")
                s.close()
                time.sleep(3.0)
                continue

            if any(err in safety for err in ["PROTECTIVE_STOP", "VIOLATION", "FAULT"]):
                print("[URSim] Đang mở khoá Protective Stop / Safety popup...")
                _cmd("unlock protective stop", 1.0)
                _cmd("close safety popup", 1.0)
                time.sleep(1.0)
                mode = _cmd("robotmode", 0.3)

            if "POWER_OFF" in mode or "IDLE" in mode or "DISCONNECTED" in mode:
                print("[URSim] Đang bật nguồn (power on) và nhả phanh (brake release)...")
                _cmd("power on", w=2.5)
                _cmd("brake release", w=4.0)
                mode = _cmd("robotmode", 0.5)
                print(f"[URSim] Trạng thái sau kích hoạt: {mode}")

            s.close()
            if "RUNNING" in mode:
                return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            if s:
                try:
                    s.close()
                except Exception:
                    pass
            print("\n" + "\033[93m" + "!" * 72 + "\033[0m")
            print(f"\033[93m[CẢNH BÁO KẾT NỐI] Không thể kết nối tới Dashboard ({robot_ip}:{DASHBOARD_PORT})! ({e})\033[0m")
            print("\033[91m-> Vui lòng kiểm tra cáp mạng và chuyển Teach Pendant sang chế độ Remote Control.\033[0m")
            print("\033[93m" + "!" * 72 + "\033[0m\n")

            if is_real:
                # Đối với robot thật, kiểm tra xem giao diện RTDE (port 30004) có hoạt động không
                print("[Hệ thống] Đang thử kết nối trực tiếp qua giao diện RTDE (Port 30004)...")
                try:
                    s_rtde = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s_rtde.settimeout(1.0)
                    s_rtde.connect((robot_ip, 30004))
                    s_rtde.close()
                    test_rr = rtde_receive.RTDEReceiveInterface(robot_ip)
                    test_q = test_rr.getActualQ()
                    test_rr.disconnect()
                    if test_q and len(test_q) == 6:
                        print(f"[RTDE Kết nối thành công] Đã đọc được góc khớp robot thật: {[round(x, 3) for x in test_q]}")
                        return True
                except Exception as rtde_err:
                    print(f"[RTDE Thất bại] Cổng RTDE 30004 chưa phản hồi: {rtde_err}")
                time.sleep(1.0)
            else:
                time.sleep(2.0)
        except Exception as e:
            if s:
                try:
                    s.close()
                except Exception:
                    pass
            print(f"[Dashboard] Lỗi ngoại lệ ({e})... Thử lại sau 2 giây.")
            time.sleep(2.0)

    print(f"[Dashboard] Hết thời gian chờ {target_type} chuyển sang trạng thái RUNNING.")
    return False


def send_urscript(script: str, settle: float = 0.3) -> None:
    """Upload URScript via secondary interface (port 30002). No Remote Control mode needed."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((get_robot_ip(), SECONDARY_PORT))
    time.sleep(0.1)
    try:
        s.recv(4096)
    except Exception:
        pass
    s.sendall(script.encode())
    time.sleep(settle)
    s.close()


def movej_script(q: list, a=JOINT_ACCEL, v=JOINT_SPEED, r=BLEND_RADIUS) -> str:
    joints = ", ".join(f"{x:.6f}" for x in q)
    return f"movej([{joints}], a={a}, v={v}, r={r})\n"


# ===========================================================================
# PyRoki CPU IK Solver
# ===========================================================================

class PyrokiIK:
    """
    CPU-accelerated IK using PyRoki + JAX.
    Solves kinematics on CPU, completely independent of NVIDIA GPU/CUDA.
    """

    def __init__(self, scene_file: Optional[str] = None):
        print("[PyRoki] Khởi tạo bộ giải IK CPU cho UR3e...")
        self._solver = PyrokiIKSolver()
        print("[PyRoki] IK solver ready")

    def solve(
        self,
        *args,
        position: Optional[list] = None,
        quaternion: Optional[list] = None,
        current_q: Optional[list] = None,
        q_seed: Optional[list] = None,
        pose: Any = None,
        **kwargs
    ) -> Any:
        """
        Solve IK using PyRoki. Hỗ trợ đa dạng cú pháp gọi:
        1. solve(position, quaternion, current_q=...) -> list khớp
        2. solve(pose, q_seed=...) -> dict kết quả
        """
        seed = q_seed if q_seed is not None else current_q
        is_legacy = False
        target_pose = pose

        if len(args) == 1:
            target_pose = args[0]
        elif len(args) == 2:
            if isinstance(args[1], (list, tuple, np.ndarray)) and len(args[1]) == 6:
                target_pose = args[0]
                if seed is None:
                    seed = args[1]
            else:
                target_pose = (args[0], args[1])
                is_legacy = True
        elif len(args) >= 3:
            target_pose = (args[0], args[1])
            if seed is None:
                seed = args[2]
            is_legacy = True

        if target_pose is None and position is not None and quaternion is not None:
            target_pose = (position, quaternion)
            is_legacy = True

        res = self._solver.solve(pose=target_pose, q_seed=seed)
        if is_legacy and not kwargs.get("return_dict", False):
            return res.get("q")
        return res

    def __getattr__(self, name):
        return getattr(self._solver, name)


# ===========================================================================
# High-Level Controller
# ===========================================================================

class PyrokiURSim:
    """
    PyRoki CPU IK -> URSim / UR3e via URScript / RTDE.
    Hoàn toàn không yêu cầu GPU.
    Hỗ trợ cả chế độ Robot thật (Real) và Mô phỏng (Sim).
    """

    def __init__(self, scene_file: Optional[str] = None, robot_ip: Optional[str] = None, is_real: Optional[bool] = None):
        if robot_ip:
            os.environ["ROBOT_IP"] = str(robot_ip)
        current_ip = get_robot_ip()
        if is_real is None:
            is_real = current_ip not in ("127.0.0.1", "localhost", "192.168.56.101")
        self.is_real = bool(is_real)

        print("\n" + "=" * 60)
        mode_str = "ROBOT THẬT (UR3e Real Hardware)" if self.is_real else "MÔ PHỎNG (URSim Docker)"
        print(f"  PyRoki (CPU JIT) -> {mode_str} Controller")
        print("=" * 60)

        # Kiểm tra Dashboard (cổng 29999) với xử lý lỗi thân thiện
        dashboard_ok = setup_ursim(timeout=8.0 if self.is_real else 35.0, is_real=self.is_real)
        if not dashboard_ok and not self.is_real:
            raise RuntimeError("Robot not RUNNING. Check URSim container.")
        elif not dashboard_ok and self.is_real:
            print("\033[93m[CẢNH BÁO] Không có phản hồi Dashboard. Tiếp tục thử kết nối trực tiếp cổng RTDE...\033[0m")

        print(f"[Setup] Connecting RTDEReceive ({current_ip}:30004)...")
        max_retries = 5 if self.is_real else 3
        retry_delay = 1.0
        self.rr = None
        for attempt in range(1, max_retries + 1):
            try:
                self.rr = rtde_receive.RTDEReceiveInterface(current_ip)
                q = self.rr.getActualQ()
                print(f"[Setup] Connected (attempt {attempt}). Joints: {[round(j, 4) for j in q]}")
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"[Setup] RTDEReceive thử lần {attempt}/{max_retries} chưa thành công ({e}). Đang thử lại sau {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    import subprocess
                    ping_ok = subprocess.run(["ping", "-c", "1", "-W", "1", current_ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                    print("\n" + "\033[91m" + "!" * 72 + "\033[0m")
                    print(f"\033[91m[LỖI KẾT NỐI RTDE] Không thể kết nối tới robot tại {current_ip}:30004 sau {max_retries} lần thử ({e})\033[0m")
                    if not ping_ok:
                        print(f"\033[93m-> CHẨN ĐOÁN MẠNG: Robot tại IP {current_ip} HOÀN TOÀN KHÔNG PHẢN HỒI (Ping 100% loss / ARP FAILED)!\033[0m")
                        print("   1. Mở màn hình Teach Pendant (PolyScope): Vào Settings -> System -> Network.")
                        print("   2. Xem trạng thái đèn mạng: Nếu màu ĐỎ -> cáp Ethernet chưa nhận hoặc IP chưa được kích hoạt.")
                        print(f"   3. Kiểm tra địa chỉ IP hiển thị trên màn hình PolyScope: Có đúng là {current_ip} không?")
                        print(f"      - Nếu robot dùng DHCP: Xem IP mới được cấp là gì rồi chạy lại lệnh với --robot-ip <IP_MOI>.")
                        print(f"      - Nếu robot dùng IP Tĩnh: Đặt IP={current_ip}, Subnet=255.255.255.0, Gateway=10.42.0.1 rồi bấm Apply.")
                    else:
                        print(f"\033[93m-> CHẨN ĐOÁN: Robot phản hồi Ping nhưng cổng RTDE 30004 bị từ chối kết nối!\033[0m")
                        print("   1. Đảm bảo robot đã bật nguồn và nhả phanh (Power On -> Brake Release) ở trạng thái RUNNING.")
                        print("   2. Chuyển Teach Pendant sang chế độ Remote Control (góc trên bên phải màn hình PolyScope).")
                    print("\033[91m" + "!" * 72 + "\033[0m\n")
                    raise

        self._ik = None
        self.scene_file = scene_file
        print("=" * 60 + "\n")

    @property
    def ik_solver(self) -> PyrokiIK:
        if self._ik is None:
            self._ik = PyrokiIK(scene_file=self.scene_file)
        return self._ik

    # -- State ---------------------------------------------------------------

    def joints(self) -> list:
        """Current joint positions [rad] x 6."""
        return self.rr.getActualQ()

    def tcp(self) -> list:
        """Current TCP pose [x, y, z, rx, ry, rz]."""
        return self.rr.getActualTCPPose()

    # -- Extended RTDE Multimodal Telemetry ----------------------------------

    def get_kinematics(self) -> Dict[str, Any]:
        """Truy xuất toàn diện dữ liệu động học (Góc khớp, vận tốc, gia tốc, TCP pose, TCP speed)."""
        return {
            "actual_q": list(self.rr.getActualQ()),
            "actual_qd": list(self.rr.getActualQd()),
            "target_q": list(self.rr.getTargetQ()),
            "target_qd": list(self.rr.getTargetQd()),
            "target_qdd": list(self.rr.getTargetQdd()),
            "actual_tcp_pose": list(self.rr.getActualTCPPose()),
            "actual_tcp_speed": list(self.rr.getActualTCPSpeed()),
            "target_tcp_pose": list(self.rr.getTargetTCPPose()),
            "target_tcp_speed": list(self.rr.getTargetTCPSpeed()),
        }

    def get_dynamics(self) -> Dict[str, Any]:
        """Truy xuất dữ liệu động lực học & điện học (Dòng điện, mô-men, điện áp các khớp và robot)."""
        return {
            "actual_current": list(self.rr.getActualCurrent()),
            "target_current": list(self.rr.getTargetCurrent()),
            "target_moment": list(self.rr.getTargetMoment()),
            "joint_voltages": list(self.rr.getActualJointVoltage()),
            "robot_current": float(self.rr.getActualRobotCurrent()),
            "robot_voltage": float(self.rr.getActualRobotVoltage()),
        }

    def get_forces(self) -> Dict[str, Any]:
        """Truy xuất dữ liệu lực & mô-men tiếp xúc (TCP Force/Wrench, FT raw sensor, gia tốc kế cổ tay)."""
        return {
            "actual_tcp_force": list(self.rr.getActualTCPForce()),
            "ft_raw_wrench": list(self.rr.getFtRawWrench()),
            "tool_accelerometer": list(self.rr.getActualToolAccelerometer()),
        }

    def get_temperatures(self) -> List[float]:
        """Nhiệt độ 6 động cơ khớp [°C]."""
        return list(self.rr.getJointTemperatures())

    def get_payload(self) -> Dict[str, Any]:
        """Khối lượng tải và trọng tâm CoG của dụng cụ/vật thể ở đầu ngàm."""
        return {
            "mass": float(self.rr.getPayload()),
            "cog": list(self.rr.getPayloadCog()),
        }

    def get_safety_status(self) -> Dict[str, Any]:
        """Trạng thái chế độ robot và các cờ an toàn (E-Stop, Protective Stop, Speed Scaling)."""
        return {
            "robot_mode": int(self.rr.getRobotMode()),
            "safety_mode": int(self.rr.getSafetyMode()),
            "is_emergency_stopped": bool(self.rr.isEmergencyStopped()),
            "is_protective_stopped": bool(self.rr.isProtectiveStopped()),
            "speed_scaling": float(self.rr.getSpeedScaling()),
        }

    def get_robot_state(self, streams: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Trích xuất toàn bộ gói tin Telemetry trạng thái robot chuẩn hóa.
        Hỗ trợ cờ lọc streams để tùy chọn chỉ lấy các nhóm trường cần thiết.
        """
        t_host_ns = time.perf_counter_ns()
        t_host_ms = int(time.time() * 1000)
        t_robot = float(self.rr.getTimestamp())

        state: Dict[str, Any] = {
            "timestamp_host_ns": t_host_ns,
            "timestamp_host_ms": t_host_ms,
            "timestamp_robot": t_robot,
        }

        # Nếu không truyền filter -> lấy toàn bộ
        filter_kin = True if streams is None else streams.get("robot_kinematics", True)
        filter_dyn = True if streams is None else streams.get("robot_dynamics", True)
        filter_frc = True if streams is None else streams.get("robot_contact_wrench", True)
        filter_tmp = True if streams is None else streams.get("robot_temperatures", True)
        filter_pld = True if streams is None else streams.get("robot_payload", True)
        filter_sft = True if streams is None else streams.get("robot_safety_status", True)

        if filter_kin:
            state["actual_q"] = list(self.rr.getActualQ())
            state["actual_qd"] = list(self.rr.getActualQd())
            state["actual_tcp_pose"] = list(self.rr.getActualTCPPose())
            state["actual_tcp_speed"] = list(self.rr.getActualTCPSpeed())
            state["target_q"] = list(self.rr.getTargetQ())
            state["target_qd"] = list(self.rr.getTargetQd())
            state["target_qdd"] = list(self.rr.getTargetQdd())
            state["target_tcp_pose"] = list(self.rr.getTargetTCPPose())
            state["target_tcp_speed"] = list(self.rr.getTargetTCPSpeed())

        if filter_dyn:
            state["actual_current"] = list(self.rr.getActualCurrent())
            state["target_current"] = list(self.rr.getTargetCurrent())
            state["target_moment"] = list(self.rr.getTargetMoment())
            state["joint_voltages"] = list(self.rr.getActualJointVoltage())
            state["robot_current"] = float(self.rr.getActualRobotCurrent())
            state["robot_voltage"] = float(self.rr.getActualRobotVoltage())

        if filter_frc:
            state["actual_tcp_force"] = list(self.rr.getActualTCPForce())
            state["ft_raw_wrench"] = list(self.rr.getFtRawWrench())
            state["tool_accelerometer"] = list(self.rr.getActualToolAccelerometer())

        if filter_tmp:
            state["joint_temperatures"] = list(self.rr.getJointTemperatures())

        if filter_pld:
            state["payload"] = float(self.rr.getPayload())
            state["payload_cog"] = list(self.rr.getPayloadCog())

        if filter_sft:
            state["robot_mode"] = int(self.rr.getRobotMode())
            state["safety_mode"] = int(self.rr.getSafetyMode())
            state["is_emergency_stopped"] = bool(self.rr.isEmergencyStopped())
            state["is_protective_stopped"] = bool(self.rr.isProtectiveStopped())
            state["speed_scaling"] = float(self.rr.getSpeedScaling())

        return state

    def close(self):
        self.rr.disconnect()
        print("[Setup] Disconnected")


