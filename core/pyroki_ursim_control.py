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
from typing import Optional

# -- PyRoki CPU Kinematics ----------------------------
from teleop.pyroki_solver import PyrokiIKSolver, normalize_to_current, validate_joints, UR3E_LIMITS

# -- ur-rtde -----------------------------------------------------------------
import rtde_receive

# ===========================================================================
# Configuration
def get_robot_ip() -> str:
    return os.environ.get("ROBOT_IP", "127.0.0.1")

_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_CONFIG    = os.path.join(_BASE_DIR, "assets", "ur3e_custom.yml")
TCP_LINK        = "tool0"          # End-effector link defined in ur3e.yml

SECONDARY_PORT  = 30002            # URScript secondary interface
DASHBOARD_PORT  = 29999            # Dashboard server

JOINT_ACCEL     = 0.8              # rad/s^2
JOINT_SPEED     = 0.5              # rad/s
BLEND_RADIUS    = 0.0              # 0 = stop at each waypoint

IK_RETURN_SEEDS = 5                # Number of IK solutions to evaluate

# UR3e joint limits — CONSERVATIVE bounds (±π) to avoid protective stops.
# URSim's soft limits trigger protective stop for large joint angles even when
# within URDF range (±2π). Clamping to ±180° keeps motion safe and predictable.
UR3E_LIMITS = [
    (-2 * math.pi, 2 * math.pi),  # shoulder_pan_joint   — ±360°
    (-math.pi, 0.25),             # shoulder_lift_joint  — [-180°, ~14°] vươn tự nhiên, chống chúi lật bàn
    (-2.80, 2.80),                # elbow_joint          — Cho phép gập sâu tự nhiên (-160° đến 160°)
    (-2 * math.pi, 2 * math.pi),  # wrist_1_joint        — Cho phép quay tự nhiên tránh đảo nhánh IK gây vặn xoắn
    (-2 * math.pi, 2 * math.pi),  # wrist_2_joint        — Cho phép quay tự nhiên tránh đảo nhánh IK gây vặn xoắn
    (-4 * math.pi, 4 * math.pi),  # wrist_3_joint        — ±720°
]

# ===========================================================================
# Joint normalisation utilities
# ===========================================================================

def _nearest_equivalent(angle: float, current: float, lo: float, hi: float) -> float:
    """Return the angle equivalent (mod 2*pi) that is within [lo, hi] and
    closest to *current*. Falls back to the raw angle if no equivalent fits."""
    best, best_dist = angle, float("inf")
    for k in range(-3, 4):
        candidate = angle + k * 2 * math.pi
        if lo <= candidate <= hi:
            d = abs(candidate - current)
            if d < best_dist:
                best_dist, best = d, candidate
    return best


def normalize_to_current(q_ik: list, q_current: list) -> list:
    """
    For each joint, replace the raw IK angle with the equivalent angle that
    is within the UR3e safe limits and closest to the current joint position.
    This minimises total joint travel and prevents unnecessary wrap-arounds.
    """
    return [
        _nearest_equivalent(q_ik[i], q_current[i], *UR3E_LIMITS[i])
        for i in range(6)
    ]


def validate_joints(q: list) -> bool:
    """Return True if all joints are within UR3E_LIMITS. Rejects unsafe solutions."""
    for i, (lo, hi) in enumerate(UR3E_LIMITS):
        if not (lo <= q[i] <= hi):
            return False
    return True


# ===========================================================================
# URSim helper functions
# ===========================================================================

def setup_ursim(timeout: float = 35.0) -> bool:
    """Wait for URControl to be ready, then power-on robot and release brakes via Dashboard. Returns True when RUNNING."""
    start_time = time.time()
    print("[URSim] Đang kiểm tra trạng thái robot qua Dashboard (Port 29999)...")
    print("[URSim] Giao diện Teach Pendant PolyScope (noVNC): http://192.168.56.101:6080/vnc.html")

    while time.time() - start_time < timeout:
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((get_robot_ip(), DASHBOARD_PORT))
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
            print(f"[URSim] Trạng thái: Mode={mode}, Safety={safety}")

            if "RUNNING" in mode:
                # Dừng script cũ đang chạy trên robot để giải phóng thanh ghi RTDE
                _cmd("stop", 0.2)
                # Kiểm tra chế độ Remote Control
                remote = _cmd("is in remote control", 0.3).lower()
                if "true" not in remote:
                    print("\n" + "!" * 68)
                    print("[URSim CẢNH BÁO] Robot CHƯA bật chế độ Remote Control!")
                    print("-> Hãy mở PolyScope Teach Pendant (noVNC):")
                    print("   Chọn biểu tượng Robot/Remote Control ở góc trên bên phải màn hình.")
                    print("   Chuyển từ 'Local' sang 'Remote' để robot có thể nhận lệnh RTDE.")
                    print("!" * 68 + "\n")
                s.close()
                return True

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
        except Exception as e:
            if s:
                try:
                    s.close()
                except Exception:
                    pass
            print(f"[URSim] Đang kết nối Dashboard ({e})... Thử lại sau 2 giây.")
            time.sleep(2.0)

    print("[URSim] Lỗi: Hết thời gian chờ URSim chuyển sang trạng thái RUNNING.")
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
        position: list,
        quaternion: list,
        current_q: Optional[list] = None,
    ) -> Optional[list]:
        """
        Solve IK using PyRoki. Returns joint angles [rad]*6 closest to current_q, or None.
        """
        res = self._solver.solve(pose=(position, quaternion), q_seed=current_q)
        return res.get("q")


# ===========================================================================
# High-Level Controller
# ===========================================================================

class PyrokiURSim:
    """
    PyRoki CPU IK -> URSim via URScript / RTDE.
    Hoàn toàn không yêu cầu GPU.
    """

    def __init__(self, scene_file: Optional[str] = None, robot_ip: Optional[str] = None):
        if robot_ip:
            os.environ["ROBOT_IP"] = str(robot_ip)
        print("\n" + "=" * 58)
        print("  PyRoki (CPU JIT) -> URSim Controller (No GPU Required)")
        print("=" * 58)

        if not setup_ursim():
            raise RuntimeError("Robot not RUNNING. Check URSim container.")

        print("[Setup] Connecting RTDEReceive...")
        self.rr = rtde_receive.RTDEReceiveInterface(get_robot_ip())
        q = self.rr.getActualQ()
        print(f"[Setup] Connected. Joints: {[round(j, 4) for j in q]}")

        self._ik = None
        self.scene_file = scene_file
        print("=" * 58 + "\n")

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

    def close(self):
        self.rr.disconnect()
        print("[Setup] Disconnected")

