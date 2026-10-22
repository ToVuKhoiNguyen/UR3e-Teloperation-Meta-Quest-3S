import os
import time
import socket
import threading
import numpy as np
import torch
from teleop.pyroki_solver import PyrokiIKSolver

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: PyRoki IK Solver (CPU)
# ─────────────────────────────────────────────────────────────────────────────
# Input:  pose = { "pos": [X,Y,Z], "quat": [Qw,Qx,Qy,Qz] }
#         q_real = [6 góc khớp hiện tại - Radian]
# Output: { "q": [6 Radian] hoặc None, "latency_ms": float, "solver": "pyroki" }
#
class PyrokiIKSolverWrapper:
    def __init__(self, ik_solver=None):
        if ik_solver is not None and hasattr(ik_solver, "solve") and hasattr(ik_solver, "fk"):
            self._ik = ik_solver
        elif ik_solver is not None and hasattr(ik_solver, "_solver"):
            self._ik = ik_solver._solver
        else:
            self._ik = PyrokiIKSolver()

    def solve(self, pose, q_real):
        return self._ik.solve(pose, q_seed=q_real)

    def fk(self, q: list) -> dict:
        return self._ik.fk(q)


def clamp_cartesian_safety(pose, z_min=0.035, z_max=0.60, r_min=0.14, r_max=0.52):
    """
    Rào chắn an toàn không gian Descartes (Virtual Safety Envelope).
    Giữ khoảng cách an toàn tối thiểu 3.5cm (0.035m) đối với mặt bàn (Z_min >= 0.035m),
    cách thân đế robot tối thiểu (R_min >= 0.14m bảo vệ cột trụ thân robot cách vỏ đế 7.6cm),
    và chặn vươn quá tầm với cơ học (R_max <= 0.52m).
    Độ trễ tính toán < 0.05ms.
    """
    if isinstance(pose, dict) and "pos" in pose:
        p = list(pose["pos"])
    elif isinstance(pose, (tuple, list)) and len(pose) == 2:
        p = list(pose[0])
    else:
        return pose

    # 1. Giới hạn độ cao Z: Duy trì khoảng cách tối thiểu 5cm (0.05m) so với mặt bàn
    p[2] = max(z_min, min(z_max, float(p[2])))

    # 2. Giới hạn bán kính ngang XY:
    r_xy = float(np.hypot(p[0], p[1]))
    # Cột trụ an toàn bảo vệ thân robot (R_min = 0.18m) bất khả xâm phạm ở mọi độ cao Z
    current_r_min = r_min
    if r_xy < current_r_min:
        scale = current_r_min / max(r_xy, 1e-6)
        p[0] *= scale
        p[1] *= scale
    elif r_xy > r_max:
        scale = r_max / r_xy
        p[0] *= scale
        p[1] *= scale

    if isinstance(pose, dict):
        pose_safe = dict(pose)
        pose_safe["pos"] = p
        return pose_safe
    else:
        return (p, pose[1])


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: SafetyGate
# ─────────────────────────────────────────────────────────────────────────────
# Input:  q = [6 Radian] từ IK solver
# Output: { "pass": bool, "reason": str, "sigma_min": float, "damping": float }
#
class SafetyGate:
    SHOULDER_PAN_LIMIT = 3.14   # ±180°
    SHOULDER_LIFT_MAX = 0.20    # q1 không được vượt quá +0.20 rad (~11.5 độ)
    SHOULDER_LIFT_MIN = -2.65   # q1 không được nhỏ hơn -2.65 rad (-151.8 độ)
    ELBOW_MIN = 0.10            # q2 >= +0.10 rad (+5.7 độ) - Chống gập cùi chỏ ngược xuống bàn
    ELBOW_MAX = 2.40            # q2 <= +2.40 rad (+137.5 độ) - Chống cẳng tay ép sát bắp tay
    WRIST_1_MIN = -3.20         # q3 >= -3.20 rad
    WRIST_1_MAX = 0.10          # q3 <= +0.10 rad - Chống bẻ ngược cổ tay đâm vào forearm
    WRIST_2_MIN = -2.95         # q4 >= -2.95 rad
    WRIST_2_MAX = -0.20         # q4 <= -0.20 rad
    WRIST_3_LIMIT = 3.14        # ±180°

    def __init__(self, guard):
        self._guard = guard

    def check(self, q):
        if q is None:
            return {"pass": False, "reason": "IK failed", "sigma_min": 0.0, "damping": 0.0}
        # Lớp 2: Chống chúi/ngửa bắp tay đâm bàn hoặc đâm đế robot
        if q[1] > self.SHOULDER_LIFT_MAX or q[1] < self.SHOULDER_LIFT_MIN:
            return {"pass": False, "reason": f"shoulder_lift_limit(q1={q[1]:.2f})", "sigma_min": 0.0, "damping": 0.0}
        # Lớp 4: Chống tự va chạm gập khuỷu tay (Elbow self-collision)
        if q[2] < self.ELBOW_MIN or q[2] > self.ELBOW_MAX:
            return {"pass": False, "reason": f"elbow_clamp(q2={q[2]:.2f} not in [{self.ELBOW_MIN}, {self.ELBOW_MAX}])", "sigma_min": 0.0, "damping": 0.0}
        # Lớp 5: Chống tự kẹp cổ tay và đâm ngược vào bắp tay
        if q[3] < self.WRIST_1_MIN or q[3] > self.WRIST_1_MAX:
            return {"pass": False, "reason": f"wrist1_clamp(q3={q[3]:.2f} not in [{self.WRIST_1_MIN}, {self.WRIST_1_MAX}])", "sigma_min": 0.0, "damping": 0.0}
        if q[4] < self.WRIST_2_MIN or q[4] > self.WRIST_2_MAX:
            return {"pass": False, "reason": f"wrist2_clamp(q4={q[4]:.2f} not in [{self.WRIST_2_MIN}, {self.WRIST_2_MAX}])", "sigma_min": 0.0, "damping": 0.0}
        if abs(q[0]) > self.SHOULDER_PAN_LIMIT or abs(q[5]) > self.WRIST_3_LIMIT:
            return {"pass": False, "reason": "pan_or_wrist3_limit", "sigma_min": 0.0, "damping": 0.0}
        # Lớp 1: Kiểm tra kỳ dị
        m = self._guard.metrics_calc.get_metrics(
            torch.tensor([q], device=self._guard.metrics_calc.device)
        )
        sigma = m["sigma_min"].item()
        if sigma < self._guard.danger_threshold:
            return {"pass": False, "reason": f"singularity(σ={sigma:.4f})", "sigma_min": sigma, "damping": 0.0}

        # Continuous Singularity Damping: Khi tiến gần biên kỳ dị (danger < sigma < warn),
        # tự động hãm vận tốc mượt mà thay vì phanh chết đột ngột
        damping = 1.0
        if sigma < self._guard.warn_threshold:
            # Hermite smoothstep từ 0.45 (khi sát danger) tới 1.0 (khi chạm warn)
            t = (sigma - self._guard.danger_threshold) / max(self._guard.warn_threshold - self._guard.danger_threshold, 1e-4)
            t = max(0.0, min(1.0, t))
            damping = 0.45 + 0.55 * (t * t * (3.0 - 2.0 * t))

        return {"pass": True, "reason": "ok", "sigma_min": sigma, "damping": damping}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: ServoJ Executor  (dùng RTDEControlInterface - ĐÚNG cách)
# ─────────────────────────────────────────────────────────────────────────────
class ServoJExecutor:
    ROBOT_IP = os.environ.get("ROBOT_IP", "127.0.0.1")

    def __init__(self, dt=0.05, lookahead_time=0.12, gain=250, robot_ip: str = None):
        self._dt = dt
        self._lookahead = float(os.environ.get("SERVO_LOOKAHEAD", str(lookahead_time)))
        self._gain = int(os.environ.get("SERVO_GAIN", str(gain)))
        if robot_ip:
            self.ROBOT_IP = robot_ip
        self._ctrl = None
        self._last_warn_time = 0.0
        self._lock = threading.Lock()

    def connect(self):
        """Khởi tạo RTDEControlInterface. Gọi 1 lần khi bật Teleop."""
        with self._lock:
            import rtde_control
            if self._ctrl is not None:
                try:
                    if self._ctrl.isConnected():
                        return
                except Exception:
                    pass

            robot_ip = os.environ.get("ROBOT_IP", self.ROBOT_IP)

            # Dừng script cũ trên robot qua Dashboard (Port 29999) để giải phóng hoàn toàn thanh ghi RTDE
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((robot_ip, 29999))
                s.recv(1024)
                s.sendall(b"stop\n")
                time.sleep(0.1)
                s.recv(1024)
                s.close()
            except Exception:
                pass

            try:
                self._ctrl = rtde_control.RTDEControlInterface(robot_ip)
                print(f"[ServoJ] RTDEControl connected: {self._ctrl.isConnected()}")
            except Exception as e:
                err_msg = str(e).lower()
                if "input registers are already in use" in err_msg or "fieldbus" in err_msg:
                    print("[ServoJ] Xung đột thanh ghi, thử dải Upper Range Registers...")
                    try:
                        flags_upper = int(rtde_control.RTDEControlInterface.FLAG_UPLOAD_SCRIPT) | int(rtde_control.RTDEControlInterface.FLAG_UPPER_RANGE_REGISTERS)
                        self._ctrl = rtde_control.RTDEControlInterface(robot_ip, -1.0, flags_upper)
                        print(f"[ServoJ] RTDEControl connected (Upper Range Registers): {self._ctrl.isConnected()}")
                    except Exception as inner_e:
                        print(f"[ServoJ] Không thể kết nối RTDE Control: {inner_e}")
                        self._ctrl = None
                else:
                    print(f"[ServoJ] Lỗi kết nối RTDE: {e}")
                    if "remote control" in err_msg:
                        print("[Gợi ý] Hãy bật Remote Control trên PolyScope (góc trên phải màn hình Teach Pendant).")
                    self._ctrl = None

    def disconnect(self):
        """Dừng servoJ và ngắt kết nối. Gọi 1 lần khi tắt Teleop."""
        with self._lock:
            if self._ctrl:
                try:
                    self._ctrl.servoStop()
                    self._ctrl.stopScript()
                    self._ctrl.disconnect()
                except Exception:
                    pass
                self._ctrl = None
                print("[ServoJ] RTDEControl disconnected.")

    def send(self, q):
        """
        Input:  q list[6 Radian]
        Output: None

        RTDEControlInterface.servoJ() gọi trực tiếp hàm servoj bên trong
        bộ điều khiển UR, không cần gửi script URScript qua port 30002.
        Tham số:
          speed=0, acceleration=0 → robot tự tính dựa theo lookahead_time
          dt           = chu kỳ lấy mẫu (s), khớp với vòng lặp 20Hz
          lookahead_time = nội suy mượt (0.03–0.2s)
          gain         = độ cứng bám vị trí (100–2000)
        """
        with self._lock:
            if self._ctrl is None:
                now = time.time()
                if now - self._last_warn_time >= 1.0:
                    print("[ServoJ CẢNH BÁO] Chưa kết nối RTDEControlInterface (_ctrl is None) - không thể gửi lệnh tới robot! Hãy đảm bảo đã bật Remote Control trên PolyScope.")
                    self._last_warn_time = now
                return
            try:
                # RTDEControlInterface C++ bindings only support positional arguments
                self._ctrl.servoJ(
                    q,
                    0.0,               # speed
                    0.0,               # acceleration
                    self._dt,          # dt
                    self._lookahead,   # lookahead_time
                    self._gain         # gain
                )
            except Exception as e:
                pass

    def stop(self, a=10.0):
        """Dừng servo mode và phanh robot ngay lập tức (mặc định a=10.0 m/s^2)."""
        with self._lock:
            if self._ctrl:
                try:
                    self._ctrl.servoStop(a)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: TeleoperationController (Điều phối tổng)
# ─────────────────────────────────────────────────────────────────────────────
# Giao diện với main loop:
#   - teleop.enabled = True/False  (toggle từ Web button)
#   - teleop.step(q_real)          (gọi mỗi vòng lặp 20Hz)
#   - teleop.get_status()          → string hiển thị lên UI
#   - teleop.get_latency()         → dict latency PyRoki
#
class TeleoperationController:
    def __init__(self, pose_source, pyroki_solver, safety_gate, executor, dt=0.05):
        self._pose   = pose_source
        self._solver = pyroki_solver
        self._gate   = safety_gate
        self._exec   = executor
        self._dt     = dt

        self._enabled = False
        self._last_valid_q = None
        self._status  = "Đã tắt"
        self._latency = {"pyroki_ms": 0.0}
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        """Bật/tắt Teleop: tự động mở/đóng RTDEControl."""
        if value and not self._enabled:
            self._exec.connect()
            self._status = "Chờ tín hiệu tay cầm..."
            self._last_valid_q = None
        elif not value and self._enabled:
            self._exec.disconnect()
            self._status = "Đã tắt"
        self._enabled = value

    def stop(self, a=10.0):
        """Dừng robot khẩn cấp hoặc khi nhả phím điều khiển."""
        self._exec.stop(a)

    def get_status(self):
        with self._lock:
            return self._status

    def get_latency(self):
        with self._lock:
            return dict(self._latency)

    def step(self, q_real, pose=None):
        """
        Một vòng lặp 20Hz.

        Input:  q_real list[6 Radian] - góc khớp thực tế từ RTDE Port 30004
                pose dict (tùy chọn) - pose mục tiêu từ PoseSource đã lấy sẵn
        Output: None (side-effect: gọi servoJ hoặc đứng im)
        """
        if not self._enabled:
            return

        # ── Lấy pose mới từ PoseSource và áp dụng Hàng rào ảo 5cm ─────────────
        if pose is None:
            pose = self._pose.get_pose()
        if pose is None:
            self._exec.stop()
            return  # pose không đổi / không ấn phím → dừng servoJ để robot đứng yên lập tức

        # Tầng 1: Rào chắn an toàn không gian Descartes (Cách mặt bàn/vật cản >= 3.5cm, cách thân đế >= 14cm, r_max <= 52cm)
        pose_safe = clamp_cartesian_safety(pose, z_min=0.035, z_max=0.60, r_min=0.14, r_max=0.52)

        # ── Giải IK bằng PyRoki CPU ─────────────────────────────────────────
        q_seed = self._last_valid_q if self._last_valid_q is not None else q_real
        result_ik = self._solver.solve(pose_safe, q_seed)

        with self._lock:
            self._latency["pyroki_ms"] = result_ik["latency_ms"]

        # ── Kiểm tra nghiệm IK PyRoki ────────────────────────────────────────
        if result_ik["q"] is not None:
            q_target = result_ik["q"]
            # Kiểm tra bước nhảy nhánh IK bất thường (phát hiện lật khớp do gần điểm kỳ dị Singularity)
            if self._last_valid_q is not None:
                max_jump = max(abs(q_target[i] - self._last_valid_q[i]) for i in range(6))
                if max_jump > 2.2:  # ~126 độ (chỉ chặn khi xảy ra đảo nhánh IK đột ngột)
                    with self._lock:
                        self._status = "Tư thế kẹt (Lỗi vặn xoắn khớp) - Dừng im"
                    self._exec.stop()
                    return
                
                # Scaled Vector Rate Limiter: Co giãn đồng thời cả 6 khớp theo cùng một tỷ lệ
                # để bảo toàn 100% hướng vector chuyển động trong không gian Descartes (không làm méo lệch quỹ đạo)
                # Mặc định 1.0 rad/s (~57.3 độ/s) cho phản hồi nhanh nhạy, bắt kịp chuyển động tay
                MAX_VEL = float(os.environ.get("MAX_JOINT_VEL", "1.0"))
                DT = self._dt
                max_dq = MAX_VEL * DT
                dq = [q_target[i] - self._last_valid_q[i] for i in range(6)]
                max_ratio = max(abs(dq[i]) / max_dq for i in range(6))
                if max_ratio > 1.0:
                    q_target = [self._last_valid_q[i] + dq[i] / max_ratio for i in range(6)]
        else:
            # Phân tách chính xác nguyên nhân không tìm được nghiệm IK
            pos = pose_safe["pos"] if isinstance(pose_safe, dict) else pose_safe[0]
            dist_base = (pos[0]**2 + pos[1]**2 + (pos[2] - 0.15)**2)**0.5
            with self._lock:
                if dist_base > 0.535:
                    self._status = "Ngoài tầm với (Out of Reach) - Dừng im"
                elif pos[2] <= 0.038:
                    self._status = "Cách vật cản/mặt bàn ≤ 3.5cm - Dừng an toàn"
                else:
                    self._status = "Tư thế không khả thi / Vướng vật cản - Dừng im"
            self._exec.stop()
            return

        # ── Kiểm tra SafetyGate (Tầng 2 & Tầng 3) ───────────────────────────
        gate = self._gate.check(q_target)
        if not gate["pass"]:
            with self._lock:
                self._status = f"Dừng an toàn - {gate['reason']}"
            self._exec.stop()
            return

        # Continuous Singularity Damping: Hãm mượt vận tốc khi lướt qua vùng lân cận kỳ dị
        damping = gate.get("damping", 1.0)
        if damping < 0.99 and self._last_valid_q is not None:
            q_target = [self._last_valid_q[i] + (q_target[i] - self._last_valid_q[i]) * damping for i in range(6)]

        # Kiểm định lại bằng Forward Kinematics (FK) để đảm bảo TCP/Camera cách mặt bàn ít nhất 3.5cm
        try:
            fk_chk = self._solver.fk(q_target)
            if fk_chk and "pos" in fk_chk and fk_chk["pos"][2] < 0.035:
                with self._lock:
                    self._status = f"Chặn FK bàn máy: Z={fk_chk['pos'][2]*1000:.0f}mm < 35mm"
                self._exec.stop()
                return
        except Exception:
            pass

        # ── Gọi RTDEControl.servoJ ───────────────────────────────────────────
        self._exec.send(q_target)
        self._last_valid_q = q_target

        sigma = gate["sigma_min"]
        damp_str = f" | damp={damping:.2f}" if damping < 0.99 else ""
        with self._lock:
            self._status = (
                f"[PyRoki] | "
                f"sigma={sigma:.3f}{damp_str} | "
                f"IK:{result_ik['latency_ms']:.1f}ms"
            )
