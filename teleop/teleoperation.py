"""
teleoperation.py - UR3e Teleoperation Control Pipeline with PyRoki IK Solver.
Runs entirely on CPU with zero GPU / NVIDIA CUDA dependencies.

Features:
  - 1cm Uniform Armor (WYSIWYG) Collision Protection
  - Constrained Surface Sliding & Dynamic Barrier Avoidance
  - 6-DoF Elastic Tracking Leash (Position 250mm, Orientation 1.2 rad)
  - Real-time Physical Singularity Detection (Wrist, Elbow, Shoulder)
  - Ultra-responsive ServoJ Control (lookahead 0.04s, gain 500)
  - Standardized English Telemetry Status Formatting
"""

import os
import sys
import time
import math
import socket
import threading
from typing import Optional, Union, Dict, Any, List

import numpy as np
import torch
from teleop.pyroki_solver import UR3E_LIMITS


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Pyroki IK Solver Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class PyrokiIKSolverWrapper:
    """Wrapper around PyrokiIKSolver to match standard IK Solver interface."""

    def __init__(self, solver=None, scene_path: Optional[str] = None):
        if solver is not None:
            self._solver = solver
        else:
            from teleop.pyroki_solver import PyrokiIKSolver
            self._solver = PyrokiIKSolver(scene_path=scene_path)

    def solve(
        self,
        pose: Union[tuple, list, Dict[str, Any]],
        q_seed: Optional[List[float]] = None,
        arm_anchor: Optional[Union[List[float], np.ndarray]] = None,
        is_pure_rot: bool = False,
    ) -> Dict[str, Any]:
        return self._solver.solve(pose, q_seed, arm_anchor=arm_anchor, is_pure_rot=is_pure_rot)

    def fk(self, q: List[float]) -> Dict[str, Any]:
        return self._solver.fk(q)

    def __getattr__(self, name: str):
        return getattr(self._solver, name)


def clamp_cartesian_safety(
    pose: Union[dict, tuple],
    z_min: float = 0.020,
    z_max: float = 0.60,
    r_min: float = 0.0,
    r_max: float = 0.52
) -> Union[dict, tuple]:
    """
    Cartesian Safety Envelope.
    Maintains safe clearance from table armor top surface at Z=+0.020m (Z_table_armor, WYSIWYG),
    and clamps reaches beyond mechanical limits (R_max <= 0.52m).
    Computation latency < 0.05ms.
    """
    if isinstance(pose, dict) and "pos" in pose:
        p = list(pose["pos"])
    elif isinstance(pose, (tuple, list)) and len(pose) == 2:
        p = list(pose[0])
    else:
        return pose

    # 1. Z height clamping: Maintain minimum clearance above table armor (Z_min >= 0.015m)
    p[2] = max(z_min, min(z_max, float(p[2])))

    # 2. Horizontal XY radius clamping:
    r_xy = float(np.hypot(p[0], p[1]))
    if r_min > 0.0 and r_xy < r_min:
        scale = r_min / max(r_xy, 1e-6)
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
class SafetyGate:
    """
    Multi-layer safety validation gate for joint configurations:
      - Layer 1: SingularityGuard sigma check
      - Layer 2: Shoulder lift safe envelope
      - Layer 3: Elbow clamp to prevent folding downward into base/table
      - Layer 4: Wrist travel limits
      - Layer 5: Strict 1cm Uniform Armor Self-Collision check (WYSIWYG 0.0mm)
      - Layer 6: Table Armor Penetration check (WYSIWYG 0.0mm)
    """
    TOL = 0.02
    SHOULDER_PAN_LIMIT = math.pi   # ±180°
    SHOULDER_LIFT_MAX = 0.20    # q1 max +0.20 rad (+11.5 deg)
    SHOULDER_LIFT_MIN = -2.65   # q1 min -2.65 rad (-151.8 deg)
    ELBOW_MIN = -0.08           # q2 >= -0.08 rad (-4.6 deg, allows straight arm down to -0.86°)
    ELBOW_MAX = 3.00            # q2 <= +3.00 rad (+171.8 deg)
    WRIST_1_MIN = -math.pi         # ±180°
    WRIST_1_MAX = math.pi
    WRIST_2_MIN = -math.pi         # ±180°
    WRIST_2_MAX = math.pi
    WRIST_3_LIMIT = math.pi        # ±180°

    def __init__(self, guard, solver=None):
        self._guard = guard
        if hasattr(solver, "_ik"):
            self._solver = solver._ik
        elif hasattr(solver, "_solver"):
            self._solver = solver._solver
        else:
            self._solver = solver

    def check(self, q: Optional[List[float]]) -> Dict[str, Any]:
        if q is None:
            return {"pass": False, "reason": "IK failed", "sigma_min": 0.0, "damping": 0.0}

        # Shoulder lift safe limits
        if q[1] > self.SHOULDER_LIFT_MAX + self.TOL or q[1] < self.SHOULDER_LIFT_MIN - self.TOL:
            return {"pass": False, "reason": f"shoulder_lift_limit(q1={q[1]:.2f})", "sigma_min": 0.0, "damping": 0.0}

        # Elbow folding clamp
        if q[2] < self.ELBOW_MIN - self.TOL or q[2] > self.ELBOW_MAX + self.TOL:
            return {"pass": False, "reason": f"elbow_clamp(q2={q[2]:.2f} not in [{self.ELBOW_MIN}, {self.ELBOW_MAX}])", "sigma_min": 0.0, "damping": 0.0}

        # Wrist angle limits
        if q[3] < self.WRIST_1_MIN - self.TOL or q[3] > self.WRIST_1_MAX + self.TOL:
            return {"pass": False, "reason": f"wrist1_clamp(q3={q[3]:.2f} not in [{self.WRIST_1_MIN}, {self.WRIST_1_MAX}])", "sigma_min": 0.0, "damping": 0.0}
        if q[4] < self.WRIST_2_MIN - self.TOL or q[4] > self.WRIST_2_MAX + self.TOL:
            return {"pass": False, "reason": f"wrist2_clamp(q4={q[4]:.2f} not in [{self.WRIST_2_MIN}, {self.WRIST_2_MAX}])", "sigma_min": 0.0, "damping": 0.0}
        if abs(q[0]) > self.SHOULDER_PAN_LIMIT + self.TOL or abs(q[5]) > self.WRIST_3_LIMIT + self.TOL:
            return {"pass": False, "reason": "pan_or_wrist3_limit", "sigma_min": 0.0, "damping": 0.0}

        # Strict Self-collision check with 1cm armor envelope (WYSIWYG 0.0mm threshold)
        if self._solver is not None:
            disable_coll = getattr(self._solver, "disable_self_coll", False) or (
                os.environ.get("DISABLE_SELF_COLLISION", "0").lower() in ("1", "true", "yes")
            )
            if not disable_coll:
                chk_func = getattr(self._solver, "check_self_collision", getattr(self._solver, "check_camera_self_collision", None))
                if chk_func is not None:
                    try:
                        coll_thresh = getattr(self._solver, "clearance_threshold", 0.01)
                        coll_chk = chk_func(q, clearance_threshold=coll_thresh)
                        if not coll_chk["safe"]:
                            link_a = coll_chk.get("link_a", "link_a")
                            link_b = coll_chk.get("link_b", "link_b")
                            return {
                                "pass": False,
                                "reason": f"self_collision({link_a} <-> {link_b}: {coll_chk['min_dist']*1000:.1f}mm <= {coll_thresh*1000:.1f}mm)",
                                "sigma_min": 0.0,
                                "damping": 0.0
                            }
                    except Exception:
                        pass

        # Strict Table armor check (WYSIWYG 0.0mm threshold)
        if self._solver is not None and hasattr(self._solver, "check_table_collision"):
            try:
                tbl_chk = self._solver.check_table_collision(q, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
                if not tbl_chk["safe"]:
                    return {
                        "pass": False,
                        "reason": f"table_collision({tbl_chk['violator']} <-> Table Armor: {tbl_chk['clearance_cm']*10:.1f}mm <= 0.0mm)",
                        "sigma_min": 0.0,
                        "damping": 0.0
                    }
            except Exception:
                pass

        # Singularity Guard sigma check: Smooth continuous damping instead of hard binary cutoff
        sigma = 1.0
        damping = 1.0
        if hasattr(self, "_guard") and self._guard is not None and hasattr(self._guard, "metrics_calc"):
            try:
                m = self._guard.metrics_calc.get_metrics(
                    torch.tensor([q], device=self._guard.metrics_calc.device)
                )
                sigma = m["sigma_min"].item()
                if sigma < self._guard.warn_threshold:
                    t = (sigma - 0.005) / max(self._guard.warn_threshold - 0.005, 1e-4)
                    t = max(0.0, min(1.0, t))
                    damping = 0.70 + 0.30 * (t * t * (3.0 - 2.0 * t))
            except Exception:
                pass

        return {"pass": True, "reason": "ok", "sigma_min": sigma, "damping": damping}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: ServoJ Executor (RTDEControlInterface)
# ─────────────────────────────────────────────────────────────────────────────
class ServoJExecutor:
    """
    High-performance real-time servoJ executor via RTDE.
    Optimized for smooth tracking: lookahead 0.09s, gain 300.
    """
    ROBOT_IP = os.environ.get("ROBOT_IP", "127.0.0.1")

    def __init__(self, dt=0.05, lookahead_time=0.11, gain=250, robot_ip: str = None):
        self._dt = dt
        self._lookahead = float(os.environ.get("SERVO_LOOKAHEAD", str(lookahead_time)))
        self._gain = int(os.environ.get("SERVO_GAIN", str(gain)))
        if robot_ip:
            self.ROBOT_IP = robot_ip
        self._ctrl = None
        self._last_warn_time = 0.0
        self._lock = threading.Lock()

    def connect(self):
        """Initialize RTDEControlInterface. Called once when enabling Teleop."""
        with self._lock:
            import rtde_control
            if self._ctrl is not None:
                try:
                    if self._ctrl.isConnected():
                        return
                except Exception:
                    pass

            robot_ip = os.environ.get("ROBOT_IP", self.ROBOT_IP)

            # Terminate previous script via Dashboard (Port 29999) to free RTDE registers
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
                    print("[ServoJ] Register conflict detected, trying Upper Range Registers...")
                    try:
                        flags_upper = int(rtde_control.RTDEControlInterface.FLAG_UPLOAD_SCRIPT) | int(rtde_control.RTDEControlInterface.FLAG_UPPER_RANGE_REGISTERS)
                        self._ctrl = rtde_control.RTDEControlInterface(robot_ip, -1.0, flags_upper)
                        print(f"[ServoJ] RTDEControl connected (Upper Range Registers): {self._ctrl.isConnected()}")
                    except Exception as inner_e:
                        print(f"[ServoJ] Failed to connect RTDE Control: {inner_e}")
                        self._ctrl = None
                else:
                    print(f"[ServoJ] RTDE connection error: {e}")
                    if "remote control" in err_msg:
                        print("[Hint] Please enable Remote Control on PolyScope (top-right of Teach Pendant screen).")
                    self._ctrl = None

    def disconnect(self):
        """Stop servoJ and disconnect."""
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

    def send(self, q: List[float]):
        """
        Send joint command to robot controller via servoJ.
        Parameters:
          speed=0, acceleration=0 (internally calculated by UR controller based on lookahead_time)
          dt             = control cycle (s)
          lookahead_time = trajectory filter (0.03-0.05s for high responsiveness)
          gain           = tracking stiffness (500-600)
        """
        with self._lock:
            if self._ctrl is None:
                now = time.time()
                if now - self._last_warn_time >= 1.0:
                    print("[ServoJ WARNING] RTDEControlInterface not connected (_ctrl is None) - cannot send command to robot! Please ensure Remote Control is enabled on PolyScope.")
                    self._last_warn_time = now
                return
            try:
                self._ctrl.servoJ(
                    q,
                    0.0,               # speed
                    0.0,               # acceleration
                    self._dt,          # dt
                    self._lookahead,   # lookahead_time
                    self._gain         # gain
                )
            except Exception:
                pass

    def stop(self, a=10.0):
        """
        Explicit emergency stop.
        NOTE: Never called inside normal teleoperation loop to prevent abrupt freezing.
        """
        with self._lock:
            if self._ctrl is None:
                return
            try:
                self._ctrl.servoStop(a)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Teleoperation Controller
_DEFAULT_FETCH_POSE = object()


class TeleoperationController:
    """
    Main 20Hz teleoperation loop coordinating PoseSource, PyRoki IK, SafetyGate, and ServoJ.
    Implements 6-DoF Elastic Tracking (Position 250mm, Orientation 1.2 rad)
    and standardized English telemetry status formatting.
    """

    def __init__(
        self,
        pose_source,
        solver=None,
        gate: Optional[SafetyGate] = None,
        executor: Optional[ServoJExecutor] = None,
        dt: float = 0.05,
        pyroki_solver=None,
        safety_gate=None
    ):
        self._pose = pose_source
        self._solver = solver if solver is not None else pyroki_solver
        self._gate = gate if gate is not None else safety_gate
        self._exec = executor
        self._dt = dt

        self._enabled = False
        self._last_valid_q = None
        self._status = "OK"
        self._last_logged_status = None
        self._latency = {"pyroki_ms": 0.0, "total_ms": 0.0}
        self._lock = threading.Lock()

        # Second-order smooth tracking states
        self._smoothed_pos: Optional[np.ndarray] = None
        self._smoothed_quat: Optional[np.ndarray] = None
        self._prev_dq: np.ndarray = np.zeros(6, dtype=float)
        self._was_clutch_released: bool = True
        self._arm_anchor_q: Optional[np.ndarray] = None
        self._arm_anchor_p: Optional[np.ndarray] = None
        self._rot_anchor_body_q: Optional[np.ndarray] = None

        # Diagnostic block logging throttling
        self._last_block_tag = None
        self._last_block_detail = None
        self._last_block_log_time = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val
        if val:
            if self._exec is not None:
                self._exec.connect()
            if self._last_valid_q is not None and self._solver is not None and self._smoothed_pos is None:
                try:
                    fk = self._solver.fk(self._last_valid_q)
                    self._smoothed_pos = np.array(fk["pos"], dtype=float)
                    self._smoothed_quat = np.array(fk["quat"], dtype=float)
                except Exception:
                    pass
            self._prev_dq = np.zeros(6, dtype=float)
            with self._lock:
                self._status = "OK"
                self._last_logged_status = "OK"
        else:
            if self._exec is not None:
                self._exec.disconnect()
            self._smoothed_pos = None
            self._smoothed_quat = None
            self._prev_dq = np.zeros(6, dtype=float)
            with self._lock:
                self._status = "OK"
                self._last_logged_status = "OK"

    def sync_with_robot(self, q_init=None):
        """Reset smoothers and align with current robot joint positions."""
        self._was_clutch_released = True
        self._arm_anchor_q = None
        self._arm_anchor_p = None
        self._rot_anchor_body_q = None
        self._prev_dq = np.zeros(6, dtype=float)
        with self._lock:
            self._status = "OK"
            self._last_logged_status = "OK"
        if q_init is not None:
            self._last_valid_q = list(q_init)
            try:
                fk = self._solver.fk(q_init)
                self._smoothed_pos = np.array(fk["pos"], dtype=float)
                self._smoothed_quat = np.array(fk["quat"], dtype=float)
            except Exception:
                self._smoothed_pos = None
                self._smoothed_quat = None
        else:
            self._smoothed_pos = None
            self._smoothed_quat = None

    def notify_clutch_released(self, q_current: Optional[List[float]] = None):
        """
        Marks clutch as released. Syncs internal smoothed pose with current robot TCP
        to ensure the virtual target is ready at the TCP for the next clutch engage.
        """
        self._was_clutch_released = True
        self._arm_anchor_q = None
        self._arm_anchor_p = None
        self._rot_anchor_body_q = None
        q_ref = q_current if q_current is not None else self._last_valid_q
        if q_ref is not None:
            self._last_valid_q = list(q_ref)
            if self._solver is not None:
                try:
                    fk = self._solver.fk(q_ref)
                    self._smoothed_pos = np.array(fk["pos"], dtype=float)
                    self._smoothed_quat = np.array(fk["quat"], dtype=float)
                except Exception:
                    pass
        self._prev_dq = np.zeros(6, dtype=float)
        self._smoothed_damping = 1.0

    def stop(self, a: float = 10.0, *args, **kwargs):
        """Stop teleoperation and halt robot motion with deceleration limit."""
        self.enabled = False
        if hasattr(self, "_exec") and self._exec is not None:
            try:
                self._exec.stop(a)
            except Exception:
                pass

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def get_latency(self) -> dict:
        with self._lock:
            return dict(self._latency)

    def log_diagnostic_block(
        self,
        block_tag: str,
        detail_reason: str = "",
        q_eval: list = None,
        ik_result: Optional[dict] = None,
        pose_cmd: Optional[Any] = None
    ):
        """
        Log ultra-concise status to Terminal only when the event status changes.
        Eliminates verbose log dumps and prevents 20Hz loop lag.
        """
        if block_tag != self._last_logged_status:
            self._last_logged_status = block_tag
            self._last_block_tag = block_tag
            print(f"[Teleop Status] {block_tag}")

    def _apply_anti_windup_reanchor(self, p_curr: np.ndarray, q_curr_wxyz: np.ndarray):
        """
        Anti-windup re-anchoring when touching limits or barriers:
        For relative delta sources (VR controller), clamps smoothed_target to within 30mm position
        and 0.25 rad orientation of current TCP, and re-anchors the delta pose source to prevent target runaway.
        For absolute sources (Viser Gizmo), does not clamp smoothed_target to prevent drift from ui.gizmo.
        """
        is_absolute = getattr(self._pose, "is_absolute", False)
        if not is_absolute:
            if self._smoothed_pos is not None:
                vec_p = self._smoothed_pos - p_curr
                dist_p = float(np.linalg.norm(vec_p))
                MAX_REANCHOR_DIST = 0.030  # 30mm
                if dist_p > MAX_REANCHOR_DIST:
                    self._smoothed_pos = p_curr + (vec_p / dist_p) * MAX_REANCHOR_DIST

            if self._smoothed_quat is not None:
                q_s = self._smoothed_quat / max(1e-6, np.linalg.norm(self._smoothed_quat))
                q_c = q_curr_wxyz / max(1e-6, np.linalg.norm(q_curr_wxyz))
                dot_q = float(np.dot(q_s, q_c))
                if dot_q < 0.0:
                    q_c = -q_c
                    dot_q = -dot_q
                dot_q = min(1.0, max(-1.0, dot_q))
                theta = float(2.0 * math.acos(dot_q))
                MAX_REANCHOR_THETA = 0.25  # rad (~14.3 deg)
                if theta > MAX_REANCHOR_THETA and math.sin(theta / 2.0) > 1e-4:
                    t_slerp = MAX_REANCHOR_THETA / theta
                    sin_th = math.sin(theta / 2.0)
                    w1 = math.sin((1.0 - t_slerp) * theta / 2.0) / sin_th
                    w2 = math.sin(t_slerp * theta / 2.0) / sin_th
                    self._smoothed_quat = w1 * q_c + w2 * q_s
                    self._smoothed_quat /= np.linalg.norm(self._smoothed_quat)

        if self._pose is not None and hasattr(self._pose, "reanchor_position"):
            try:
                self._pose.reanchor_position(p_curr)
            except Exception:
                pass

    def _apply_smooth_braking_and_send(
        self,
        status_tag: str,
        detail: str = "",
        q_real: Optional[List[float]] = None,
        eval_q: Optional[List[float]] = None,
        result_ik: Optional[Dict[str, Any]] = None,
        pose: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Smooth, jerk-free controlled deceleration when hitting a safety boundary or collision.
        Instead of an instantaneous hard-cutoff (which causes violent mechanical clunk),
        we execute a smooth deceleration ramp down to rest over 2-4 frames while strictly
        guaranteeing that the robot never violates the 1.0cm armor.
        """
        with self._lock:
            self._status = status_tag

        if status_tag != self._last_logged_status:
            print(f"[Teleop Status] {status_tag}")
            self._last_logged_status = status_tag

        if detail:
            self.log_diagnostic_block(status_tag, detail, eval_q or q_real, result_ik, pose)

        q_curr = np.array(self._last_valid_q if self._last_valid_q is not None else q_real, dtype=float)
        prev_v = np.array(self._prev_dq if self._prev_dq is not None else np.zeros(6), dtype=float)

        A_MAX = float(os.environ.get("MAX_JOINT_ACC", "0.048"))
        v_max_abs = float(np.max(np.abs(prev_v)))

        if v_max_abs < 1e-4:
            self._prev_dq = np.zeros(6, dtype=float)
            if self._exec is not None:
                self._exec.send(q_curr.tolist())
            return {"success": False, "status": status_tag, "q": q_curr.tolist(), "detail": detail}

        # Smooth deceleration ramp (ABS style): decay velocity smoothly towards zero
        # Limit deceleration magnitude so velocity change |delta_v| <= A_MAX per step
        delta_v_step = min(v_max_abs, A_MAX)
        decay = max(0.0, 1.0 - delta_v_step / max(1e-4, v_max_abs))
        v_brake = prev_v * decay

        # Check candidate next position: ensure q_cand never violates 1.0cm armor
        lambda_safe = 0.0
        for factor in [1.0, 0.75, 0.50, 0.25, 0.0]:
            q_test = (q_curr + factor * v_brake).tolist()
            sc = self._solver.check_self_collision(q_test, verbose=False) if hasattr(self._solver, "check_self_collision") else {"safe": True}
            tc = self._solver.check_table_collision(q_test, tcp_safe_clearance=0.0, cam_safe_clearance=0.0) if hasattr(self._solver, "check_table_collision") else {"safe": True}
            if sc.get("safe", True) and tc.get("safe", True):
                lambda_safe = factor
                break

        q_safe = (q_curr + lambda_safe * v_brake).tolist()
        self._prev_dq = lambda_safe * v_brake
        self._last_valid_q = q_safe

        if self._exec is not None:
            self._exec.send(q_safe)

        return {"success": False, "status": status_tag, "q": q_safe, "detail": detail}

    def step_joint_space(
        self,
        q_target_desired: List[float],
        q_real: Optional[List[float]] = None,
        max_step: float = 0.06
    ) -> Dict[str, Any]:
        """
        Interactive Joint-Space Control from Cockpit sliders, arrows, and direct degree input.
        1. Clamps target joints to UR3E_LIMITS.
        2. Slew-rate limiting: limits |delta_q| <= max_step (0.08 rad/step) for smooth jerk-free movement.
        3. Collision Guard: verifies 0.5cm uniform armor (self-collision & table-collision).
        4. Sends servoJ command if safe; stops at barrier if collision detected.
        5. Automatically synchronizes internal smoothed target and re-anchors pose source.
        """
        if not self._enabled:
            return {"success": False, "status": "DISABLED", "q": self._last_valid_q}

        from teleop.pyroki_solver import UR3E_LIMITS

        q_curr = np.array(self._last_valid_q if self._last_valid_q is not None else q_real, dtype=float)

        # 1. Clamp to safe joint limits
        lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
        upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
        SAFE_EPS = 1e-3

        q_clamped = np.clip(q_target_desired, lower_limits + SAFE_EPS, upper_limits - SAFE_EPS)

        # 2. Slew-Rate Limiting: limit maximum step to max_step (default 0.08 rad/step ~ 4.58 deg/step)
        delta_q = q_clamped - q_curr
        max_jump = float(np.max(np.abs(delta_q)))
        if max_jump > max_step:
            delta_q = delta_q * (max_step / max_jump)

        q_next = (q_curr + delta_q).tolist()

        # 3. Collision Guard & Tangential Sliding on 1.0cm uniform armor
        self_chk = self._solver.check_self_collision(q_next, verbose=False) if hasattr(self._solver, "check_self_collision") else {"safe": True}
        tbl_chk = self._solver.check_table_collision(q_next, tcp_safe_clearance=0.0, cam_safe_clearance=0.0) if hasattr(self._solver, "check_table_collision") else {"safe": True}

        if not (self_chk.get("safe", True) and tbl_chk.get("safe", True)):
            delta_q_proj = np.copy(delta_q)
            if not self_chk.get("safe", True) and hasattr(self._solver, "_coll_grad_jit"):
                try:
                    import jax.numpy as jnp
                    g_vec = np.array(self._solver._coll_grad_jit(jnp.array(q_curr, dtype=jnp.float32)))
                    dot_g = float(np.dot(g_vec, delta_q_proj))
                    if dot_g < 0.0:
                        norm_g_sq = float(np.dot(g_vec, g_vec))
                        if norm_g_sq > 1e-8:
                            delta_q_proj = delta_q_proj - (dot_g / norm_g_sq) * g_vec
                except Exception:
                    pass

            # Bisection search to find maximum safe sliding step
            lo, hi = 0.0, 1.0
            best_q = None
            for _ in range(5):
                mid = 0.5 * (lo + hi)
                q_mid = (q_curr + mid * delta_q_proj).tolist()
                sc = self._solver.check_self_collision(q_mid, verbose=False) if hasattr(self._solver, "check_self_collision") else {"safe": True}
                tc = self._solver.check_table_collision(q_mid, tcp_safe_clearance=0.0, cam_safe_clearance=0.0) if hasattr(self._solver, "check_table_collision") else {"safe": True}
                if sc.get("safe", True) and tc.get("safe", True):
                    best_q = q_mid
                    lo = mid
                else:
                    hi = mid

            if lo > 0.05 and best_q is not None:
                q_next = best_q
                delta_q = np.array(best_q) - q_curr
            else:
                status_tag = "ENV COLLISION" if not tbl_chk.get("safe", True) else "SELF COLLISION"
                detail = f"Collision barrier reached ({status_tag})"
                return self._apply_smooth_braking_and_send(status_tag, detail, q_curr.tolist())

        # Check if any joint is at boundary
        at_limit = bool(np.any((np.array(q_next) <= lower_limits + 2.0 * SAFE_EPS) |
                               (np.array(q_next) >= upper_limits - 2.0 * SAFE_EPS)))
        status_tag = "JOINT LIMIT" if at_limit else "OK"

        # 4. Send servoJ command
        if self._exec is not None:
            self._exec.send(q_next)
        self._last_valid_q = q_next
        self._prev_dq = np.array(delta_q, dtype=float)

        with self._lock:
            self._status = status_tag

        if status_tag != self._last_logged_status:
            print(f"[Teleop Status] {status_tag}")
            self._last_logged_status = status_tag

        # 5. FK Synchronization: update internal smoothed target and pose anchor
        try:
            fk_now = self._solver.fk(q_next)
            p_now = np.array(fk_now["pos"], dtype=float)
            q_now_wxyz = np.array(fk_now["quat"], dtype=float)
            self._smoothed_pos = np.copy(p_now)
            self._smoothed_quat = np.copy(q_now_wxyz)
            if self._pose is not None and hasattr(self._pose, "reanchor_position"):
                self._pose.reanchor_position(p_now)
        except Exception:
            fk_now = None

        return {"success": True, "status": status_tag, "q": q_next, "fk": fk_now}

    def step(self, q_real: List[float], pose: Any = _DEFAULT_FETCH_POSE):
        """
        One 20Hz control cycle.
        Input:  q_real list[6 rad] - actual joint angles from RTDE Port 30004
                pose dict/tuple (optional) - target pose from PoseSource (if explicitly None, holds robot stationary)
        """
        if not self._enabled:
            return

        # Fetch target pose if not explicitly provided
        if pose is _DEFAULT_FETCH_POSE:
            pose = self._pose.get_pose() if self._pose is not None else None

        if pose is None:
            self.notify_clutch_released(q_real)
            q_hold = self._last_valid_q if self._last_valid_q is not None else q_real
            self._exec.send(q_hold)
            return

        # ── 1. q_seed Hygiene & Current Configuration Estimation ──
        if self._last_valid_q is not None:
            q_seed = list(self._last_valid_q)
            if q_real is not None:
                max_drift = float(np.max(np.abs(np.array(self._last_valid_q) - np.array(q_real))))
                if max_drift > 0.35:  # Resync if drifted > 20 deg from actual robot telemetry
                    q_seed = list(q_real)
                    self._last_valid_q = list(q_real)
        else:
            q_seed = list(q_real) if q_real is not None else list(self._solver.default_home)

        lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
        upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
        SAFE_EPS = 1e-3
        q_seed = np.clip(q_seed, lower_limits + SAFE_EPS, upper_limits - SAFE_EPS).tolist()

        fk_curr = self._solver.fk(q_seed)
        p_curr = np.array(fk_curr["pos"])
        q_curr_wxyz = np.array(fk_curr["quat"])

        # Detect singularity alpha at current configuration
        sing_alpha = 0.0
        if hasattr(self._solver, "check_singularity"):
            try:
                sing_info = self._solver.check_singularity(q_seed)
                sing_alpha = float(sing_info.get("alpha", 0.0))
            except Exception:
                sing_alpha = 0.0

        # Layer 1: Cartesian safety clamping (Z >= 0.020m, R <= 0.52m)
        pose_safe = clamp_cartesian_safety(pose, z_min=0.020, z_max=0.60, r_min=0.0, r_max=0.52)

        user_cmd_p = np.array(pose_safe["pos"] if isinstance(pose_safe, dict) else pose_safe[0], dtype=float)
        raw_p = np.copy(user_cmd_p)
        raw_q = np.array(pose_safe["quat"] if isinstance(pose_safe, dict) else pose_safe[1], dtype=float)
        raw_q_norm = raw_q / max(1e-6, np.linalg.norm(raw_q))

        # ── 2. Target Pose Tethering: limit distance to current TCP to at most d_max (75mm) ──
        # Prevents target from running away to infinity when arm reaches full extension boundary.
        D_MAX_TETHER = float(os.environ.get("MAX_TETHER_DIST", "0.075"))  # 75mm leash
        vec_raw = raw_p - p_curr
        dist_raw = float(np.linalg.norm(vec_raw))
        if dist_raw > D_MAX_TETHER:
            raw_p = p_curr + (vec_raw / dist_raw) * D_MAX_TETHER

        # Measure translation vs rotation demand from current TCP
        delta_p_req = float(np.linalg.norm(raw_p - p_curr))
        cos_half_req = min(1.0, max(-1.0, abs(float(np.dot(q_curr_wxyz, raw_q_norm)))))
        delta_rot_req = float(2.0 * math.acos(cos_half_req))

        is_source_pure_rot = bool(getattr(self._pose, "is_pure_rotation", False) is True)
        is_source_translating = bool(getattr(self._pose, "is_translating", False) is True)
        is_grip_rot = bool(getattr(self._pose, "grip_engaged", False) is True)

        TRANSLATION_RELEASE_THRESH = 0.003  # 3mm movement of commanded target releases arm anchor

        if self._arm_anchor_p is not None:
            delta_target_from_anchor = float(np.linalg.norm(user_cmd_p - self._arm_anchor_p))
        else:
            delta_target_from_anchor = 0.0

        # A) Explicit translation intent (trigger in VR or moving target >= 3mm from anchor):
        if is_source_translating or (self._arm_anchor_p is not None and delta_target_from_anchor >= TRANSLATION_RELEASE_THRESH):
            self._arm_anchor_q = None
            self._arm_anchor_p = None
            self._rot_anchor_body_q = None
            is_pure_rotation = False
        else:
            # B) Pure rotation intent (VR Grip without Trigger, or Gizmo rotation while position is stationary):
            if (is_source_pure_rot or is_grip_rot) and not is_source_translating:
                is_pure_rotation = True
            elif self._arm_anchor_q is not None and not is_source_translating:
                # Already locked in pure rotation mode, maintain anchor until target translated >= 3mm
                is_pure_rotation = True
            else:
                is_pure_rotation = False

            if is_pure_rotation:
                if self._arm_anchor_q is None:
                    # Latch arm anchor from current actual body joints and current target position
                    self._arm_anchor_q = np.array(q_real[:3], dtype=float)
                    self._arm_anchor_p = np.copy(user_cmd_p)
                self._rot_anchor_body_q = self._arm_anchor_q
            else:
                self._arm_anchor_q = None
                self._arm_anchor_p = None
                self._rot_anchor_body_q = None

        # ── 3. High-Responsiveness Target Low-Pass Filter (Input Smoothing) ──
        # Responsive Cartesian EMA (BETA_POS=0.60, BETA_ROT=0.75) for smooth, solid and silky tracking
        BETA_POS = float(os.environ.get("POS_SMOOTH_BETA", os.environ.get("TARGET_FILTER_BETA_POS", "0.60")))
        BETA_ROT = float(os.environ.get("ROT_SMOOTH_BETA", os.environ.get("TARGET_FILTER_BETA_ROT", "0.75")))
        if self._was_clutch_released or self._smoothed_pos is None:
            self._smoothed_pos = np.copy(raw_p)
            self._smoothed_quat = np.copy(raw_q_norm)
            self._prev_dq = np.zeros(6, dtype=float)
            self._was_clutch_released = False
        else:
            # Tether smoothed_pos so it never accumulates runaway windup lag beyond physical reach
            dist_sm = float(np.linalg.norm(self._smoothed_pos - p_curr))
            if dist_sm > D_MAX_TETHER:
                self._smoothed_pos = p_curr + (self._smoothed_pos - p_curr) * (D_MAX_TETHER / dist_sm)

            self._smoothed_pos = BETA_POS * raw_p + (1.0 - BETA_POS) * self._smoothed_pos
            if np.dot(self._smoothed_quat, raw_q_norm) < 0.0:
                raw_q_norm = -raw_q_norm
            self._smoothed_quat = BETA_ROT * raw_q_norm + (1.0 - BETA_ROT) * self._smoothed_quat
            self._smoothed_quat /= max(1e-6, np.linalg.norm(self._smoothed_quat))

        # Snap to exact target when close to eliminate asymptotic residual error (Zero Steady-State Error)
        if np.linalg.norm(raw_p - self._smoothed_pos) < 0.0005:  # < 0.5mm
            self._smoothed_pos = np.copy(raw_p)

        cos_half = min(1.0, max(-1.0, abs(float(np.dot(self._smoothed_quat, raw_q_norm)))))
        if cos_half > 0.99995:  # < 0.01 rad
            self._smoothed_quat = np.copy(raw_q_norm)

        p_target = self._smoothed_pos
        q_target_wxyz = self._smoothed_quat

        # ── 4. Position Leash: L_step = 80mm in normal space, shortens smoothly in singularity ──
        vec_p = p_target - p_curr
        dist_p = float(np.linalg.norm(vec_p))
        BASE_P_LEASH = float(os.environ.get("MAX_P_LEASH", "0.080"))       # 80mm leash (1.6 m/s, đầm chắc mượt mà)
        SING_P_LEASH = float(os.environ.get("SING_P_LEASH", "0.035"))      # 35mm in singularity
        max_p_leash = float(BASE_P_LEASH - (BASE_P_LEASH - SING_P_LEASH) * sing_alpha)

        if dist_p > max_p_leash:
            p_clamped = p_curr + (vec_p / dist_p) * max_p_leash
        else:
            p_clamped = p_target

        # 2. Orientation Leash: 0.45 rad (~26 deg)
        q_c_norm = q_curr_wxyz / max(1e-6, np.linalg.norm(q_curr_wxyz))
        q_t_norm = q_target_wxyz / max(1e-6, np.linalg.norm(q_target_wxyz))
        dot_q = float(np.dot(q_c_norm, q_t_norm))
        if dot_q < 0.0:
            q_t_norm = -q_t_norm
            dot_q = -dot_q
        dot_q = min(1.0, max(-1.0, dot_q))
        theta = float(2.0 * math.acos(dot_q))

        BASE_ORI_LEASH = float(os.environ.get("MAX_ORI_LEASH", "0.45"))    # 0.45 rad (~26 deg)
        SING_ORI_LEASH = float(os.environ.get("SING_ORI_LEASH", "0.15"))   # 0.15 rad (~8.6 deg)
        max_ori_leash = float(BASE_ORI_LEASH - (BASE_ORI_LEASH - SING_ORI_LEASH) * sing_alpha)

        if theta > max_ori_leash and math.sin(theta / 2.0) > 1e-4:
            t_slerp = max_ori_leash / theta
            sin_theta_half = math.sin(theta / 2.0)
            w1 = math.sin((1.0 - t_slerp) * theta / 2.0) / sin_theta_half
            w2 = math.sin(t_slerp * theta / 2.0) / sin_theta_half
            q_clamped = w1 * q_c_norm + w2 * q_t_norm
            q_clamped /= np.linalg.norm(q_clamped)
        else:
            q_clamped = q_t_norm

        pose_clamped = {
            "pos": p_clamped.tolist(),
            "quat": q_clamped.tolist(),
        }

        # Solve IK on CPU using PyRoki JIT with arm anchor constraints
        result_ik = self._solver.solve(
            pose_clamped,
            q_seed,
            arm_anchor=self._arm_anchor_q,
            is_pure_rot=is_pure_rotation
        )

        with self._lock:
            self._latency["pyroki_ms"] = result_ik["latency_ms"]

        # Validate PyRoki IK solution
        if result_ik["q"] is not None:
            q_target = result_ik["q"]
            is_sliding = result_ik.get("sliding_on_barrier", False)
            at_joint_limit = result_ik.get("at_joint_limit", False) or result_ik.get("body_swing_blocked", False)

            # ── Pure-Rotation Anti-Arm-Swing & Smooth Per-Joint Clamping ──
            if is_pure_rotation and self._arm_anchor_q is not None:
                DELTA_MAX_BODY = 0.020  # rad (~1.15 deg) kẹp mềm quanh mỏ neo thân máy, không gây giật cục
                lower_b = self._arm_anchor_q - DELTA_MAX_BODY
                upper_b = self._arm_anchor_q + DELTA_MAX_BODY
                q_target[0] = float(np.clip(q_target[0], lower_b[0], upper_b[0]))
                q_target[1] = float(np.clip(q_target[1], lower_b[1], upper_b[1]))
                q_target[2] = float(np.clip(q_target[2], lower_b[2], upper_b[2]))

                lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
                upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
                LIMIT_ZONE = 0.04
                wrist_at_limit = bool(np.any((np.array(q_target[3:]) <= lower_limits[3:] + LIMIT_ZONE) |
                                             (np.array(q_target[3:]) >= upper_limits[3:] - LIMIT_ZONE)))
                body_at_anchor = result_ik.get("body_swing_blocked", False) or bool(
                    np.any(np.abs(np.array(q_target[:3]) - self._arm_anchor_q) >= 0.0055)
                )
                if wrist_at_limit or body_at_anchor or at_joint_limit:
                    at_joint_limit = True
                    try:
                        fk_achieved = self._solver.fk(q_target)
                        q_achieved_wxyz = np.array(fk_achieved["quat"], dtype=float)
                        self._smoothed_quat = np.copy(q_achieved_wxyz)
                        if self._pose is not None and hasattr(self._pose, "clamp_rotation_anchor") and not getattr(self._pose, "is_absolute", False):
                            self._pose.clamp_rotation_anchor(q_achieved_wxyz)
                    except Exception:
                        pass

            # Target Projection & Re-anchoring to prevent integrator windup when sliding or touching joint limit
            if (is_sliding or at_joint_limit) and self._pose is not None:
                self._apply_anti_windup_reanchor(p_curr, q_curr_wxyz)

            # Adaptive Joint EMA Filter: when entering singularity (sing_alpha > 0), smoothly blend
            # joint positions with previous frame to eliminate high-frequency discrete solver noise
            sing_alpha = result_ik.get("singularity_alpha", 0.0)
            if sing_alpha > 0.01 and self._last_valid_q is not None:
                ema_weight = 0.15 * sing_alpha  # 0% at alpha=0, up to 15% at alpha=1 for nimble tracking
                q_target = [
                    (1.0 - ema_weight) * q_target[i] + ema_weight * self._last_valid_q[i]
                    for i in range(6)
                ]

            # Branch-Flip Guard check
            if self._last_valid_q is not None:
                max_jump = max(abs(q_target[i] - self._last_valid_q[i]) for i in range(6))
                if max_jump > 1.60:  # ~92 deg threshold
                    detail = f"Sudden joint jump {max_jump:.2f} rad > 1.60 rad (branch-flip detected)"
                    status_msg = "JOINT LIMIT"
                    self._apply_anti_windup_reanchor(p_curr, q_curr_wxyz)
                    self._apply_smooth_braking_and_send("JOINT LIMIT", detail, q_real, q_target, result_ik, pose)
                    return

                # Joint Acceleration / Slew-Rate Limiter (Second-Order Smooth Tracking with Soft Braking)
                A_MAX_BASE = float(os.environ.get("MAX_JOINT_ACC", "0.045"))
                A_MAX = A_MAX_BASE * 1.25 if (delta_p_req > 0.003 or delta_rot_req > 0.02) else A_MAX_BASE
                V_MAX = float(os.environ.get("MAX_JOINT_VEL", "0.080"))    # 0.080 rad/step (= 1.6 rad/s at 20Hz, đầm chắc mượt mà)

                # Continuous smooth damping ramp for singularities:
                raw_damping = 1.0 - 0.30 * (sing_alpha * sing_alpha * (3.0 - 2.0 * sing_alpha))
                if hasattr(self, "_gate") and self._gate is not None:
                    try:
                        gate_m = self._gate.check(q_target)
                        raw_damping = min(raw_damping, gate_m.get("damping", 1.0))
                    except Exception:
                        pass

                if not hasattr(self, "_smoothed_damping") or self._smoothed_damping is None:
                    self._smoothed_damping = float(raw_damping)
                else:
                    self._smoothed_damping = 0.25 * float(raw_damping) + 0.75 * self._smoothed_damping

                V_MAX_eff = V_MAX * self._smoothed_damping

                e = np.array(q_target) - np.array(self._last_valid_q)
                abs_e = np.abs(e)
                sign_e = np.sign(e)

                # Exact discrete-time soft braking velocity envelope:
                # v_allow = -a/2 + sqrt(a^2/4 + 2*a*|e|)
                v_allow = -0.5 * A_MAX + np.sqrt(0.25 * A_MAX**2 + 2.0 * A_MAX * abs_e)
                desired_dq = sign_e * np.minimum(V_MAX_eff, np.minimum(abs_e, v_allow))

                delta_acc = desired_dq - self._prev_dq

                # 1. Acceleration limiting with proportional vector scaling
                max_acc = float(np.max(np.abs(delta_acc)))
                if max_acc > A_MAX:
                    delta_acc = delta_acc * (A_MAX / max_acc)

                limited_dq = self._prev_dq + delta_acc

                # 2. Velocity limiting with proportional vector scaling
                max_vel = float(np.max(np.abs(limited_dq)))
                if max_vel > V_MAX_eff:
                    limited_dq = limited_dq * (V_MAX_eff / max_vel)

                # Prevent overshoot past target if approaching target closely
                for i in range(6):
                    if abs(e[i]) > 1e-6 and (limited_dq[i] * sign_e[i] > abs_e[i]):
                        limited_dq[i] = e[i]

                q_target = (np.array(self._last_valid_q) + limited_dq).tolist()

                # Per-Joint Soft Clamping: strictly enforce UR3E_LIMITS on q_target
                lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
                upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
                SAFE_EPS = 1e-3
                LIMIT_ZONE = 0.05
                q_target = np.clip(q_target, lower_limits + SAFE_EPS, upper_limits - SAFE_EPS).tolist()
                at_joint_limit = at_joint_limit or bool(np.any((np.array(q_target) <= lower_limits + LIMIT_ZONE) |
                                                               (np.array(q_target) >= upper_limits - LIMIT_ZONE)))

                self._prev_dq = limited_dq

                # Joint-Space Smoothing: preserve exact acceleration continuity and eliminate 20Hz discrete jitter
                ALPHA_Q = float(os.environ.get("JOINT_SMOOTH_ALPHA", "0.80"))
                if ALPHA_Q < 0.999 and self._last_valid_q is not None and not self._was_clutch_released:
                    q_target = [
                        ALPHA_Q * q_target[i] + (1.0 - ALPHA_Q) * self._last_valid_q[i]
                        for i in range(6)
                    ]
                    self._prev_dq = np.array(q_target) - np.array(self._last_valid_q)
        else:
            # IK failed / rejected: Categorize root cause
            pos = pose_clamped["pos"] if isinstance(pose_clamped, dict) else pose_clamped[0]
            dist_base = (pos[0]**2 + pos[1]**2 + (pos[2] - 0.15)**2)**0.5
            eval_q = result_ik.get("raw_q", q_real)
            fail_reason = result_ik.get("fail_reason", "")

            if result_ik.get("violated_joints"):
                tag = "JOINT LIMIT"
                detail = f"Joint limit exceeded: {fail_reason}"
            elif "self_collision" in fail_reason:
                tag = "SELF COLLISION"
                detail = f"Self-collision contact: {fail_reason}"
            elif "table_collision" in fail_reason:
                tag = "ENV COLLISION"
                detail = f"Table armor contact: {fail_reason}"
            elif dist_base > 0.535:
                tag = "SINGULARITY (ELBOW)"
                detail = f"Distance to base {dist_base*1000:.1f}mm exceeds reach 535mm (Elbow Singularity)"
            elif pos[2] <= 0.009:
                tag = "ENV COLLISION"
                detail = f"Target TCP Z={pos[2]*1000:.1f}mm approaching table barrier boundary"
            else:
                tag = "JOINT LIMIT"
                detail = fail_reason or "No valid IK solution found"

            self._apply_anti_windup_reanchor(p_curr, q_curr_wxyz)
            self._apply_smooth_braking_and_send(tag, detail, q_real, eval_q, result_ik, pose)
            return

        # Check multi-layer SafetyGate
        if hasattr(self, "_gate") and self._gate is not None:
            gate = self._gate.check(q_target)
            if not gate["pass"]:
                reason = gate["reason"]
                if "singularity" in reason.lower():
                    tag = "SINGULARITY"
                elif "clamp" in reason.lower() or "limit" in reason.lower():
                    tag = "JOINT LIMIT"
                elif "self_collision" in reason.lower() or "self-collision" in reason.lower():
                    tag = "SELF COLLISION"
                elif "table_collision" in reason.lower() or "table" in reason.lower():
                    tag = "ENV COLLISION"
                else:
                    tag = "JOINT LIMIT"

                if tag == "JOINT LIMIT":
                    # Per-joint soft clamping: Never abort execution on joint limits!
                    # Clamped joint stays at boundary, remaining joints continue tracking
                    lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
                    upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
                    SAFE_EPS = 1e-3
                    q_target = np.clip(q_target, lower_limits + SAFE_EPS, upper_limits - SAFE_EPS).tolist()
                    at_joint_limit = True
                else:
                    self._apply_anti_windup_reanchor(p_curr, q_curr_wxyz)
                    self._apply_smooth_braking_and_send(tag, reason, q_real, q_target, result_ik, pose)
                    return

        # Forward Kinematics verification: Ensure TCP and Camera respect table armor (+0.020m)
        if self._solver is not None and hasattr(self._solver, "check_table_collision"):
            try:
                tbl_final = self._solver.check_table_collision(q_target, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
                if not tbl_final["safe"]:
                    detail = f"FK table boundary breach: {tbl_final['violator']} contact ({tbl_final['clearance_cm']*10:.1f}mm <= 0mm)"
                    self._apply_anti_windup_reanchor(p_curr, q_curr_wxyz)
                    self._apply_smooth_braking_and_send("ENV COLLISION", detail, q_real, q_target, result_ik, pose)
                    return
            except Exception:
                pass

        # Final Self-Collision verification: Ensure arm and camera armor respect 1.0cm clearance
        if self._solver is not None and hasattr(self._solver, "check_self_collision"):
            try:
                coll_thresh = getattr(self._solver, "clearance_threshold", 0.01)
                self_final = self._solver.check_self_collision(q_target, clearance_threshold=coll_thresh, verbose=False)
                if not self_final["safe"]:
                    detail = f"Self-collision: {self_final.get('link_a')} vs {self_final.get('link_b')} ({self_final.get('min_dist', 0.0)*1000:.1f}mm <= {coll_thresh*1000:.1f}mm)"
                    self._apply_anti_windup_reanchor(p_curr, q_curr_wxyz)
                    self._apply_smooth_braking_and_send("SELF COLLISION", detail, q_real, q_target, result_ik, pose)
                    return
            except Exception:
                pass

        # Send command to robot via servoJ
        self._exec.send(q_target)
        self._last_valid_q = q_target

        # ── Formulate standardized ultra-short English telemetry status ──
        # Check singularity status from PyRoki solver
        sing_info = result_ik.get("singularity")
        if sing_info is None and hasattr(self._solver, "check_singularity"):
            try:
                sing_info = self._solver.check_singularity(q_target)
            except Exception:
                sing_info = None

        tbl_chk = self._solver.check_table_collision(q_target, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
        self_chk = self._solver.check_self_collision(q_target, verbose=False)

        is_near_sing = bool(sing_info and sing_info.get("singular", False))
        sing_types = sing_info.get("types", []) if sing_info else []

        other_joint_at_limit = False
        if at_joint_limit:
            lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
            upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
            for idx in range(6):
                if (q_target[idx] <= lower_limits[idx] + 0.05) or (q_target[idx] >= upper_limits[idx] - 0.05):
                    other_joint_at_limit = True
                    break

        if not self_chk.get("safe", True) or self_chk.get("min_dist", 1.0) <= getattr(self._solver, "clearance_threshold", 0.01) + 0.0005:
            new_status = "SELF COLLISION"
        elif not tbl_chk.get("safe", True) or tbl_chk.get("min_clearance", 1.0) <= 0.001:
            new_status = "ENV COLLISION"
        elif at_joint_limit or other_joint_at_limit:
            new_status = "JOINT LIMIT"
        elif is_near_sing:
            if "WRIST" in sing_types:
                new_status = "SINGULARITY (WRIST)"
            elif "ELBOW" in sing_types:
                new_status = "SINGULARITY (ELBOW)"
            else:
                new_status = "SINGULARITY"
        else:
            new_status = "OK"

        with self._lock:
            self._status = new_status

        if new_status != self._last_logged_status:
            print(f"[Teleop Status] {new_status}")
            self._last_logged_status = new_status
            self._last_block_tag = None
