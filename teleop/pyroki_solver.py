"""
pyroki_solver.py - Bộ giải Động học Nghịch (IK) và Thuận (FK) cho UR3e sử dụng PyRoki.
Chạy thuần CPU, loại bỏ 100% sự phụ thuộc vào GPU NVIDIA / CUDA.
"""

import os
import time
import math
from typing import Optional, Union, Dict, Any, List

import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import yourdfpy
import pyroki as pk
from loguru import logger

# Tắt loguru debug/info từ pyroki và jaxls để không spam terminal
logger.disable("pyroki")
logger.disable("jaxls")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF_PATH = os.path.join(_BASE_DIR, "assets", "ur3e_custom.urdf")

# Giới hạn góc khớp an toàn cho UR3e (Radian)
UR3E_LIMITS = [
    (-2 * math.pi, 2 * math.pi),  # shoulder_pan_joint
    (-math.pi, 0.25),             # shoulder_lift_joint
    (-2.80, 2.80),                # elbow_joint
    (-2 * math.pi, 2 * math.pi),  # wrist_1_joint
    (-2 * math.pi, 2 * math.pi),  # wrist_2_joint
    (-4 * math.pi, 4 * math.pi),  # wrist_3_joint
]


def _nearest_equivalent(angle: float, current: float, lo: float, hi: float) -> float:
    best, best_dist = angle, float("inf")
    for k in range(-3, 4):
        candidate = angle + k * 2 * math.pi
        if lo <= candidate <= hi:
            d = abs(candidate - current)
            if d < best_dist:
                best_dist, best = d, candidate
    return best


def normalize_to_current(q_ik: List[float], q_current: List[float]) -> List[float]:
    return [
        _nearest_equivalent(q_ik[i], q_current[i], *UR3E_LIMITS[i])
        for i in range(6)
    ]


def validate_joints(q: List[float]) -> bool:
    for i, (lo, hi) in enumerate(UR3E_LIMITS):
        if not (lo <= q[i] <= hi):
            return False
    return True


class PyrokiIKSolver:
    """
    Bộ giải IK/FK thời gian thực cho robot UR3e sử dụng PyRoki trên CPU.
    """

    def __init__(self, urdf_path: Optional[str] = None, tcp_link: str = "tool0"):
        self.urdf_path = urdf_path or DEFAULT_URDF_PATH
        self.tcp_link = tcp_link

        print(f"[PyRoki] Khởi tạo mô hình Robot từ URDF: {self.urdf_path}...")
        self.urdf = yourdfpy.URDF.load(self.urdf_path, load_meshes=False)
        self.robot = pk.Robot.from_urdf(self.urdf)

        if self.tcp_link not in self.robot.links.names:
            raise ValueError(f"Không tìm thấy link '{self.tcp_link}' trong URDF. Các link có sẵn: {self.robot.links.names}")

        self.tcp_idx = jnp.array(self.robot.links.names.index(self.tcp_link))
        self.q_var = self.robot.joint_var_cls(0)
        self.default_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

        # Khởi tạo hàm giải JIT
        self._solve_jit = self._build_jit_solver()

        # Khởi động JIT warm-up để triệt tiêu độ trễ frame đầu tiên
        print("[PyRoki] Đang JIT compile đồ thị tối ưu (Warm-up)...")
        self._warm_up()
        print("[PyRoki] Bộ giải IK PyRoki CPU đã sẵn sàng!")

    def _build_jit_solver(self):
        robot = self.robot
        q_var = self.q_var
        tcp_idx = self.tcp_idx

        @jax.jit
        def _solve(target_pos, target_quat, seed_arr):
            target_se3 = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3(target_quat),
                translation=target_pos
            )
            cost_pose = pk.costs.pose_cost_analytic_jac(
                robot=robot,
                joint_var=q_var,
                target_pose=target_se3,
                target_link_index=tcp_idx,
                pos_weight=50.0,
                ori_weight=20.0
            )
            cost_limit = pk.costs.limit_cost(
                robot=robot,
                joint_var=q_var,
                weight=0.5
            )
            cost_rest = pk.costs.rest_cost(
                joint_var=q_var,
                rest_pose=seed_arr,
                weight=0.02
            )
            problem = jaxls.LeastSquaresProblem([cost_pose, cost_limit, cost_rest], [q_var]).analyze()
            sol = problem.solve(
                jaxls.VarValues.make([q_var.with_value(seed_arr)]),
                termination=jaxls.TerminationConfig(
                    max_iterations=25,
                    early_termination=True,
                    cost_tolerance=1e-5,
                    parameter_tolerance=1e-5
                ),
                verbose=False
            )
            return sol[q_var]

        return _solve

    def _warm_up(self):
        try:
            sample_pos = jnp.array([0.25, 0.15, 0.25], dtype=jnp.float32)
            sample_quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
            sample_seed = jnp.array(self.default_home, dtype=jnp.float32)
            res = self._solve_jit(sample_pos, sample_quat, sample_seed)
            if hasattr(res, "block_until_ready"):
                res.block_until_ready()
        except Exception as e:
            print(f"[PyRoki] Cảnh báo warm-up: {e}")

    def fk(self, joints: List[float]) -> Dict[str, Any]:
        """
        Tính động học thuận (FK) từ 6 góc khớp.
        Trả về vị trí TCP [x, y, z] và quaternion [w, x, y, z].
        """
        q_arr = jnp.array(joints, dtype=jnp.float32)
        fk_all = self.robot.forward_kinematics(q_arr)
        pose_7d = np.array(fk_all[int(self.tcp_idx)])
        # pose_7d có format [qw, qx, qy, qz, x, y, z]
        quat_wxyz = [float(pose_7d[0]), float(pose_7d[1]), float(pose_7d[2]), float(pose_7d[3])]
        pos_xyz = [float(pose_7d[4]), float(pose_7d[5]), float(pose_7d[6])]
        return {"pos": pos_xyz, "quat": quat_wxyz}

    def solve(
        self,
        pose: Union[tuple, list, Dict[str, Any]],
        q_seed: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Giải bài toán động học nghịch (IK) thời gian thực bằng PyRoki JIT CPU.

        Args:
            pose: tuple (pos, quat_wxyz) hoặc dict {"pos": ..., "quat": ...}
            q_seed: 6 góc khớp của frame trước đó làm điểm mồi (warm-start)

        Returns:
            {"q": list[6 rad] hoặc None, "latency_ms": float, "solver": "pyroki"}
        """
        t0 = time.perf_counter()

        if isinstance(pose, dict):
            pos = pose["pos"]
            quat = pose["quat"]
        elif isinstance(pose, (tuple, list)) and len(pose) == 2:
            pos, quat = pose[0], pose[1]
        else:
            return {"q": None, "latency_ms": 0.0, "solver": "pyroki", "error": "Invalid pose format"}

        ref_seed = q_seed if q_seed is not None else self.default_home
        seed_arr = jnp.array(ref_seed, dtype=jnp.float32)

        # Chuyển đổi sang jax array
        target_pos_j = jnp.array(pos, dtype=jnp.float32)
        target_quat_j = jnp.array(quat, dtype=jnp.float32)  # [w, x, y, z]
        # Chuẩn hóa quaternion tránh NaN
        norm_q = jnp.linalg.norm(target_quat_j)
        if norm_q > 1e-6:
            target_quat_j = target_quat_j / norm_q
        else:
            target_quat_j = jnp.array([1.0, 0.0, 0.0, 0.0])

        try:
            sol_q = self._solve_jit(target_pos_j, target_quat_j, seed_arr)
            raw_q = np.array(sol_q, dtype=float).tolist()
            norm_q = normalize_to_current(raw_q, ref_seed)

            # Kiểm tra sai số vị trí sau giải IK
            fk_check = self.fk(norm_q)
            pos_err = np.linalg.norm(np.array(fk_check["pos"]) - np.array(pos))

            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Nếu sai số vị trí < 12mm và góc khớp hợp lệ -> nghiệm thành công
            if pos_err < 0.012 and validate_joints(norm_q):
                return {"q": norm_q, "latency_ms": dt_ms, "solver": "pyroki", "pos_err_mm": pos_err * 1000.0}
            else:
                return {"q": None, "latency_ms": dt_ms, "solver": "pyroki", "pos_err_mm": pos_err * 1000.0}

        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return {"q": None, "latency_ms": dt_ms, "solver": "pyroki", "error": str(e)}
