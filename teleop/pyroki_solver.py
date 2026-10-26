"""
pyroki_solver.py - Real-time Inverse/Forward Kinematics solver for UR3e using PyRoki.
Runs entirely on CPU, zero dependency on GPU / NVIDIA CUDA.
"""

import os
import time
import math
import yaml
from typing import Optional, Union, Dict, Any, List
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import yourdfpy
import pyroki as pk
from pyroki.collision import RobotCollision, Capsule
from loguru import logger

# Suppress pyroki/jaxls debug/info log spam
logger.disable("pyroki")
logger.disable("jaxls")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF_PATH = os.path.join(_BASE_DIR, "assets", "ur3e_custom.urdf")
DEFAULT_SCENE_PATH = os.path.join(_BASE_DIR, "assets", "obstacle_scene.yml")


def _urdf_filename_handler(fname: str) -> str:
    """Resolve mesh file paths for URDF, compatible with both local and ROS description layouts."""
    base_name = os.path.basename(fname)
    direct_asset = os.path.join(_BASE_DIR, "assets", base_name)
    if os.path.exists(direct_asset):
        return direct_asset
    if fname.startswith("package://ur_description/"):
        rel_path = fname[len("package://ur_description/"):]
        candidates = [
            os.path.join(_BASE_DIR, "assets", "ur_description", rel_path),
            os.path.join(_BASE_DIR, "assets", rel_path),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
    return fname

_ACTUATED_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Safe joint angle limits for UR3e (Radians) - full freedom, protected by skin-tight collision
UR3E_LIMITS = [
    (-math.pi, math.pi),          # shoulder_pan_joint:  [-180, +180]
    (-2.65, 0.20),                # shoulder_lift_joint: [-151.8, +11.5] (extended tilt range)
    (-0.08, 3.00),                 # elbow_joint:         [-4.6, +171.8] (safe range, allows straight arm down to -0.86° / -0.015 rad)
    (-2.95, 2.95),                # wrist_1_joint:       [-169°, +169°] (chống xoắn gập camera/mount)
    (-2.95, 2.95),                # wrist_2_joint:       [-169°, +169°] (chống xoắn gập camera/mount)
    (-3.05, 3.05),                # wrist_3_joint:       [-175°, +175°] (chống vặn cáp camera)
]


def _nearest_equivalent(angle: float, current: float, lo: float, hi: float) -> float:
    """
    Chuẩn hóa góc angle về nhánh gần current nhất trong dải [lo, hi].
    Đặc biệt ngăn chặn hiện tượng lật nhánh 2*pi (branch-flip jump) khi khớp chạm sát biên ±pi:
    - Tìm nhánh k*2*pi sao cho khoảng cách tới `current` là nhỏ nhất.
    - Kẹp (clamp) mềm góc trong [lo + 1e-3, hi - 1e-3].
    - Tuyệt đối không để một góc chỉ chớm vượt biên bị cộng/trừ 2*pi đẩy sang nhánh đối diện
      gây kích hoạt Branch-Flip Guard!
    """
    safe_lo = lo + 1e-3
    safe_hi = hi - 1e-3

    # Tìm k sao cho angle + k*2*pi gần current nhất
    k_ideal = round((current - angle) / (2.0 * math.pi))
    best_candidate = angle + k_ideal * 2.0 * math.pi
    clamped = min(max(best_candidate, safe_lo), safe_hi)

    # Nếu có candidate nằm tự nhiên trong [lo, hi] mà gần current (< 1.2 rad), ưu tiên chọn
    candidates_in_range = []
    for k in (k_ideal - 1, k_ideal, k_ideal + 1):
        c = angle + k * 2.0 * math.pi
        if lo <= c <= hi and abs(c - current) < 1.2:
            candidates_in_range.append(c)

    if candidates_in_range:
        return min(candidates_in_range, key=lambda c: abs(c - current))

    return clamped


def normalize_to_current(q_ik: List[float], q_current: List[float]) -> List[float]:
    return [
        _nearest_equivalent(q_ik[i], q_current[i], *UR3E_LIMITS[i])
        for i in range(6)
    ]


def validate_joints(q: List[float], tol: float = 1e-3) -> bool:
    for i, (lo, hi) in enumerate(UR3E_LIMITS):
        if q[i] < lo - tol or q[i] > hi + tol:
            return False
    return True


def clamp_joints(q: List[float], margin: float = 1e-3) -> List[float]:
    return [
        float(np.clip(q[i], UR3E_LIMITS[i][0] + margin, UR3E_LIMITS[i][1] - margin))
        for i in range(6)
    ]


def load_table_bounds(scene_path: Optional[str] = None) -> Dict[str, float]:
    """Load table dimensions from obstacle_scene.yml to establish safety boundaries."""
    path = scene_path or DEFAULT_SCENE_PATH
    x_min, x_max = -0.45, 0.75
    y_min, y_max = -0.60, 0.60
    z_table_top, z_bottom = 0.015, -0.09
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            t = data["cuboid"]["table"]
            dims = t["dims"]
            pose = t["pose"]
            x_min = float(pose[0] - dims[0] / 2.0)
            x_max = float(pose[0] + dims[0] / 2.0)
            y_min = float(pose[1] - dims[1] / 2.0)
            y_max = float(pose[1] + dims[1] / 2.0)
            z_real_top = float(pose[2] + dims[2] / 2.0)
            # Red armor surface raised +1.0cm: Z_table_armor = +0.020m
            z_table_top = z_real_top + 0.010
            z_bottom = float(pose[2] - dims[2] / 2.0)
        except Exception:
            pass
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_table_top": z_table_top, "z_bottom": z_bottom,
        "z_floor": z_table_top,
    }


@jaxls.Cost.factory
def table_barrier_residual(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var,
    tcp_idx: jnp.ndarray,
    cam_idx: jnp.ndarray,
    table_bounds: jnp.ndarray,  # [x_min, x_max, y_min, y_max, z_table_top, z_floor]
    safety_margin: float = 0.0,  # 0.0mm WYSIWYG: 1cm armor is already embedded in red mesh geometry
    weight_floor: float = 80.0,
    weight_table: float = 120.0,
) -> jnp.ndarray:
    """Penalty function preventing TCP and Camera red armor from penetrating table/floor surfaces."""
    q = vals[joint_var]
    Ts = robot.forward_kinematics(q)
    p_tcp = Ts[tcp_idx, 4:7]
    quat_cam = Ts[cam_idx, 0:4]
    p_cam = Ts[cam_idx, 4:7]

    # Analytical OBB lowest-point calculation for Camera RealSense D455 red armor (+1.0cm all sides)
    w, x, y, z = quat_cam[0], quat_cam[1], quat_cam[2], quat_cam[3]
    R20 = 2.0 * (x * z - w * y)
    R21 = 2.0 * (y * z + w * x)
    R22 = 1.0 - 2.0 * (x * x + y * y)

    # Box center: [0, 0, 0.013], half-extents of red armor (+1.0cm): [0.072, 0.025, 0.025]
    c_z_cam = p_cam[2] + R22 * 0.013
    ext_z_cam = jnp.abs(R20) * 0.072 + jnp.abs(R21) * 0.025 + jnp.abs(R22) * 0.025
    z_cam_armor_min = c_z_cam - ext_z_cam

    # Lowest point of TCP red armor (+1.0cm below tool0 flange)
    z_tcp_armor_min = p_tcp[2] - 0.010

    x_min, x_max = table_bounds[0], table_bounds[1]
    y_min, y_max = table_bounds[2], table_bounds[3]
    z_top = table_bounds[4]      # Z_table_armor (+0.015m)
    z_floor = table_bounds[5]    # Z_table_armor (+0.015m)

    z_safe_table = z_top + safety_margin
    z_safe_floor = z_floor + safety_margin

    # 1. Floor constraint (z >= z_floor + margin)
    vf_tcp = jnp.maximum(0.0, z_safe_floor - z_tcp_armor_min)
    vf_cam = jnp.maximum(0.0, z_safe_floor - z_cam_armor_min)

    # 2. Table surface constraint: extended XY bounds and Z ceiling
    in_table_xy_tcp = jnp.logical_and(
        jnp.logical_and(p_tcp[0] >= x_min - safety_margin, p_tcp[0] <= x_max + safety_margin),
        jnp.logical_and(p_tcp[1] >= y_min - safety_margin, p_tcp[1] <= y_max + safety_margin)
    )
    vt_tcp = jnp.where(in_table_xy_tcp, jnp.maximum(0.0, z_safe_table - z_tcp_armor_min), 0.0)

    in_table_xy_cam = jnp.logical_and(
        jnp.logical_and(p_cam[0] >= x_min - safety_margin, p_cam[0] <= x_max + safety_margin),
        jnp.logical_and(p_cam[1] >= y_min - safety_margin, p_cam[1] <= y_max + safety_margin)
    )
    vt_cam = jnp.where(in_table_xy_cam, jnp.maximum(0.0, z_safe_table - z_cam_armor_min), 0.0)

    return jnp.array([
        (vf_tcp + vf_cam) * weight_floor,
        (vt_tcp + vt_cam) * weight_table
    ])


@jaxls.Cost.factory
def joint_limit_barrier_residual(
    vals: jaxls.VarValues,
    joint_var: jaxls.Var,
    lower_limits: jnp.ndarray,
    upper_limits: jnp.ndarray,
    barrier_buffer: float = 0.05,
    barrier_weight: float = 20.0,
    violation_weight: float = 500.0,
) -> jnp.ndarray:
    """
    Joint Limit Repulsive Cost / Barrier:
    Exerts smooth repulsive quadratic force when any joint comes within 0.05 rad of its limits,
    allowing the optimizer to naturally steer other DOFs to track Cartesian target while preventing
    limit violation.
    """
    q = vals[joint_var]
    rep_lower = jnp.maximum(0.0, (lower_limits + barrier_buffer) - q)
    rep_upper = jnp.maximum(0.0, q - (upper_limits - barrier_buffer))
    viol_lower = jnp.maximum(0.0, lower_limits - q)
    viol_upper = jnp.maximum(0.0, q - upper_limits)
    res_barrier = (rep_lower + rep_upper) * barrier_weight
    res_viol = (viol_lower + viol_upper) * violation_weight
    return jnp.concatenate([res_barrier, res_viol])


class PyrokiIKSolver:
    """
    Real-time IK/FK solver for UR3e robot using PyRoki on CPU.
    Includes 3-type singularity detection (Wrist / Elbow / Shoulder).
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        scene_path: Optional[str] = None,
        tcp_link: str = "tool0",
        disable_self_coll: Optional[bool] = None,
        coll_margin: Optional[float] = None,
        coll_weight: Optional[float] = None,
        clearance_threshold: Optional[float] = None,
    ):
        self.urdf_path = urdf_path or DEFAULT_URDF_PATH
        self.scene_path = scene_path or DEFAULT_SCENE_PATH
        self.tcp_link = tcp_link

        # Bypass & Collision Margin config (supports both arguments and env vars)
        if disable_self_coll is None:
            self.disable_self_coll = os.environ.get("DISABLE_SELF_COLLISION", "0").lower() in ("1", "true", "yes")
        else:
            self.disable_self_coll = bool(disable_self_coll)

        if coll_margin is None:
            self.coll_margin = float(os.environ.get("COLLISION_MARGIN", "0.01"))
        else:
            self.coll_margin = float(coll_margin)

        if coll_weight is None:
            self.coll_weight = float(os.environ.get("COLLISION_WEIGHT", "40.0"))
        else:
            self.coll_weight = float(coll_weight)

        if clearance_threshold is None:
            self.clearance_threshold = float(os.environ.get("CLEARANCE_THRESHOLD", "0.01"))
        else:
            self.clearance_threshold = float(clearance_threshold)

        self._last_coll_log_time = 0.0
        self._last_coll_was_safe = True

        bounds_dict = load_table_bounds(self.scene_path)
        self.table_bounds = jnp.array([
            bounds_dict["x_min"], bounds_dict["x_max"],
            bounds_dict["y_min"], bounds_dict["y_max"],
            bounds_dict["z_table_top"], bounds_dict["z_floor"]
        ], dtype=jnp.float32)

        print(f"[PyRoki] Loading robot model from URDF: {self.urdf_path}...")
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            filename_handler=_urdf_filename_handler,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )

        # Inject UR3E_LIMITS into the URDF model with soft buffer (margin = 0.02 rad ~ 1.15 deg)
        # so pk.costs.limit_cost auto-enforces safe ranges smoothly before reaching the hard boundary
        SOFT_LIMIT_MARGIN = 0.02
        for name, (lo, hi) in zip(_ACTUATED_JOINT_NAMES, UR3E_LIMITS):
            if name in self.urdf.joint_map and self.urdf.joint_map[name].limit is not None:
                self.urdf.joint_map[name].limit.lower = lo + SOFT_LIMIT_MARGIN
                self.urdf.joint_map[name].limit.upper = hi - SOFT_LIMIT_MARGIN

        self.robot = pk.Robot.from_urdf(self.urdf)

        if self.tcp_link not in self.robot.links.names:
            raise ValueError(f"Link '{self.tcp_link}' not found in URDF. Available links: {self.robot.links.names}")

        self.tcp_idx = jnp.array(self.robot.links.names.index(self.tcp_link))

        # Auto-detect camera / wrist-mount link
        if "d455_link" in self.robot.links.names:
            self.cam_idx = jnp.array(self.robot.links.names.index("d455_link"))
        elif "camera_mount_link" in self.robot.links.names:
            self.cam_idx = jnp.array(self.robot.links.names.index("camera_mount_link"))
        else:
            self.cam_idx = self.tcp_idx

        # Full self-collision model setup (Camera D455 + Mount, Wrist 1/2/3, Body links)
        coll_full = RobotCollision.from_urdf(self.urdf)

        body_links = {"base_link_inertia", "shoulder_link", "upper_arm_link"}
        wrist_links = {"wrist_1_link", "wrist_2_link", "wrist_3_link"}

        monitored_pair_tuples = []
        # 1. Camera D455 (red box) vs all body + forearm + wrist_1/2 + motors (8 pairs)
        for b in body_links | {"forearm_link", "wrist_1_link", "wrist_2_link", "wrist_1_motor_link", "wrist_2_motor_link"}:
            monitored_pair_tuples.append(("d455_link", b))

        # 2. Camera mount vs body links + forearm + wrist_1 + wrist_1_motor (6 pairs)
        for b in body_links | {"forearm_link", "wrist_1_link", "wrist_1_motor_link"}:
            monitored_pair_tuples.append(("camera_mount_link", b))

        # 3. Wrist links (1, 2, 3) vs body links (shoulder, base, upper arm - 9 pairs)
        for w in wrist_links:
            for b in body_links:
                monitored_pair_tuples.append((w, b))

        # 4. Wrist_3 vs forearm (1 pair)
        monitored_pair_tuples.append(("wrist_3_link", "forearm_link"))

        # 5. Forearm vs shoulder & base (prevents forearm folding into body - 2 pairs)
        monitored_pair_tuples.append(("forearm_link", "shoulder_link"))
        monitored_pair_tuples.append(("forearm_link", "base_link_inertia"))

        # 6. Wrist_2 vs forearm blind-spot fix (1 pair)
        monitored_pair_tuples.append(("wrist_2_link", "forearm_link"))

        # 7. Wrist_1 motor dummy vs body links (3 pairs)
        for b in body_links:
            monitored_pair_tuples.append(("wrist_1_motor_link", b))

        # 8. Wrist_2 motor dummy vs body + forearm (4 pairs)
        for b in body_links | {"forearm_link"}:
            monitored_pair_tuples.append(("wrist_2_motor_link", b))

        filtered_i, filtered_j = [], []
        for n_i, n_j in monitored_pair_tuples:
            if n_i in coll_full.link_names and n_j in coll_full.link_names:
                filtered_i.append(coll_full.link_names.index(n_i))
                filtered_j.append(coll_full.link_names.index(n_j))

        # Standardize camera_mount_link flange base capsule in coll_full to avoid mathematical ghost overlap at home
        coll_geom_caps = coll_full.coll
        if "camera_mount_link" in coll_full.link_names:
            idx_mt_caps = coll_full.link_names.index("camera_mount_link")
            new_size = coll_geom_caps.size.at[idx_mt_caps, 0].set(0.040).at[idx_mt_caps, 1].set(0.021)
            new_pos = coll_geom_caps.pose.translation().at[idx_mt_caps].set(jnp.array([0.0, 0.0, 0.003]))
            new_pose = jaxlie.SE3.from_rotation_and_translation(
                coll_geom_caps.pose.rotation(),
                new_pos
            )
            coll_geom_caps = Capsule(pose=new_pose, size=new_size)

        # Use exact bounding geometry from URDF collision tags, synchronized with Viser UI display
        self.coll_robot = RobotCollision(
            num_links=coll_full.num_links,
            link_names=coll_full.link_names,
            active_idx_i=tuple(filtered_i),
            active_idx_j=tuple(filtered_j),
            coll=coll_geom_caps,
            _geom_to_link_idx=coll_full._geom_to_link_idx,
        )
        self.coll_camera = self.coll_robot  # Backward-compatible alias
        self._dist_jit = jax.jit(lambda q: self.coll_robot.compute_self_collision_distance(self.robot, q))

        @jax.jit
        def _min_coll(q_arr):
            dists = self.coll_robot.compute_self_collision_distance(self.robot, q_arr)
            min_idx = jnp.argmin(dists)
            return dists[min_idx], min_idx

        self._min_coll_jit = _min_coll

        # ── Comprehensive 100% Boundary Check for camera_mount ──
        # Dense hull points covering entire 3D volume (flange base, neck, ears, screw tabs, corners)
        mount_mesh_path = os.path.join(_BASE_DIR, "assets", "robot_camera_mount_1.stl")
        try:
            import trimesh
            mesh_mt = trimesh.load(mount_mesh_path)
            hull = mesh_mt.convex_hull
            pts_surf = trimesh.sample.sample_surface(hull, 64)[0]
            corners = np.array([
                [-0.058, 0.058, 0.0], [0.058, 0.058, 0.0],
                [-0.058, 0.058, 0.006], [0.058, 0.058, 0.006],
                [-0.058, 0.076, 0.0], [0.058, 0.076, 0.0],
                [-0.058, 0.076, 0.006], [0.058, 0.076, 0.006],
                [-0.027, 0.096, 0.0], [0.027, 0.096, 0.0],
                [-0.027, 0.096, 0.006], [0.027, 0.096, 0.006],
                [-0.032, -0.031, 0.0], [0.032, -0.031, 0.0],
                [-0.032, -0.031, 0.006], [0.032, -0.031, 0.006],
                [-0.032, 0.0, 0.0], [0.032, 0.0, 0.0],
                [-0.032, 0.0, 0.006], [0.032, 0.0, 0.006],
                [0.0, 0.096, 0.003], [0.0, -0.031, 0.003]
            ], dtype=np.float32)
            self._mount_hull_pts = jnp.array(np.vstack([pts_surf, corners]), dtype=jnp.float32)
        except Exception:
            self._mount_hull_pts = jnp.array([
                [-0.058, 0.058, 0.0], [0.058, 0.058, 0.0],
                [-0.058, 0.058, 0.006], [0.058, 0.058, 0.006],
                [-0.058, 0.076, 0.0], [0.058, 0.076, 0.0],
                [-0.058, 0.076, 0.006], [0.058, 0.076, 0.006],
                [-0.027, 0.096, 0.0], [0.027, 0.096, 0.0],
                [-0.027, 0.096, 0.006], [0.027, 0.096, 0.006],
                [-0.032, -0.031, 0.0], [0.032, -0.031, 0.0],
                [-0.032, -0.031, 0.006], [0.032, -0.031, 0.006],
                [-0.032, 0.0, 0.0], [0.032, 0.0, 0.0],
                [-0.032, 0.0, 0.006], [0.032, 0.0, 0.006],
                [0.0, 0.096, 0.003], [0.0, -0.031, 0.003],
                [0.0, 0.0324, 0.003]
            ], dtype=jnp.float32)

        def _get_idx(name, fallback=0):
            return jnp.array(self.robot.links.names.index(name)) if name in self.robot.links.names else jnp.array(fallback)

        idx_mt = _get_idx("camera_mount_link", int(self.tcp_idx))
        idx_fa = _get_idx("forearm_link")
        idx_w1 = _get_idx("wrist_1_link")
        idx_w2 = _get_idx("wrist_2_link")
        idx_ua = _get_idx("upper_arm_link")
        idx_w1m = _get_idx("wrist_1_motor_link", int(idx_w1))
        idx_w2m = _get_idx("wrist_2_motor_link", int(idx_w2))

        self._mount_target_names = [
            "forearm_link", "wrist_1_link", "wrist_2_link", "upper_arm_link",
            "wrist_1_motor_link", "wrist_2_motor_link"
        ]
        hull_pts = self._mount_hull_pts

        def _pt_to_seg_sq(p, a, b):
            ab = b - a
            ab_sq = jnp.dot(ab, ab)
            t = jnp.clip(jnp.dot(p - a, ab) / ab_sq, 0.0, 1.0)
            proj = a + t * ab
            diff = p - proj
            return jnp.dot(diff, diff)

        def _mount_chk_raw(q_arr):
            Ts = self.robot.forward_kinematics(q_arr)
            T_mt = jaxlie.SE3(Ts[idx_mt])
            pts_w = jax.vmap(T_mt.apply)(hull_pts)

            # 1. Forearm (length 0.2132m along X, radius 0.030m)
            T_fa = jaxlie.SE3(Ts[idx_fa])
            p_fa_0 = T_fa.apply(jnp.array([0.0, 0.0, 0.0]))
            p_fa_1 = T_fa.apply(jnp.array([-0.2132, 0.0, 0.0]))
            d_fa = jnp.sqrt(jnp.min(jax.vmap(lambda p: _pt_to_seg_sq(p, p_fa_0, p_fa_1))(pts_w)) + 1e-8) - 0.030

            # 2. Wrist 1 (cylinder along Y from -0.075 to -0.005, radius 0.030m)
            T_w1 = jaxlie.SE3(Ts[idx_w1])
            p_w1_0 = T_w1.apply(jnp.array([0.0, -0.075, 0.0]))
            p_w1_1 = T_w1.apply(jnp.array([0.0, -0.005, 0.0]))
            d_w1 = jnp.sqrt(jnp.min(jax.vmap(lambda p: _pt_to_seg_sq(p, p_w1_0, p_w1_1))(pts_w)) + 1e-8) - 0.030

            # 3. Wrist 2 (cylinder along Y from -0.035 to +0.035, radius 0.030m)
            T_w2 = jaxlie.SE3(Ts[idx_w2])
            p_w2_0 = T_w2.apply(jnp.array([0.0, -0.035, 0.0]))
            p_w2_1 = T_w2.apply(jnp.array([0.0, +0.035, 0.0]))
            d_w2 = jnp.sqrt(jnp.min(jax.vmap(lambda p: _pt_to_seg_sq(p, p_w2_0, p_w2_1))(pts_w)) + 1e-8) - 0.030

            # 4. Upper arm (length 0.24355m along X, radius 0.035m)
            T_ua = jaxlie.SE3(Ts[idx_ua])
            p_ua_0 = T_ua.apply(jnp.array([0.0, 0.0, 0.0]))
            p_ua_1 = T_ua.apply(jnp.array([-0.24355, 0.0, 0.0]))
            d_ua = jnp.sqrt(jnp.min(jax.vmap(lambda p: _pt_to_seg_sq(p, p_ua_0, p_ua_1))(pts_w)) + 1e-8) - 0.035

            # 5. Wrist 1 motor (cylinder along Y from -0.0325 to +0.0325, radius 0.033m)
            T_w1m = jaxlie.SE3(Ts[idx_w1m])
            p_w1m_0 = T_w1m.apply(jnp.array([0.0, -0.0325, 0.0]))
            p_w1m_1 = T_w1m.apply(jnp.array([0.0, +0.0325, 0.0]))
            d_w1m = jnp.sqrt(jnp.min(jax.vmap(lambda p: _pt_to_seg_sq(p, p_w1m_0, p_w1m_1))(pts_w)) + 1e-8) - 0.033

            # 6. Wrist 2 motor (sphere radius 0.031m at origin)
            T_w2m = jaxlie.SE3(Ts[idx_w2m])
            p_w2m = T_w2m.translation()
            d_w2m = jnp.sqrt(jnp.min(jnp.sum((pts_w - p_w2m)**2, axis=-1)) + 1e-8) - 0.031

            dists = jnp.array([d_fa, d_w1, d_w2, d_ua, d_w1m, d_w2m])
            min_idx = jnp.argmin(dists)
            return dists[min_idx], min_idx

        self._mount_chk_raw = _mount_chk_raw
        self._mount_chk_jit = jax.jit(_mount_chk_raw)

        # JIT-compiled collision distance gradient for tangential sliding projection
        def _min_dist_fn(q_arr):
            dists = self.coll_robot.compute_self_collision_distance(self.robot, q_arr)
            min_d_robot = jnp.min(dists)
            min_d_mount, _ = self._mount_chk_raw(q_arr)
            return jnp.minimum(min_d_robot, min_d_mount)

        self._coll_grad_jit = jax.jit(jax.grad(_min_dist_fn))

        # JIT-compiled table clearance check for TCP and Camera red armor (latency < 0.02ms)
        @jax.jit
        def _table_chk(q_arr, z_top):
            Ts = self.robot.forward_kinematics(q_arr)
            p_tcp = Ts[self.tcp_idx, 4:7]
            quat_cam = Ts[self.cam_idx, 0:4]
            p_cam = Ts[self.cam_idx, 4:7]

            w, x, y, z = quat_cam[0], quat_cam[1], quat_cam[2], quat_cam[3]
            R20 = 2.0 * (x * z - w * y)
            R21 = 2.0 * (y * z + w * x)
            R22 = 1.0 - 2.0 * (x * x + y * y)

            # Box center: [0, 0, 0.013], half-extents of red armor (+1.0cm): [0.072, 0.025, 0.025]
            c_z_cam = p_cam[2] + R22 * 0.013
            ext_z_cam = jnp.abs(R20) * 0.072 + jnp.abs(R21) * 0.025 + jnp.abs(R22) * 0.025
            z_cam_armor_min = c_z_cam - ext_z_cam

            # Lowest point of TCP red armor (+1.0cm below tool0 flange)
            z_tcp_armor_min = p_tcp[2] - 0.010

            clr_tcp = z_tcp_armor_min - z_top
            clr_cam = z_cam_armor_min - z_top
            return clr_tcp, clr_cam, z_tcp_armor_min, z_cam_armor_min

        self._table_chk_jit = _table_chk

        # JIT-compiled wrist center position for shoulder singularity detection
        # Wrist center = wrist_1_link origin position
        if "wrist_1_link" in self.robot.links.names:
            self._wrist_center_idx = jnp.array(self.robot.links.names.index("wrist_1_link"))
        else:
            self._wrist_center_idx = self.tcp_idx

        @jax.jit
        def _wrist_center_xy(q_arr):
            Ts = self.robot.forward_kinematics(q_arr)
            p_wrist = Ts[self._wrist_center_idx, 4:7]
            return jnp.sqrt(p_wrist[0]**2 + p_wrist[1]**2)

        self._wrist_center_xy_jit = _wrist_center_xy

        self.q_var = self.robot.joint_var_cls(0)
        self.default_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

        # Build JIT-compiled IK solver
        self._solve_jit = self._build_jit_solver()

        # JIT warm-up to eliminate first-frame latency
        print("[PyRoki] JIT compiling optimization graph (Warm-up)...")
        self._warm_up()
        print("[PyRoki] IK Solver ready!")

    def _build_jit_solver(self):
        robot = self.robot
        q_var = self.q_var
        tcp_idx = self.tcp_idx
        cam_idx = self.cam_idx
        table_bounds = self.table_bounds
        coll_robot = self.coll_robot

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
                pos_weight=60.0,
                ori_weight=20.0
            )
            lower_limits_j = jnp.array([lo for lo, hi in UR3E_LIMITS], dtype=jnp.float32)
            upper_limits_j = jnp.array([hi for lo, hi in UR3E_LIMITS], dtype=jnp.float32)
            cost_limit = joint_limit_barrier_residual(
                joint_var=q_var,
                lower_limits=lower_limits_j,
                upper_limits=upper_limits_j,
                barrier_buffer=0.05,
                barrier_weight=20.0,
                violation_weight=500.0,
            )
            # Ultra-low rest weight: just enough to prevent drift near singularities,
            # does NOT drag the solution back toward seed (allows 1-frame convergence)
            cost_rest = pk.costs.rest_cost(
                joint_var=q_var,
                rest_pose=seed_arr,
                weight=0.005
            )
            cost_barrier = table_barrier_residual(
                robot=robot,
                joint_var=q_var,
                tcp_idx=tcp_idx,
                cam_idx=cam_idx,
                table_bounds=table_bounds,
                safety_margin=0.0,
                weight_floor=1200.0,
                weight_table=1500.0,
            )
            # Self-collision avoidance cost
            eff_weight = 0.0 if self.disable_self_coll else self.coll_weight
            eff_margin = 0.0 if self.disable_self_coll else self.coll_margin
            cost_coll = pk.costs.self_collision_cost(
                robot=robot,
                robot_coll=coll_robot,
                joint_var=q_var,
                margin=eff_margin,
                weight=eff_weight,
            )
            problem = jaxls.LeastSquaresProblem([cost_pose, cost_limit, cost_rest, cost_barrier, cost_coll], [q_var]).analyze()
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
            d_res = self._dist_jit(sample_seed)
            if hasattr(d_res, "block_until_ready"):
                d_res.block_until_ready()
            m_res = self._min_coll_jit(sample_seed)
            if hasattr(m_res[0], "block_until_ready"):
                m_res[0].block_until_ready()
            if hasattr(self, "_mount_chk_jit"):
                mt_res = self._mount_chk_jit(sample_seed)
                if hasattr(mt_res[0], "block_until_ready"):
                    mt_res[0].block_until_ready()
            t_res = self._table_chk_jit(sample_seed, self.table_bounds[4])
            if hasattr(t_res[0], "block_until_ready"):
                t_res[0].block_until_ready()
            w_res = self._wrist_center_xy_jit(sample_seed)
            if hasattr(w_res, "block_until_ready"):
                w_res.block_until_ready()
            g_res = self._coll_grad_jit(sample_seed)
            if hasattr(g_res, "block_until_ready"):
                g_res.block_until_ready()
        except Exception as e:
            print(f"[PyRoki] Warm-up warning: {e}")

    def check_singularity(
        self,
        q: Union[List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Detect all 3 physical singularity types of UR3e:

        1. WRIST singularity: Axes of wrist_1 (q3) and wrist_3 (q5) become coplanar
           when sin(q4) ≈ 0, i.e. |q4| < 0.08 rad or ||q4| - π| < 0.08 rad.
        2. ELBOW singularity / Max Reach: Arm fully extended, sin(q2) ≈ 0,
           i.e. |q2| < 0.10 rad or R_xy >= 0.51m.
        3. SHOULDER / Overhead singularity: Wrist center directly above base axis,
           sqrt(p_wrist_x² + p_wrist_y²) < 0.04m.

        Returns:
            {
                "singular": bool (any singularity active),
                "types": list of active singularity type strings,
                "damping": float (1.0 = no damping, < 1.0 = near singularity),
                "details": str (human-readable description),
                "wrist_sin_q4": float, "elbow_sin_q2": float, "shoulder_r_xy": float,
            }
        """
        q4 = float(q[4])  # wrist_2_joint
        q2 = float(q[2])  # elbow_joint

        sin_q4 = abs(math.sin(q4))
        sin_q2 = abs(math.sin(q2))

        # Compute wrist center XY distance from base axis
        q_arr = jnp.array(q, dtype=jnp.float32)
        shoulder_r_xy = float(self._wrist_center_xy_jit(q_arr))

        # Analytical planar reach from shoulder axis: r_arm = |a2*cos(q1) + a3*cos(q1+q2)|
        # For UR3e, a2 = 0.24365m, a3 = 0.21325m, and shoulder offset d2 = 0.13105m
        q1 = float(q[1])
        r_arm = abs(0.24365 * math.cos(q1) + 0.21325 * math.cos(q1 + q2))
        r_effective = min(shoulder_r_xy, r_arm)

        types = []
        details_parts = []

        # 1. Wrist Singularity: alpha ramps 0.0 -> 1.0 as sin_q4 goes from 0.20 rad down to 0.03 rad
        WRIST_OUTER = 0.20   # rad (~11.5 deg)
        WRIST_INNER = 0.03   # rad (~1.7 deg)
        if sin_q4 >= WRIST_OUTER:
            raw_wrist = 0.0
        elif sin_q4 <= WRIST_INNER:
            raw_wrist = 1.0
        else:
            raw_wrist = (WRIST_OUTER - sin_q4) / (WRIST_OUTER - WRIST_INNER)
        alpha_wrist = raw_wrist * raw_wrist * (3.0 - 2.0 * raw_wrist)

        if alpha_wrist > 0.05:
            types.append("WRIST")
            details_parts.append(f"WRIST: sin(q4)={sin_q4:.4f} (q4={q4:.3f} rad, alpha={alpha_wrist:.2f})")

        # 2. Elbow Singularity / Max Reach: alpha ramps 0.0 -> 1.0 as sin_q2 goes from 0.25 rad down to 0.05 rad
        ELBOW_OUTER = 0.25   # rad (~14.3 deg)
        ELBOW_INNER = 0.05   # rad (~2.9 deg)
        if sin_q2 >= ELBOW_OUTER:
            raw_elbow = 0.0
        elif sin_q2 <= ELBOW_INNER:
            raw_elbow = 1.0
        else:
            raw_elbow = (ELBOW_OUTER - sin_q2) / (ELBOW_OUTER - ELBOW_INNER)
        alpha_elbow = raw_elbow * raw_elbow * (3.0 - 2.0 * raw_elbow)

        REACH_LIMIT = 0.51  # meters
        REACH_START = 0.48
        if shoulder_r_xy >= REACH_LIMIT:
            alpha_elbow = 1.0
        elif shoulder_r_xy > REACH_START:
            r_t = (shoulder_r_xy - REACH_START) / (REACH_LIMIT - REACH_START)
            alpha_elbow = max(alpha_elbow, r_t * r_t * (3.0 - 2.0 * r_t))

        if alpha_elbow > 0.05:
            types.append("ELBOW")
            details_parts.append(f"ELBOW: sin(q2)={sin_q2:.4f}, R_xy={shoulder_r_xy:.3f}m (q2={q2:.3f} rad, alpha={alpha_elbow:.2f})")

        # 3. Shoulder / Overhead Singularity: alpha ramps 0.0 -> 1.0 as r_effective goes from 0.05m down to 0.02m
        SHOULDER_OUTER = 0.05  # meters
        SHOULDER_INNER = 0.02  # meters
        if r_effective >= SHOULDER_OUTER:
            raw_shoulder = 0.0
        elif r_effective <= SHOULDER_INNER:
            raw_shoulder = 1.0
        else:
            raw_shoulder = (SHOULDER_OUTER - r_effective) / (SHOULDER_OUTER - SHOULDER_INNER)
        alpha_shoulder = raw_shoulder * raw_shoulder * (3.0 - 2.0 * raw_shoulder)

        if alpha_shoulder > 0.05:
            types.append("SHOULDER")
            details_parts.append(f"SHOULDER: overhead r={r_effective:.4f}m (alpha={alpha_shoulder:.2f})")

        # Combined continuous alpha in [0.0, 1.0]
        alpha = max(alpha_wrist, alpha_elbow, alpha_shoulder)
        # Continuous smooth damping factor: 1.0 (no singularity) down to 0.75 (deep singularity)
        damping = 1.0 - 0.25 * alpha

        is_singular = alpha > 0.05
        details = " | ".join(details_parts) if details_parts else "No singularity detected"

        return {
            "singular": is_singular,
            "alpha": float(alpha),
            "types": types,
            "damping": float(damping),
            "details": details,
            "wrist_sin_q4": float(sin_q4),
            "elbow_sin_q2": float(sin_q2),
            "shoulder_r_xy": float(r_effective),
        }

    def check_table_collision(
        self,
        q: Union[List[float], np.ndarray],
        z_safe_clearance: Optional[float] = None,
        tcp_safe_clearance: float = 0.0,
        cam_safe_clearance: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Fast clearance check from TCP (tool0) and Camera lowest point to table surface.
        Returns safe=True if both components are above table armor (default 0.0mm clearance, WYSIWYG).
        Computation latency < 0.02ms.
        """
        if z_safe_clearance is not None:
            tcp_safe_clearance = z_safe_clearance
            cam_safe_clearance = z_safe_clearance

        q_arr = jnp.array(q, dtype=jnp.float32)
        clr_tcp, clr_cam, z_tcp, z_cam_min = self._table_chk_jit(q_arr, self.table_bounds[4])
        clr_tcp_f = float(clr_tcp)
        clr_cam_f = float(clr_cam)
        tcp_safe = clr_tcp_f >= (tcp_safe_clearance - 3e-4)
        cam_safe = clr_cam_f >= (cam_safe_clearance - 3e-4)
        safe = bool(tcp_safe and cam_safe)
        min_clr = min(clr_tcp_f, clr_cam_f)
        violator = "Camera" if clr_cam_f < clr_tcp_f else "TCP"
        return {
            "safe": safe,
            "min_clearance": min_clr,
            "clearance_cm": min_clr * 100.0,
            "violator": violator,
            "tcp_clearance_cm": clr_tcp_f * 100.0,
            "cam_clearance_cm": clr_cam_f * 100.0,
            "z_tcp": float(z_tcp),
            "z_cam_min": float(z_cam_min),
        }

    def check_self_collision(
        self,
        q: Union[List[float], np.ndarray],
        clearance_threshold: Optional[float] = None,
        verbose: bool = True,
        force_log: bool = False,
    ) -> Dict[str, Any]:
        """
        Fast self-collision distance check across all monitored link pairs.
        Uses JIT-compiled collision model (latency < 0.05ms).
        Logs detailed debug info when two links are approaching each other (< 2.5cm).
        """
        threshold = clearance_threshold if clearance_threshold is not None else self.clearance_threshold

        if getattr(self, "disable_self_coll", False):
            return {
                "safe": True,
                "min_dist": 999.0,
                "clearance_cm": 999.0,
                "active_pairs": 0,
                "link_a": "none",
                "link_b": "none",
                "min_idx": -1,
            }

        q_arr = jnp.array(q, dtype=jnp.float32)
        min_d_jax, min_idx_jax = self._min_coll_jit(q_arr)
        min_d = float(min_d_jax)
        min_idx = int(min_idx_jax)
        link_a = self.coll_robot.link_names[self.coll_robot.active_idx_i[min_idx]]
        link_b = self.coll_robot.link_names[self.coll_robot.active_idx_j[min_idx]]

        # Rigorous 100% boundary check for camera_mount against all arm links
        if hasattr(self, "_mount_chk_jit"):
            min_d_mt_jax, mt_target_idx_jax = self._mount_chk_jit(q_arr)
            min_d_mt = float(min_d_mt_jax)
            if min_d_mt < min_d:
                min_d = min_d_mt
                link_a = "camera_mount_link"
                link_b = self._mount_target_names[int(mt_target_idx_jax)]

        is_safe = bool(min_d >= (threshold - 1e-5))

        # Debug logging: print when two links approach each other (< 2.5cm) or on collision
        now = time.time()
        should_log = (
            verbose
            and (min_d < 0.025 or not is_safe)
            and (now - self._last_coll_log_time >= 0.15 or (not is_safe and self._last_coll_was_safe) or force_log)
        )
        if should_log:
            self._last_coll_log_time = now
            self._last_coll_was_safe = is_safe
            status_str = "Allowed" if is_safe else "BLOCKED"
            print(
                f"[Self-Collision] Distance between {link_a} and {link_b}: "
                f"{min_d:.3f}m. Safe threshold: {threshold:.3f}m -> {status_str}"
            )

        return {
            "safe": is_safe,
            "min_dist": min_d,
            "clearance_cm": min_d * 100.0,
            "active_pairs": len(self.coll_robot.active_idx_i),
            "link_a": link_a,
            "link_b": link_b,
            "min_idx": min_idx,
        }

    # Backward-compatible alias
    check_camera_self_collision = check_self_collision

    def fk(self, joints: List[float]) -> Dict[str, Any]:
        """
        Forward Kinematics: compute TCP position [x, y, z] and quaternion [w, x, y, z] from 6 joint angles.
        """
        q_arr = jnp.array(joints, dtype=jnp.float32)
        fk_all = self.robot.forward_kinematics(q_arr)
        pose_7d = np.array(fk_all[int(self.tcp_idx)])
        # pose_7d format: [qw, qx, qy, qz, x, y, z]
        quat_wxyz = [float(pose_7d[0]), float(pose_7d[1]), float(pose_7d[2]), float(pose_7d[3])]
        pos_xyz = [float(pose_7d[4]), float(pose_7d[5]), float(pose_7d[6])]
        return {"pos": pos_xyz, "quat": quat_wxyz}

    def solve(
        self,
        pose: Union[tuple, list, Dict[str, Any]],
        q_seed: Optional[List[float]] = None,
        arm_anchor: Optional[Union[List[float], np.ndarray]] = None,
        is_pure_rot: bool = False,
    ) -> Dict[str, Any]:
        """
        Real-time Inverse Kinematics using PyRoki JIT on CPU.

        Args:
            pose: tuple (pos, quat_wxyz) or dict {"pos": ..., "quat": ...}
            q_seed: 6 joint angles from previous frame as warm-start seed
            arm_anchor: Optional [q0, q1, q2] body joint anchor configuration to prevent arm/base swing during rotation
            is_pure_rot: Boolean indicating pure rotation mode (no commanded translation)

        Returns:
            {"q": list[6 rad] or None, "latency_ms": float, "solver": "pyroki", ...}
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

        # q_seed Hygiene: Ensure elbow seed has positive bias when near/past zero
        # to prevent gradient descent from getting trapped in negative boundary barrier
        effective_seed = list(ref_seed)
        if effective_seed[2] < 0.02:
            effective_seed[2] = 0.025
        seed_arr = jnp.array(effective_seed, dtype=jnp.float32)

        # Convert to jax arrays
        target_pos_j = jnp.array(pos, dtype=jnp.float32)
        target_quat_j = jnp.array(quat, dtype=jnp.float32)  # [w, x, y, z]
        # Normalize quaternion to prevent NaN
        norm_q = jnp.linalg.norm(target_quat_j)
        if norm_q > 1e-6:
            target_quat_j = target_quat_j / norm_q
        else:
            target_quat_j = jnp.array([1.0, 0.0, 0.0, 0.0])

        try:
            sol_q = self._solve_jit(target_pos_j, target_quat_j, seed_arr)
            raw_q = np.array(sol_q, dtype=float).tolist()
            norm_q = normalize_to_current(raw_q, ref_seed)

            # Smooth Per-Joint Step Limiting: Bảo toàn tính liên tục vận tốc C1/C2
            delta_q_raw = np.array(norm_q) - np.array(ref_seed)
            dq_clamped = np.copy(delta_q_raw)

            # Bảo vệ khớp khuỷu (elbow) gần 0 rad để triệt tiêu branch-flip và nổ Jacobian
            if abs(ref_seed[2]) < 0.08:
                dq_clamped[2] = np.clip(delta_q_raw[2], -0.06, 0.06)

            # Giới hạn bước nhảy góc xoay cổ tay per-frame để triệt tiêu hoàn toàn giật cục và xoắn gập:
            MAX_WRIST_STEP = float(os.environ.get("MAX_WRIST_STEP", "0.14"))  # rad/step (~8.0°/step)
            dq_clamped[3] = np.clip(delta_q_raw[3], -MAX_WRIST_STEP, MAX_WRIST_STEP)
            dq_clamped[4] = np.clip(delta_q_raw[4], -MAX_WRIST_STEP, MAX_WRIST_STEP)
            dq_clamped[5] = np.clip(delta_q_raw[5], -MAX_WRIST_STEP, MAX_WRIST_STEP)

            # Khi đang ở chế độ xoay thuần túy hoặc có Arm Anchor:
            # Kẹp chặt bước nhảy per-frame của thân máy để chống giật
            if is_pure_rot or arm_anchor is not None:
                dq_clamped[0] = np.clip(delta_q_raw[0], -0.008, 0.008)
                dq_clamped[1] = np.clip(delta_q_raw[1], -0.015, 0.015)
                dq_clamped[2] = np.clip(delta_q_raw[2], -0.015, 0.015)
            else:
                # Khi đang tịnh tiến: Giới hạn bước dịch chuyển khớp thân per-frame đầm chắc, mượt mà giống bản stable-22-10
                MAX_BODY_STEP = float(os.environ.get("MAX_BODY_STEP", "0.060"))  # rad/step (~3.4°/step)
                dq_clamped[0] = np.clip(delta_q_raw[0], -MAX_BODY_STEP, MAX_BODY_STEP)
                dq_clamped[1] = np.clip(delta_q_raw[1], -MAX_BODY_STEP, MAX_BODY_STEP)
                dq_clamped[2] = np.clip(delta_q_raw[2], -MAX_BODY_STEP, MAX_BODY_STEP)

            q_candidate = (np.array(ref_seed) + dq_clamped).tolist()

            # Ràng buộc biên độ tuyệt đối của Khớp Thân so với Mỏ Neo (Hard Excursion Envelope):
            # Base q0 <= +-0.006 rad (~0.34 deg), Shoulder/Elbow <= +-0.018 rad (~1.03 deg)
            body_swing_blocked = False
            if is_pure_rot or arm_anchor is not None:
                anchor_body = np.array(arm_anchor[:3] if arm_anchor is not None else ref_seed[:3], dtype=float)
                EXCURSION_LIMITS = np.array([0.006, 0.018, 0.018], dtype=float)
                body_err = np.abs(np.array(q_candidate[:3]) - anchor_body)
                if np.any(body_err >= EXCURSION_LIMITS - 1e-4):
                    body_swing_blocked = True
                    for b_i in range(3):
                        q_candidate[b_i] = float(np.clip(
                            q_candidate[b_i],
                            anchor_body[b_i] - EXCURSION_LIMITS[b_i],
                            anchor_body[b_i] + EXCURSION_LIMITS[b_i]
                        ))

            # Post-solve FK position error check
            fk_check = self.fk(q_candidate)
            pos_err = np.linalg.norm(np.array(fk_check["pos"]) - np.array(pos))

            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Per-joint soft clamping: Never hard-reject a solution due to joint limits
            lower_limits = np.array([lo for lo, hi in UR3E_LIMITS], dtype=float)
            upper_limits = np.array([hi for lo, hi in UR3E_LIMITS], dtype=float)
            SAFE_EPS = 1e-3

            LIMIT_ZONE = 0.03  # Limit zone for flagging JOINT LIMIT status (30 mrad)
            clamped_norm_q = np.clip(q_candidate, lower_limits + SAFE_EPS, upper_limits - SAFE_EPS).tolist()
            at_joint_limit = bool(np.any((np.array(clamped_norm_q) <= lower_limits + LIMIT_ZONE) |
                                         (np.array(clamped_norm_q) >= upper_limits - LIMIT_ZONE))) or body_swing_blocked

            # Collision safety check (Self-collision & Table-collision) at 0.0mm threshold (WYSIWYG)
            self_chk = self.check_self_collision(norm_q, verbose=False)
            tbl_chk = self.check_table_collision(norm_q, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
            if self_chk["safe"] and tbl_chk["safe"]:
                self_chk_cand = self.check_self_collision(clamped_norm_q, verbose=False)
                tbl_chk_cand = self.check_table_collision(clamped_norm_q, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
                is_safe = self_chk_cand["safe"] and tbl_chk_cand["safe"]
                if not is_safe:
                    self_chk = self_chk_cand
                    tbl_chk = tbl_chk_cand
            else:
                is_safe = False


            # Orientation error measurement
            q_act = np.array(fk_check["quat"])
            q_tgt = np.array(quat)
            q_tgt_norm = np.linalg.norm(q_tgt)
            if q_tgt_norm > 1e-6:
                q_tgt = q_tgt / q_tgt_norm
            cos_half = min(1.0, max(-1.0, abs(float(np.dot(q_tgt, q_act)))))
            ori_err_rad = float(2.0 * math.acos(cos_half))

            # ── ELASTIC TRACKING & JOINT VELOCITY SCALING ──
            # Never reject a solution due to joint limits. Clamped joint stays at boundary,
            # remaining 5 joints slide freely along boundary to escape instantly.
            if is_safe:
                delta_q = np.array(clamped_norm_q) - np.array(ref_seed)
                s_max = float(np.max(np.abs(delta_q)))

                # Singularity detection and continuous alpha blending
                sing_info = self.check_singularity(clamped_norm_q)
                alpha = sing_info.get("alpha", 0.0)
                if q_seed is not None:
                    try:
                        seed_sing = self.check_singularity(ref_seed)
                        alpha = max(alpha, seed_sing.get("alpha", 0.0))
                    except Exception:
                        pass

                # Continuous interpolation of s_allowed (khớp với mức độ đầm chắc, mượt mà của bản demo stable-22-10)
                base_s = float(os.environ.get("SOLVER_S_ALLOWED", "0.10" if not is_pure_rot else "0.14"))
                s_allowed = float(max(0.04, base_s - 0.03 * alpha))
                branch_flip_guard = 1.60  # rad (~92 deg) - branch-flip rejection threshold

                is_body_flip = float(np.max(np.abs(delta_q[:3]))) > 1.20

                # 1. Small step (<= s_allowed): accept in full
                if s_max <= s_allowed:
                    q_cmd = clamped_norm_q
                    is_scaled = False
                # 2. Medium step (or pure wrist rotation): Joint Velocity Scaling
                elif s_max <= branch_flip_guard or (not is_body_flip):
                    scale = s_allowed / s_max
                    q_cmd = (np.array(ref_seed) + delta_q * scale).tolist()
                    # Verify collision safety for the intermediate scaled point
                    self_cmd = self.check_self_collision(q_cmd, verbose=False)
                    tbl_cmd = self.check_table_collision(q_cmd, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
                    if not (self_cmd["safe"] and tbl_cmd["safe"]):
                        q_cmd = ref_seed
                    is_scaled = True
                # 3. Excessive jump (body branch flip): Branch-Flip Guard rejection
                else:
                    q_cmd = ref_seed
                    is_scaled = True

                # Step 1 Requirement: Per-joint clamping on q_cmd to ensure strict compliance
                q_cmd_clamped = np.clip(q_cmd, lower_limits + SAFE_EPS, upper_limits - SAFE_EPS).tolist()
                at_joint_limit = at_joint_limit or bool(np.any((np.array(q_cmd_clamped) <= lower_limits + LIMIT_ZONE) |
                                                               (np.array(q_cmd_clamped) >= upper_limits - LIMIT_ZONE))) or body_swing_blocked
                q_cmd = q_cmd_clamped

                # Strict post-clamp verification: ensure clamped configuration never penetrates
                self_cmd_final = self.check_self_collision(q_cmd, verbose=False)
                tbl_cmd_final = self.check_table_collision(q_cmd, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)
                if not (self_cmd_final["safe"] and tbl_cmd_final["safe"]):
                    q_cmd = ref_seed

                is_sliding = (pos_err >= 0.018) or at_joint_limit or body_swing_blocked

                # Singularity detection for the accepted solution
                sing_info = self.check_singularity(q_cmd)
                final_alpha = max(alpha, sing_info.get("alpha", 0.0))

                return {
                    "q": q_cmd,
                    "latency_ms": dt_ms,
                    "solver": "pyroki",
                    "pos_err_mm": pos_err * 1000.0,
                    "ori_err_rad": ori_err_rad,
                    "valid_joints": True,
                    "at_joint_limit": at_joint_limit,
                    "sliding_on_barrier": is_sliding,
                    "raw_q": norm_q,
                    "singularity": sing_info,
                    "singularity_alpha": float(final_alpha),
                    "body_swing_blocked": body_swing_blocked,
                }
            else:
                fail_reasons = []
                if not self_chk["safe"]:
                    fail_reasons.append(f"self_collision({self_chk.get('link_a')} vs {self_chk.get('link_b')}: {self_chk.get('min_dist')*1000:.1f}mm)")
                if not tbl_chk["safe"]:
                    fail_reasons.append(f"table_collision({tbl_chk.get('violator')}: {tbl_chk.get('min_clearance')*1000:.1f}mm)")

                return {
                    "q": None,
                    "latency_ms": dt_ms,
                    "solver": "pyroki",
                    "pos_err_mm": pos_err * 1000.0,
                    "ori_err_rad": ori_err_rad,
                    "valid_joints": True,
                    "at_joint_limit": at_joint_limit,
                    "sliding_on_barrier": False,
                    "fail_reason": " | ".join(fail_reasons),
                    "raw_q": norm_q,
                    "body_swing_blocked": body_swing_blocked,
                }

        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "q": None,
                "latency_ms": dt_ms,
                "solver": "pyroki",
                "pos_err_mm": 999.0,
                "valid_joints": False,
                "fail_reason": f"Exception: {str(e)}",
                "error": str(e),
            }
