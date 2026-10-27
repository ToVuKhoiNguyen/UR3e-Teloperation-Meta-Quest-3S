import os
import time
import math
import numpy as np
import transforms3d
from typing import Optional, Dict, Any, List, Union
from scipy.spatial.transform import Rotation as R

def isolate_rotation_axes(rotvec, snap_ratio=0.30, smooth_high=0.60, deadband=0.026):
    """
    Thuật toán lọc nhiễu trục xoay độc lập (Rotation Axis Isolation):
    - rotvec: mảng 3 phần tử [pitch, yaw, roll] (theo radian hoặc độ).
    - deadband: ngưỡng vi rung tĩnh (mặc định 0.026 rad ~ 1.5 độ). Nếu norm < deadband -> trả về [0, 0, 0].
    - snap_ratio: ngưỡng cô lập trục (mặc định 0.30, tương ứng trục chính > 3.3x trục phụ).
      Nếu tỷ số abs(rotvec[j]) / abs(rotvec[main]) < snap_ratio -> ép trục phụ về 0.0.
    - smooth_high: ngưỡng mở xoay 3D đa trục tự do (mặc định 0.60).
      Nếu snap_ratio <= ratio < smooth_high: làm mềm bằng đa thức Hermite smoothstep.
    - Bảo toàn 100% biên độ của trục chủ đạo.
    """
    v = np.array(rotvec, dtype=float)
    mag = np.linalg.norm(v)
    if mag < deadband:
        return np.zeros(3, dtype=float)

    abs_v = np.abs(v)
    max_idx = int(np.argmax(abs_v))
    max_val = abs_v[max_idx]

    if max_val < 1e-6:
        return np.zeros(3, dtype=float)

    out = np.zeros(3, dtype=float)
    out[max_idx] = v[max_idx]

    for i in range(3):
        if i == max_idx:
            continue
        ratio = abs_v[i] / max_val
        if ratio < snap_ratio:
            out[i] = 0.0
        elif ratio < smooth_high:
            t = (ratio - snap_ratio) / max(smooth_high - snap_ratio, 1e-6)
            scale = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep
            out[i] = v[i] * scale
        else:
            out[i] = v[i]

    return out

class DeltaPoseSource:
    """
    Implements PoseSource contract.
    Chuyển delta tọa độ từ kính Quest sang base_link của UR3e.
    """
    def __init__(self, server, ctrl, scale: float = 1.0, max_delta_m: float = 0.22, anchor_reference_func=None, alpha: float = 0.28):
        self.server = server
        self.ctrl = ctrl
        self.sensitivity_scale = float(os.environ.get("TELEOP_SCALE", str(scale if scale is not None else 1.0)))
        self.max_delta_m = float(os.environ.get("MAX_DELTA_M", str(max_delta_m)))
        self.anchor_reference_func = anchor_reference_func
        self.alpha = float(os.environ.get("TELEOP_ALPHA", str(alpha)))
        
        self._last_diag_msg = ""
        self._last_diag_time = 0

        self._anchor_pos_quest = None
        self._anchor_quat_quest_xyzw = None
        self._anchor_tcp = None
        self._anchor_quat_robot_xyzw = None
        self._anchor_theta_arm = None
        
        self._ema_pos = None
        self._ema_quat_xyzw = None
        
        self.enable_axis_snapping = os.environ.get("AXIS_SNAPPING", "1") == "1"
        self.nav_mode = os.environ.get("NAV_MODE", "cylindrical").lower()
        self.rotation_mode = os.environ.get("ROTATION_MODE", "decoupled_3axis").lower()
        # Hiệu chuẩn hướng bàn máy (Desk-Aligned):
        # Mặc định khi chưa hiệu chuẩn: [0, 0, -1] và [1, 0, 0]
        self._desk_u_fwd = np.array([0.0, 0.0, -1.0])
        self._desk_u_right = np.array([1.0, 0.0, 0.0])
        self._last_btn_b = False
        self._last_calib_combo = False

        raw_heading = os.environ.get("HEADING_MODE", "desk_aligned").lower()
        self.heading_mode = raw_heading
        self.axis_snap_thresh = float(os.environ.get("AXIS_SNAP_THRESH", "0.50"))
        self.rot_scale = float(os.environ.get("ROT_SCALE", "1.1"))
        self._current_delta_theta = 0.0
        self._anchor_u_fwd = np.copy(self._desk_u_fwd)
        self._anchor_u_right = np.copy(self._desk_u_right)
        self._last_dominant_trans_axis = None
        self.last_delta_trans = np.zeros(3, dtype=float)
        self.last_delta_mag = 0.0

        # Cấu hình chống nhiễu chuyên sâu (Anti-Noise & Jitter Suppression)
        self.rot_deadband = float(os.environ.get("ROT_DEADBAND", "0.026"))  # ~1.5 độ (triệt tiêu vi rung sinh học khi bóp giữ phím)
        self.pos_deadband = float(os.environ.get("POS_DEADBAND", "0.003"))  # 3mm (triệt tiêu vi rung khi giữ yên tay)
        self.fwd_deadband = float(os.environ.get("FWD_DEADBAND", "0.0015")) # 1.5mm (nhạy bén cho trục Tiến/Lùi dải nhỏ)
        self.lat_deadband = float(os.environ.get("LAT_DEADBAND", "0.0035")) # 3.5mm (triệt tiêu rung lắc ngang do bóp ngón trỏ)
        self.vert_deadband = float(os.environ.get("VERT_DEADBAND", "0.0035")) # 3.5mm (triệt tiêu vi rung nâng hạ tay)
        self.enable_cross_axis_damping = os.environ.get("CROSS_AXIS_DAMPING_ENABLE", "1") == "1"
        self.cone_deadzone_low = float(os.environ.get("CONE_DEADZONE_LOW", os.environ.get("CROSS_AXIS_THRESH_LOW", "0.40")))   # < 40%: ép trục phụ về 0.0 (rãnh khóa trục cứng cáp, chống lệch chéo)
        self.cone_deadzone_high = float(os.environ.get("CONE_DEADZONE_HIGH", os.environ.get("CROSS_AXIS_THRESH_HIGH", "0.70"))) # > 70%: mở khóa 100% đi chéo chủ ý
        self.cross_axis_low = self.cone_deadzone_low
        self.cross_axis_high = self.cone_deadzone_high
        self.cross_axis_damping = 0.0
        self.rot_snap_low = float(os.environ.get("ROT_SNAP_LOW", "0.30"))   # < 30%: triệt tiêu 100% trục phụ về 0.0 (Snap to single axis)
        self.rot_snap_high = float(os.environ.get("ROT_SNAP_HIGH", "0.60")) # > 60%: mở xoay 3D đa trục khi chủ ý bẻ chéo lớn
        self._last_dominant_rot_axis = None
        self._locked_axis_idx = None
        self.axis_lock_deadzone_rad = float(os.environ.get("AXIS_LOCK_DEADZONE_RAD", str(np.deg2rad(4.0))))  # 4 degrees ~ 0.06981 rad
        self.grip_rotation_gain = float(os.environ.get("GRIP_ROTATION_GAIN", "0.55"))  # Giảm 45% tốc độ xoay
        self.max_rot_step = float(os.environ.get("MAX_ROT_STEP", "0.12"))  # ~6.9 độ / 50ms (giảm 35-40% so với 0.20)
        self._last_output_quat_wxyz = None
        self.is_pure_rotation = False
        self.is_translating = False

        # Khởi tạo toàn bộ biến trạng thái phục vụ cơ chế xoay Grip (Clock-face & Single-Axis Lock)
        self.clutch_engaged = False
        self.grip_engaged = False
        self.R_ctrl_origin = None
        self.R_tcp_origin = None
        self.prev_target_quat = None

        # Kẹp biên độ góc quay tuyệt đối (Strict Absolute Angle Clamping) chống vặn xoắn thân máy:
        # Hạn chế tối đa góc quay trục Z (Yaw): mặc định ±35° (~0.61 rad) để ngăn chặn hoàn toàn việc vặn xoắn thân máy
        self.max_z_rotation_delta = float(os.environ.get("MAX_YAW_DELTA", os.environ.get("MAX_Z_ROTATION_DELTA", str(math.radians(35.0)))))
        self.max_roll_rotation_delta = float(os.environ.get("MAX_ROLL_DELTA", str(math.radians(85.0))))
        self.max_pitch_rotation_delta = float(os.environ.get("MAX_PITCH_DELTA", str(math.radians(65.0))))

        # Biến trạng thái liên tục góc (Angle Unwrapping) và chốt cố định hệ trục Clock-face
        self._clock_u_radial = None
        self._clock_u_tangential = None
        self._clock_u_up = None
        self._clock_u_tool = None
        self._last_gesture_angle = None
        self._last_locked_axis_idx = None
        self._last_hand_quat_xyzw = None
        self._prev_raw_target_quat_xyzw = None

        # Neo hệ 3 trục xoay bám theo tư thế động của đầu máy (Intrinsic Tool Frame theo chuẩn Home)
        self._R_home = None
        self._e_yaw_local = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._e_pitch_local = np.array([-0.743525, 0.668707, 0.0], dtype=np.float64)
        self._e_roll_local = np.array([-0.668707, -0.743525, 0.0], dtype=np.float64)

        # Trục xoay bị khóa đơn trục và kẹp giới hạn động với Zero Windup
        self.locked_rotation_axis = None
        self._axis_theta_max_pos = {}
        self._axis_theta_min_neg = {}
        self._gesture_angle_offset = 0.0

        self._last_button_state = 0
        self.out_of_reach_time = 0.0

        if hasattr(self.server, "set_q0_provider") and hasattr(self.ctrl, "joints"):
            try:
                self.server.set_q0_provider(lambda: self.ctrl.joints()[0] if self.ctrl is not None and hasattr(self.ctrl, "joints") and self.ctrl.joints() is not None else None)
            except Exception:
                pass

    @property
    def locked_axis_idx(self):
        return self._locked_axis_idx

    @locked_axis_idx.setter
    def locked_axis_idx(self, val):
        self._locked_axis_idx = val

    @property
    def scale(self):
        return self.sensitivity_scale

    @scale.setter
    def scale(self, val):
        self.sensitivity_scale = float(val)

    def _apply_aggressive_axis_isolation(self, delta_vec):
        """
        Khóa trục tịnh tiến thông minh (Soft Translation Snapping) & Bộ lọc triệt tiêu rò rỉ chéo:
        - Sử dụng một lớp deadband duy nhất (ở tầng ngoài): Triệt tiêu vi rung sinh học khi bóp giữ phím.
        - Chống đảo trục chập chờn (Hysteresis): Giữ ổn định trục chính nếu trục mới chưa vượt trội > 20%.
        - Triệt tiêu trục phụ với đường cong Hermite Smoothstep:
          + < 35%: Triệt tiêu hoàn toàn về 0.0 nhằm giữ đường bay thẳng tắp.
          + 35% - 70%: Chuyển tiếp mượt mà, không giật cục.
          + > 70%: Mở khóa tự do khi người dùng chủ ý di chuyển chéo đa trục.
        """
        vec = np.array(delta_vec, dtype=float)
        abs_v = np.abs(vec)
        curr_max_idx = int(np.argmax(abs_v))
        curr_max_val = abs_v[curr_max_idx]

        if curr_max_val < 1e-4:
            self._last_dominant_trans_axis = None
            return np.zeros(3)

        # Hysteresis: Giữ ổn định trục chính đã chọn nếu trục mới chưa vượt trội > 20%
        dominant_idx = curr_max_idx
        if self._last_dominant_trans_axis is not None and self._last_dominant_trans_axis != curr_max_idx:
            prev_val = abs_v[self._last_dominant_trans_axis]
            if prev_val > 0.80 * curr_max_val:
                dominant_idx = self._last_dominant_trans_axis
                curr_max_val = prev_val
        self._last_dominant_trans_axis = dominant_idx

        out = np.zeros(3)
        out[dominant_idx] = vec[dominant_idx]

        snap_thresh = self.axis_snap_thresh
        low_thresh = min(0.35, snap_thresh * 0.7)
        high_thresh = max(0.65, snap_thresh * 1.3)
        for i in range(3):
            if i == dominant_idx:
                continue
            ratio = abs_v[i] / (curr_max_val + 1e-6)
            if ratio < low_thresh:
                out[i] = 0.0
            elif ratio < high_thresh:
                t = (ratio - low_thresh) / (high_thresh - low_thresh)
                scale = t * t * (3.0 - 2.0 * t)
                out[i] = vec[i] * scale
            else:
                out[i] = vec[i]

        return out

    def _apply_bidirectional_cross_axis_damping(self, delta_vec):
        """
        Bộ lọc nhiễu rung tay Smooth Axial Cone Deadzone (SACD) - Phương án 1A (Direct Component Gating):
        - Xác định trục chính có lực đẩy lớn nhất (Dominant Axis) và giữ nguyên 100% biên độ cử động thật.
        - Trục phụ có tỷ số rò rỉ r <= CONE_DEADZONE_LOW (mặc định 0.20 ~ 11.3°): Triệt tiêu HOÀN TOÀN về 0.0.
          Giúp robot di chuyển thẳng tắp, không bị trôi dạt do rung lắc vô thức của tay người.
        - Khoảng [CONE_DEADZONE_LOW, CONE_DEADZONE_HIGH] (0.20 - 0.50): Chuyển tiếp mượt mà bằng đường cong
          Hermite Smoothstep bậc 3 từ 0.0 -> 1.0 (đạo hàm bậc 1 triệt tiêu ở 2 đầu biên, chống giật khựng).
        - Khi r >= CONE_DEADZONE_HIGH (mặc định 0.50 ~ 26.6°): Mở khóa 100% cho phép di chuyển chéo đa trục tự do.
        - Xử lý an toàn cực trị: Khi vector bằng 0 (người dùng đứng im), trả về vector 0 mà không phát sinh ZeroDivisionError.
        """
        vec = np.asarray(delta_vec, dtype=float)
        abs_v = np.abs(vec)
        max_val = float(np.max(abs_v))

        # Xử lý cực trị: người dùng đứng im hoặc vi dịch chuyển cực nhỏ < 1e-6 -> trả về an toàn, chống chia 0
        if max_val < 1e-6:
            return np.zeros_like(vec)

        curr_max_idx = int(np.argmax(abs_v))
        out = np.copy(vec)
        low_t = self.cone_deadzone_low
        high_t = self.cone_deadzone_high

        for i in range(len(vec)):
            if i == curr_max_idx:
                continue
            ratio = abs_v[i] / (max_val + 1e-9)
            if ratio <= low_t:
                out[i] = 0.0
            elif ratio < high_t:
                t = (ratio - low_t) / (high_t - low_t)
                scale = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep C1
                out[i] = vec[i] * scale
            else:
                out[i] = vec[i]

        return out

    def _apply_soft_rotation_snapping(self, rotvec_tcp, rot_deadband=None):
        """
        Khóa trục xoay thông minh (Soft Rotation Snapping) & Bộ lọc triệt tiêu rò rỉ chéo đa tầng:
        - Angular Deadband (mặc định 0.026 rad ~ 1.5 độ): Triệt tiêu hoàn toàn vi rung sinh học khi bóp giữ phím Grip.
        - Khóa trục độc lập (< 30%): Khi người dùng chủ ý xoay 1 trục (Pitch, Yaw hoặc Roll),
          các trục phụ nhỏ hơn 30% trục chính (tỷ số < 0.30) bị triệt tiêu hoàn toàn về 0.0.
        - Chống đảo trục chập chờn (Hysteresis Latch 0.80): Giữ ổn định trục chính đã chọn nếu trục mới chưa vượt trội > 20%.
        - Hòa trộn 3D tự nhiên (30% - 60%): Chuyển tiếp mượt mà bằng đường cong Hermite Smoothstep;
          khi >= 60%, mở khóa hoàn toàn cho phép xoay 3D đa trục tự do.
        - Bảo toàn 100% biên độ của trục chủ đạo.
        """
        if rot_deadband is None:
            rot_deadband = self.rot_deadband

        mag = np.linalg.norm(rotvec_tcp)
        if mag < rot_deadband:
            self._last_dominant_rot_axis = None
            return np.zeros(3, dtype=float)

        abs_v = np.abs(rotvec_tcp)
        curr_max_idx = int(np.argmax(abs_v))
        curr_max_val = abs_v[curr_max_idx]

        if curr_max_val < 1e-5:
            self._last_dominant_rot_axis = None
            return np.zeros(3, dtype=float)

        # Hysteresis Latch: Giữ ổn định trục chính, tránh hiện tượng rung giật lật trục khi 2 trục gần bằng nhau
        dominant_idx = curr_max_idx
        if self._last_dominant_rot_axis is not None and self._last_dominant_rot_axis != curr_max_idx:
            prev_val = abs_v[self._last_dominant_rot_axis]
            if prev_val > 0.80 * curr_max_val:
                dominant_idx = self._last_dominant_rot_axis
                curr_max_val = prev_val
        self._last_dominant_rot_axis = dominant_idx

        snapped = np.zeros(3, dtype=float)
        snapped[dominant_idx] = rotvec_tcp[dominant_idx]
        for i in range(3):
            if i == dominant_idx:
                continue
            ratio = abs_v[i] / curr_max_val
            if ratio < self.rot_snap_low:
                snapped[i] = 0.0
            elif ratio < self.rot_snap_high:
                t = (ratio - self.rot_snap_low) / max(self.rot_snap_high - self.rot_snap_low, 1e-6)
                scale = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep
                snapped[i] = rotvec_tcp[i] * scale
            else:
                snapped[i] = rotvec_tcp[i]

        return snapped

    def calibrate_desk(self, frame=None):
        """Hiệu chuẩn hướng bàn máy khi người dùng nhấn tổ hợp Grip + B (lấy yaw của head_quat)."""
        if frame is None:
            frame = self.server.get_frame()
        if not frame:
            return
        head_quat = frame.get("head_quat", None)
        if head_quat is not None:
            r_head = R.from_quat([head_quat[1], head_quat[2], head_quat[3], head_quat[0]])
            v_fwd = r_head.apply([0, 0, -1])  # Hướng nhìn trong không gian phòng WebXR
            fx, fz = float(v_fwd[0]), float(v_fwd[2])
            norm_h = float(np.hypot(fx, fz))
            if norm_h > 0.05:
                self._desk_u_fwd = np.array([fx / norm_h, 0.0, fz / norm_h])
                self._desk_u_right = np.array([-fz / norm_h, 0.0, fx / norm_h])
            else:
                self._desk_u_fwd = np.array([0.0, 0.0, -1.0])
                self._desk_u_right = np.array([1.0, 0.0, 0.0])
            desk_yaw = math.degrees(math.atan2(self._desk_u_fwd[0], -self._desk_u_fwd[2]))
            self._R_home = None
            print(f"[HIỆU CHUẨN] Bàn máy đã căn chỉnh theo kính VR (Grip + B): yaw={desk_yaw:+.1f}°")
        else:
            self._desk_u_fwd = np.array([0.0, 0.0, -1.0])
            self._desk_u_right = np.array([1.0, 0.0, 0.0])
            self._R_home = None
            print("[HIỆU CHUẨN] Bàn máy mặc định: yaw=+0.0°")

    def get_pose(self):
        # Tự động đồng bộ và xả cờ kẹt nếu có phiên WebXR mới hoặc kết nối lại
        if hasattr(self.server, "check_and_clear_new_session"):
            val = self.server.check_and_clear_new_session()
            if isinstance(val, bool) and val is True:
                self.reset_anchor()

        frame = self.server.get_frame()
        if not frame:
            return None

        # Kiểm tra mất tracking tạm thời hoặc không có pose (tay cầm khuất tầm nhìn camera kính)
        if frame.get("tracking_lost") or frame.get("pos") is None:
            self._last_button_state = 0
            self.clutch_engaged = False
            self.grip_engaged = False
            self.is_pure_rotation = False
            self.is_translating = False
            self._locked_axis_idx = None
            self.R_ctrl_origin = None
            self.R_tcp_origin = None
            self._anchor_pos_quest = None
            self._anchor_quat_quest_xyzw = None
            return None

        # 1. Hiệu chuẩn hướng bàn máy khi người dùng bấm tổ hợp phím Grip + Button B (bắt cạnh lên):
        # Nút B đơn lẻ dành riêng cho Bật/Tắt Teleop; Tổ hợp Grip + B dùng để hiệu chuẩn bàn máy.
        btn_b = bool(frame.get("button_b", False) or frame.get("buttons", {}).get("button_b", False))
        is_rotating = frame.get("grip", False)
        calib_combo = is_rotating and btn_b
        if calib_combo and not self._last_calib_combo:
            self.calibrate_desk(frame)
        self._last_calib_combo = calib_combo
        self._last_btn_b = btn_b


        # Tách biệt điều khiển Tịnh tiến (Trigger) và Xoay (Grip)
        is_translating = frame.get("trigger", False)
        is_rotating = frame.get("grip", False)
        
        current_button_state = 0
        if is_translating: current_button_state += 1
        if is_rotating: current_button_state += 2
        
        if current_button_state == 0:
            self._last_button_state = 0
            self.clutch_engaged = False
            self.grip_engaged = False
            self.is_pure_rotation = False
            self.is_translating = False
            self._locked_axis_idx = None
            self.R_ctrl_origin = None
            self.R_tcp_origin = None
            return None

        # TÍNH NĂNG CLUTCH THÔNG MINH:
        # Bất cứ khi nào thay đổi nút bấm (Vd: Đang Tịnh tiến -> Chuyển sang Xoay hoặc nhả ra bóp lại),
        # Hệ thống sẽ TỰ ĐỘNG CHỐT MỎ NEO MỚI (Reset Anchor). 
        # Loại bỏ hoàn toàn lỗi "nhảy cóc (teleport) tọa độ" do bị ép biến delta về 0.
        if current_button_state != self._last_button_state:
            self._anchor_pos_quest = np.array(frame["pos"])
            self._locked_axis_idx = None
            self.clutch_engaged = bool(is_translating)
            self.grip_engaged = bool(is_rotating)
            self.is_translating = bool(is_translating)
            self.is_pure_rotation = bool(is_rotating and not is_translating)
            
            q_q = frame.get("quat", [1.0, 0.0, 0.0, 0.0]) # [w, x, y, z]
            self._anchor_quat_quest_xyzw = np.array([q_q[1], q_q[2], q_q[3], q_q[0]], dtype=float)
            norm_qq = np.linalg.norm(self._anchor_quat_quest_xyzw)
            if norm_qq > 1e-6 and not np.isnan(norm_qq):
                self._anchor_quat_quest_xyzw /= norm_qq
            else:
                self._anchor_quat_quest_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
            
            # Neo vào vị trí thực tế của robot (hoặc Widget 3D trên màn hình)
            try:
                if self.anchor_reference_func is not None:
                    # Lấy tọa độ từ Widget 3D hoặc Robot FK (x, y, z, qw, qx, qy, qz)
                    ref_pose = self.anchor_reference_func()
                    self._anchor_tcp = np.array(ref_pose)
                else:
                    self._anchor_tcp = np.array(self.ctrl.tcp())
            except Exception:
                self._last_button_state = 0
                return None
            
            # Tính toán quaternion neo của robot
            if len(self._anchor_tcp) >= 7:
                qwxyz = self._anchor_tcp[3:7]
                self._anchor_quat_robot_xyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=float)
            else:
                axang = self._anchor_tcp[3:6]
                angle = np.linalg.norm(axang)
                if angle > 1e-6:
                    axis = axang / angle
                    qwxyz = transforms3d.quaternions.axangle2quat(axis, angle)
                    self._anchor_quat_robot_xyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=float)
                else:
                    self._anchor_quat_robot_xyzw = np.array([0., 0., 0., 1.], dtype=float)

            norm_q_r = np.linalg.norm(self._anchor_quat_robot_xyzw)
            if norm_q_r > 1e-6 and not np.isnan(norm_q_r):
                self._anchor_quat_robot_xyzw /= norm_q_r
            else:
                self._anchor_quat_robot_xyzw = np.array([0., 0., 0., 1.], dtype=float)
            
            # Lấy góc quay của mặt phẳng cánh tay cơ học dựa trên tọa độ XY của TCP trên mặt đồng hồ
            try:
                p_xy = self._anchor_tcp[:2]
                r_tcp = float(np.hypot(p_xy[0], p_xy[1]))
                if r_tcp > 0.03:
                    self._anchor_theta_arm = float(np.arctan2(p_xy[1], p_xy[0]))
                else:
                    joints = self.ctrl.joints() if self.ctrl is not None and hasattr(self.ctrl, "joints") else None
                    if joints is not None and len(joints) > 0:
                        self._anchor_theta_arm = float(joints[0])
                    else:
                        self._anchor_theta_arm = 0.0
            except Exception:
                self._anchor_theta_arm = 0.0
            
            # Xác lập hệ quy chiếu điều khiển (Heading Reference):
            head_quat = frame.get("head_quat", None)
            if head_quat is not None:
                r_head = R.from_quat([head_quat[1], head_quat[2], head_quat[3], head_quat[0]])
                v_fwd = r_head.apply([0, 0, -1])  # Hướng nhìn trong không gian phòng WebXR
                fx, fz = float(v_fwd[0]), float(v_fwd[2])
                norm_h = float(np.hypot(fx, fz))
                if norm_h > 0.05:
                    self._anchor_u_fwd = np.array([fx / norm_h, 0.0, fz / norm_h])
                    self._anchor_u_right = np.array([-fz / norm_h, 0.0, fx / norm_h])
                else:
                    self._anchor_u_fwd = np.copy(self._desk_u_fwd)
                    self._anchor_u_right = np.copy(self._desk_u_right)
            else:
                self._anchor_u_fwd = np.copy(self._desk_u_fwd)
                self._anchor_u_right = np.copy(self._desk_u_right)

            self._ema_pos = np.copy(self._anchor_tcp[:3])
            self._ema_quat_xyzw = np.copy(self._anchor_quat_robot_xyzw)
            self._last_dominant_rot_axis = None
            self._last_dominant_trans_axis = None
            self._last_button_state = current_button_state
            self._last_gesture_angle = None
            self._last_locked_axis_idx = None
            self._last_hand_quat_xyzw = np.copy(self._anchor_quat_quest_xyzw)
            self._prev_raw_target_quat_xyzw = None
            self.locked_rotation_axis = None
            self._axis_theta_max_pos.clear()
            self._axis_theta_min_neg.clear()
            self._gesture_angle_offset = 0.0

            # Chốt cố định hệ 3 trục quay bám động theo tư thế đầu máy R_tcp_origin (Intrinsic Tool Frame)
            R_tcp_origin_snap = R.from_quat(self._anchor_quat_robot_xyzw)
            if self._R_home is None:
                self._R_home = R_tcp_origin_snap
                p_anc_xy = self._anchor_tcp[:2] if self._anchor_tcp is not None else np.zeros(2)
                r_anc_xy = float(np.hypot(p_anc_xy[0], p_anc_xy[1]))
                if r_anc_xy > 0.03:
                    theta_arm_home = float(np.arctan2(p_anc_xy[1], p_anc_xy[0]))
                elif self._anchor_theta_arm is not None:
                    theta_arm_home = float(self._anchor_theta_arm)
                else:
                    theta_arm_home = 0.0
                a_roll_home = np.array([np.cos(theta_arm_home), np.sin(theta_arm_home), 0.0], dtype=np.float64)
                a_pitch_home = np.array([-np.sin(theta_arm_home), np.cos(theta_arm_home), 0.0], dtype=np.float64)
                a_yaw_home = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                self._e_yaw_local = self._R_home.inv().apply(a_yaw_home)
                self._e_pitch_local = self._R_home.inv().apply(a_pitch_home)
                self._e_roll_local = self._R_home.inv().apply(a_roll_home)
                norm_ey = np.linalg.norm(self._e_yaw_local)
                if norm_ey > 1e-6:
                    self._e_yaw_local /= norm_ey
                norm_ep = np.linalg.norm(self._e_pitch_local)
                if norm_ep > 1e-6:
                    self._e_pitch_local /= norm_ep
                norm_er = np.linalg.norm(self._e_roll_local)
                if norm_er > 1e-6:
                    self._e_roll_local /= norm_er

            u_yaw = R_tcp_origin_snap.apply(self._e_yaw_local)
            u_pitch = R_tcp_origin_snap.apply(self._e_pitch_local)
            u_roll = R_tcp_origin_snap.apply(self._e_roll_local)
            self._clock_u_up = u_yaw / max(float(np.linalg.norm(u_yaw)), 1e-6)
            self._clock_u_tangential = u_pitch / max(float(np.linalg.norm(u_pitch)), 1e-6)
            self._clock_u_radial = u_roll / max(float(np.linalg.norm(u_roll)), 1e-6)
            self._clock_u_tool = np.copy(self._clock_u_radial)

            # Khởi tạo ma trận gốc cho cơ chế Grip
            try:
                self.R_ctrl_origin = R.from_quat(self._anchor_quat_quest_xyzw)
                self.R_tcp_origin = R_tcp_origin_snap
            except Exception:
                self.R_ctrl_origin = None
                self.R_tcp_origin = None

            target_quat_wxyz = [
                float(self._anchor_quat_robot_xyzw[3]),
                float(self._anchor_quat_robot_xyzw[0]),
                float(self._anchor_quat_robot_xyzw[1]),
                float(self._anchor_quat_robot_xyzw[2])
            ]
            if hasattr(self, "_last_output_quat_wxyz") and self._last_output_quat_wxyz is not None:
                if np.dot(target_quat_wxyz, self._last_output_quat_wxyz) < 0.0:
                    target_quat_wxyz = [-w for w in target_quat_wxyz]
            self.prev_target_quat = target_quat_wxyz
            self._last_output_quat_wxyz = list(target_quat_wxyz)

            # Frame đầu tiên khi chỉ bóp Grip xoay: chốt mỏ neo xoay và trả về None
            self.grip_engaged = is_rotating
            self.is_translating = bool(is_translating)
            self.is_pure_rotation = bool(is_rotating and not is_translating)
            if is_rotating and not is_translating:
                return None

            return {
                "pos": self._ema_pos.tolist(),
                "quat": target_quat_wxyz
            }

        # Kiểm tra an toàn trước khi tính toán delta (chống ValueError / TypeError / NaN)
        if self._anchor_quat_quest_xyzw is None or self._anchor_quat_robot_xyzw is None:
            return None
        if self._anchor_pos_quest is None or self._anchor_tcp is None:
            return None
        if np.any(np.isnan(self._anchor_quat_quest_xyzw)) or np.any(np.isnan(self._anchor_quat_robot_xyzw)):
            return None

        # ===============================
        # TÍNH TOÁN DELTA TỊNH TIẾN (HEAD-CENTRIC & WORLD)
        # ===============================
        diag_msg = "STANDING STILL"
        delta_mag = 0.0
        
        R_q_anchor = R.from_quat(self._anchor_quat_quest_xyzw)
        R_r_anchor = R.from_quat(self._anchor_quat_robot_xyzw)
        
        if is_translating:
            current_pos_quest = np.array(frame["pos"])
            # delta_q_raw là sự thay đổi vị trí tay người dùng trong KHÔNG GIAN THỰC
            # so với lúc vừa bấm cò
            delta_q_raw = current_pos_quest - self._anchor_pos_quest
            delta_mag = np.linalg.norm(delta_q_raw)
            
            # Chiếu cử động thô theo hệ quy chiếu hướng nhìn / bàn máy:
            if self._anchor_u_fwd is not None and self._anchor_u_right is not None:
                d_fwd_unfiltered = float(np.dot(delta_q_raw, self._anchor_u_fwd))
                d_lat_unfiltered = float(np.dot(delta_q_raw, self._anchor_u_right))
                d_vert_unfiltered = float(delta_q_raw[1])
            else:
                d_fwd_unfiltered = -float(delta_q_raw[2])
                d_lat_unfiltered =  float(delta_q_raw[0])
                d_vert_unfiltered = float(delta_q_raw[1])

            # Áp dụng Soft Deadband riêng từng trục:
            # 1. Trục Tiến/Lùi: deadband 1.5mm (nhạy bén ngay khi vừa nhích nhẹ)
            abs_fwd = abs(d_fwd_unfiltered)
            if abs_fwd < self.fwd_deadband:
                d_forward_raw = 0.0
            else:
                d_forward_raw = math.copysign(abs_fwd - self.fwd_deadband, d_fwd_unfiltered)

            # 2. Trục Phải/Trái: deadband 3.5mm (triệt tiêu rung lắc ngang do bóp ngón trỏ)
            abs_lat = abs(d_lat_unfiltered)
            if abs_lat < self.lat_deadband:
                d_lateral_raw = 0.0
            else:
                d_lateral_raw = math.copysign(abs_lat - self.lat_deadband, d_lat_unfiltered)

            # 3. Trục Cao/Thấp: deadband 3.5mm (triệt tiêu vi rung nâng hạ tay)
            abs_vert = abs(d_vert_unfiltered)
            if abs_vert < self.vert_deadband:
                d_vertical_raw = 0.0
            else:
                d_vertical_raw = math.copysign(abs_vert - self.vert_deadband, d_vert_unfiltered)

            delta_q_world = delta_q_raw

            # Áp dụng Bộ lọc giảm nhiễu chéo hai chiều (Bi-directional Cross-Axis Damping):
            if self.enable_cross_axis_damping:
                d_damped = self._apply_bidirectional_cross_axis_damping(np.array([d_forward_raw, d_lateral_raw, d_vertical_raw]))
                d_forward = d_damped[0]
                d_lateral = d_damped[1]
                d_vertical = d_damped[2]
            elif self.enable_axis_snapping:
                d_snapped = self._apply_aggressive_axis_isolation(np.array([d_forward_raw, d_lateral_raw, d_vertical_raw]))
                d_forward = d_snapped[0]
                d_lateral = d_snapped[1]
                d_vertical = d_snapped[2]
            else:
                d_forward = d_forward_raw
                d_lateral = d_lateral_raw
                d_vertical = d_vertical_raw

            # Áp dụng hệ số độ nhạy SAU bộ lọc deadband/damping để bảo toàn 100% ngưỡng lọc và hành vi chống rung
            d_forward  *= self.sensitivity_scale
            d_lateral  *= self.sensitivity_scale
            d_vertical *= self.sensitivity_scale

            self.last_delta_trans = np.array([d_forward, d_lateral, d_vertical], dtype=float)
            self.last_delta_mag = float(delta_mag)

            cm_fwd = d_forward * 100.0
            cm_lat = d_lateral * 100.0
            cm_vert = d_vertical * 100.0
            fwd_str = f"Tiến {cm_fwd:+.1f}cm" if cm_fwd >= 0 else f"Lùi {cm_fwd:+.1f}cm"
            lat_str = f"Phải {cm_lat:+.1f}cm" if cm_lat >= 0 else f"Trái {cm_lat:+.1f}cm"
            vert_str = f"Cao {cm_vert:+.1f}cm" if cm_vert >= 0 else f"Thấp {cm_vert:+.1f}cm"

            # Tự động sắp xếp: Trục nào có độ lớn dịch chuyển lớn nhất sẽ nằm ĐẦU TIÊN
            movements = [
                (abs(cm_fwd), fwd_str),
                (abs(cm_lat), lat_str),
                (abs(cm_vert), vert_str)
            ]
            movements.sort(key=lambda m: m[0], reverse=True)
            if delta_mag >= self.pos_deadband:
                diag_msg = f"{movements[0][1]} | {movements[1][1]} | {movements[2][1]}"

            if self.nav_mode in ("cylindrical", "clock_face") and self._anchor_tcp is not None:
                # ===============================================================
                # CHẾ ĐỘ MẶT ĐỒNG HỒ & THU/DUỖI TAY MÁY CƠ HỌC (TRUE MECHANICAL CLOCK)
                # ===============================================================
                # Tâm đồng hồ: Trục quay khớp đế Base Z (0, 0)
                # Kim đồng hồ: Góc vật lý của khớp đế q0
                # Vươn / Thu kim: Duỗi hoặc gập bắp tay theo phương vươn u_reach của q0
                p_anc = np.array(self._anchor_tcp[:3], dtype=float)
                r_reach = float(np.hypot(p_anc[0], p_anc[1]))

                q0_anc = float(self._anchor_theta_arm) if self._anchor_theta_arm is not None else 0.0

                # 1. Sang Phải / Sang Trái -> Chỉ xoay kim đồng hồ (khớp đế q0) quanh Base Z:
                # - Sang Phải (d_lateral > 0) -> q0 giảm (quay theo chiều kim đồng hồ)
                # - Sang Trái (d_lateral < 0) -> q0 tăng (quay ngược chiều kim đồng hồ)
                delta_arc = d_lateral
                delta_arc = np.clip(delta_arc, -self.max_delta_m, self.max_delta_m)
                delta_q0 = -delta_arc / max(r_reach, 0.18)
                delta_q0 = np.clip(delta_q0, -0.65, 0.65)  # Khống chế góc quay tối đa ~37 độ / lần bóp cò

                q0_target = q0_anc + delta_q0

                # Quay vị trí mỏ neo quanh trục Base Z theo góc delta_q0:
                cos_q = np.cos(delta_q0)
                sin_q = np.sin(delta_q0)
                p_rot = np.array([
                    cos_q * p_anc[0] - sin_q * p_anc[1],
                    sin_q * p_anc[0] + cos_q * p_anc[1],
                    p_anc[2]
                ])

                # 2. Tiến / Lùi -> Duỗi dài hoặc Thu ngắn kim đồng hồ theo phương vươn cơ học:
                # Vector vươn hướng tâm u_reach từ tâm Base Z ra điểm mỏ neo đã xoay p_rot:
                r_rot_xy = float(np.hypot(p_rot[0], p_rot[1]))
                if r_rot_xy > 1e-6:
                    u_reach = np.array([p_rot[0] / r_rot_xy, p_rot[1] / r_rot_xy, 0.0])
                else:
                    u_reach = np.array([np.cos(q0_target), np.sin(q0_target), 0.0])

                # Khuếch đại phi tuyến dải nhỏ: < 2cm dùng scale_reach = 1.35 giúp đẩy nhẹ tay
                # thì cánh tay duỗi rõ rệt; khi đẩy dài hơn (> 2cm), chuyển tiếp mượt về 1.05
                base_scale_reach = float(os.environ.get("REACH_SCALE", "1.05"))
                abs_fwd_m = abs(d_forward)
                if abs_fwd_m < 0.02:
                    t_boost = abs_fwd_m / 0.02
                    scale_reach = 1.35 - (1.35 - base_scale_reach) * t_boost
                else:
                    scale_reach = base_scale_reach

                delta_reach = d_forward * scale_reach
                delta_reach = np.clip(delta_reach, -self.max_delta_m, self.max_delta_m)

                # 3. Lên / Xuống -> Nâng hạ chiều cao Z:
                delta_z = d_vertical
                delta_z = np.clip(delta_z, -self.max_delta_m, self.max_delta_m)
                z_target = np.clip(p_anc[2] + delta_z, 0.05, 0.55)

                # Tọa độ đích kết hợp:
                target_pos = p_rot + delta_reach * u_reach
                target_pos[2] = z_target

                # Giới hạn an toàn bán kính cơ học:
                # Dải [0.18m, 0.485m] loại bỏ hoàn toàn tự va chạm thân đế (cột trụ ảo R_min = 0.18m)
                r_target_xy = np.hypot(target_pos[0], target_pos[1])
                if r_target_xy < 0.18:
                    scale_r = 0.18 / max(r_target_xy, 1e-6)
                    target_pos[0] *= scale_r
                    target_pos[1] *= scale_r
                elif r_target_xy > 0.485:
                    scale_r = 0.485 / r_target_xy
                    target_pos[0] *= scale_r
                    target_pos[1] *= scale_r

                self._current_delta_theta = delta_q0
            elif self.nav_mode == "arm_plane" and self._anchor_tcp is not None:
                theta_arm = self._anchor_theta_arm
                if theta_arm is None:
                    x_anc = self._anchor_tcp[0]
                    y_anc = self._anchor_tcp[1]
                    r_anc = np.hypot(x_anc, y_anc)
                    theta_arm = np.arctan2(y_anc, x_anc) if r_anc > 0.03 else 0.0

                dx_robot = d_forward * np.cos(theta_arm) + d_lateral * np.sin(theta_arm)
                dy_robot = d_forward * np.sin(theta_arm) - d_lateral * np.cos(theta_arm)
                dz_robot = d_vertical
                delta_robot_world = np.array([dx_robot, dy_robot, dz_robot])
                delta_robot_world = np.clip(delta_robot_world, -self.max_delta_m, self.max_delta_m)
                target_pos = self._anchor_tcp[:3] + delta_robot_world
                self._current_delta_theta = 0.0
            else:
                delta_robot_world = np.array([
                    d_forward,
                    -d_lateral,
                    d_vertical
                ])
                delta_robot_world = np.clip(delta_robot_world, -self.max_delta_m, self.max_delta_m)
                target_pos = self._anchor_tcp[:3] + delta_robot_world
                self._current_delta_theta = 0.0
        else:
            delta_robot_world = np.zeros(3)
            diag_msg = "CHỈ XOAY (KHÓA TỊNH TIẾN)"
            self.last_delta_trans = np.zeros(3, dtype=float)
            self.last_delta_mag = 0.0
            target_pos = np.copy(self._anchor_tcp[:3])
            self._current_delta_theta = 0.0
            self._last_dominant_trans_axis = None

        # Workspace safety clamp (Cách mặt bàn ít nhất 3.5cm, cách thân đế 14cm, bảo vệ tầm với UR3e)
        target_pos[0] = np.clip(target_pos[0], -0.52, 0.52) # X
        target_pos[1] = np.clip(target_pos[1], -0.52, 0.52) # Y
        target_pos[2] = np.clip(target_pos[2], 0.035, 0.60) # Z (cách mặt bàn ít nhất 3.5cm an toàn)

        # Slew Rate Limiter: Tối ưu trần max_step 0.091 (+30% so với 0.070, ~1.82 m/s) vừa nhanh vừa mượt mà
        if self._ema_pos is not None:
            pos_step = target_pos - self._ema_pos
            step_mag = np.linalg.norm(pos_step)
            max_step = float(os.environ.get("MAX_POS_STEP", "0.091"))  # tối đa 9.1cm / 50ms = 1.82 m/s (+30% từ 0.070)
            if step_mag > max_step:
                target_pos = self._ema_pos + pos_step * (max_step / step_mag)

        # Dynamic Adaptive EMA Filter cho Position (Làm mượt C1/C2 liên tục, triệt tiêu rung giật):
        # - Khi đứng yên hoặc vi rung nhẹ (< 1cm): alpha = 0.28 (triệt tiêu vi rung sinh học)
        # - Khi di chuyển dứt khoát (> 3cm): alpha = 0.66 (+20% từ 0.55, phản hồi tức thì, bám sát cử động tay)
        if self._ema_pos is None:
            self._ema_pos = target_pos
        else:
            pos_diff = np.linalg.norm(target_pos - self._ema_pos)
            t_pos = np.clip(pos_diff / 0.03, 0.0, 1.0)
            smooth_pos = t_pos * t_pos * (3.0 - 2.0 * t_pos)
            alpha_pos = 0.28 + (0.66 - 0.28) * smooth_pos
            self._ema_pos = alpha_pos * target_pos + (1.0 - alpha_pos) * self._ema_pos

        current_time = time.time()
        if diag_msg != self._last_diag_msg or (current_time - self._last_diag_time > 0.6):
            if diag_msg != "STANDING STILL":
                tp = self._ema_pos
                r_now = float(np.hypot(tp[0], tp[1]))
                try:
                    q0_cur_deg = np.degrees(self.ctrl.joints()[0])
                except Exception:
                    q0_cur_deg = 0.0
                raw_str = ""
                if is_translating:
                    raw_str = f"RAW: [dx={delta_q_world[0]*100:+.1f}, dy={delta_q_world[1]*100:+.1f}, dz={delta_q_world[2]*100:+.1f}]cm | "
                print(f"[Đồng Hồ VR] {diag_msg} | {raw_str}Base q0: {q0_cur_deg:+.1f}° (Δ{np.degrees(self._current_delta_theta):+.1f}°) | Vươn R: {r_now:.3f}m | Đích: [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]")
            self._last_diag_msg = diag_msg
            self._last_diag_time = current_time

        # ===============================
        # TÍNH TOÁN DELTA XOAY (CLOCK-FACE & STRICT SINGLE-AXIS DOMINANT LOCK)
        # ===============================
        if is_rotating:
            self.grip_engaged = True
            self.is_translating = bool(is_translating)
            self.is_pure_rotation = bool(not is_translating)
            curr_q = frame["quat"]
            curr_quat_quest_xyzw = np.array([curr_q[1], curr_q[2], curr_q[3], curr_q[0]], dtype=np.float64)
            norm_curr = np.linalg.norm(curr_quat_quest_xyzw)
            if norm_curr > 1e-6:
                curr_quat_quest_xyzw /= norm_curr

            # 1. Chống lật dấu Quaternion đầu vào (Quest Controller Continuity)
            if self._last_hand_quat_xyzw is not None:
                if np.dot(curr_quat_quest_xyzw, self._last_hand_quat_xyzw) < 0.0:
                    curr_quat_quest_xyzw = -curr_quat_quest_xyzw
            self._last_hand_quat_xyzw = np.copy(curr_quat_quest_xyzw)

            R_hand_current = R.from_quat(curr_quat_quest_xyzw)

            # 1. Trích xuất độ lệch xoay trong Hệ quy chiếu Nội tại của Tay cầm (Controller Local Frame):
            # Delta R_ctrl_local = R_ctrl_origin^{-1} * R_ctrl_now
            # Phép nhân bên phải (Post-multiply với nghịch đảo của anchor) bảo toàn 100% trục giải phẫu cổ tay,
            # hoàn toàn độc lập với góc nghiêng tự nhiên của bàn tay hay hướng đứng của người dùng.
            R_delta_local = R_q_anchor.inv() * R_hand_current
            omega_local = R_delta_local.as_rotvec()

            # 2. Tách Đúng 3 Trục Nội tại của Tay cầm Quest (Delta R_hand) & Single-Axis Lock:
            # - Trục 1 — yaw_input (Gập cổ tay lắc trái / phải): omega_local[2]
            # - Trục 2 — pitch_input (Uống nước ngửa lên / Đặt cốc chúc xuống): -omega_local[0]
            # - Trục 3 — roll_input (Vặn chìa khóa Clock-face): -omega_local[1]
            yaw_input   =  float(omega_local[2])
            pitch_input = -float(omega_local[0])
            roll_input  = -float(omega_local[1])

            omega_gesture = np.array([
                roll_input,   # Kênh 0: Roll (Vặn chìa khóa Clock-face quanh A_roll)
                pitch_input,  # Kênh 1: Pitch (Uống nước ngửa lên / Đặt cốc chúc xuống quanh A_pitch)
                yaw_input     # Kênh 2: Yaw (Gập cổ tay lắc trái / phải quanh A_yaw)
            ], dtype=np.float64)

            # 3. Khóa Độc Quyền 1 Trục (Strict Single-Axis Dominant Lock):
            deadzone = self.axis_lock_deadzone_rad
            abs_w = np.abs(omega_gesture)
            max_abs = float(np.max(abs_w))

            if self._locked_axis_idx is None:
                if max_abs < deadzone:
                    omega_filtered = np.zeros(3, dtype=np.float64)
                else:
                    self._locked_axis_idx = int(np.argmax(abs_w))
                    self._last_locked_axis_idx = self._locked_axis_idx
                    self._last_gesture_angle = float(omega_gesture[self._locked_axis_idx])
                    self._gesture_angle_offset = 0.0

            if self._locked_axis_idx is not None:
                omega_filtered = np.zeros(3, dtype=np.float64)
                active_angle = float(omega_gesture[self._locked_axis_idx])

                # 3. Chống lật dấu góc +-pi (Unwrap Incremental Angle) & Slew-Rate Clamping:
                # Triệt tiêu hoàn toàn hiện tượng tự đảo chiều quay ngược lại khi góc quay tay cầm vượt +-pi (180°)
                if self._last_gesture_angle is not None and self._last_locked_axis_idx == self._locked_axis_idx:
                    d_raw = active_angle - self._last_gesture_angle
                    if d_raw < -math.pi:
                        d_wrap = d_raw + 2.0 * math.pi
                    elif d_raw > math.pi:
                        d_wrap = d_raw - 2.0 * math.pi
                    else:
                        d_wrap = d_raw
                    # Kẹp gia số frame-to-frame chống spike (tối đa 0.35 rad ~ 20°/frame tại 20Hz)
                    d_wrap = float(np.clip(d_wrap, -0.35, 0.35))
                    active_angle = self._last_gesture_angle + d_wrap
                self._last_gesture_angle = active_angle
                self._last_locked_axis_idx = self._locked_axis_idx

                # Trừ đi gesture_angle_offset (Zero Windup) và deadzone để chuyển động bắt đầu mượt từ 0
                effective_hand_angle = active_angle - self._gesture_angle_offset
                signed_angle = float(np.sign(effective_hand_angle) * max(0.0, abs(effective_hand_angle) - deadzone))

                # 4. Giảm tốc độ quay (Angular Scaling: GRIP_ROTATION_GAIN = 0.55) & Kẹp biên độ góc quay tuyệt đối:
                theta_applied = self.grip_rotation_gain * signed_angle

                # Kẹp biên độ góc quay tuyệt đối so với mốc gốc ban đầu (_tcp_origin_rot):
                # Khôi phục biên độ góc quay tự nhiên, mượt mà chuẩn stable-22-10-chieu-demo-robot
                if self._locked_axis_idx == 2:  # Yaw (Z)
                    max_limit = getattr(self, "max_z_rotation_delta", float(os.environ.get("MAX_YAW_DELTA", os.environ.get("MAX_Z_ROTATION_DELTA", str(math.radians(170.0))))))
                elif self._locked_axis_idx == 0:  # Roll
                    max_limit = getattr(self, "max_roll_rotation_delta", float(os.environ.get("MAX_ROLL_DELTA", str(math.radians(170.0)))))
                elif self._locked_axis_idx == 1:  # Pitch
                    max_limit = getattr(self, "max_pitch_rotation_delta", float(os.environ.get("MAX_PITCH_DELTA", str(math.radians(170.0)))))
                else:
                    max_limit = float(os.environ.get("MAX_ROT_ENVELOPE_RAD", str(math.radians(170.0))))

                # Áp dụng giới hạn động từ FK Axis-Purity Guard (Zero Windup Dynamic Clamping)
                if self._locked_axis_idx in self._axis_theta_max_pos:
                    dyn_max = self._axis_theta_max_pos[self._locked_axis_idx]
                    if dyn_max < max_limit:
                        max_limit = dyn_max
                if self._locked_axis_idx in self._axis_theta_min_neg:
                    dyn_min = self._axis_theta_min_neg[self._locked_axis_idx]
                    if dyn_min > -max_limit:
                        min_limit = dyn_min
                    else:
                        min_limit = -max_limit
                else:
                    min_limit = -max_limit

                # Xử lý Zero Windup: Nếu tay người dùng tiếp tục vặn vượt quá giới hạn khả dụng,
                # hấp thu góc vượt quá vào _gesture_angle_offset để khi đảo chiều thì góc quay giảm ngay lập tức!
                if theta_applied > max_limit:
                    excess_theta = theta_applied - max_limit
                    self._gesture_angle_offset += excess_theta / max(self.grip_rotation_gain, 1e-4)
                    theta_applied = max_limit
                elif theta_applied < min_limit:
                    excess_theta = theta_applied - min_limit
                    self._gesture_angle_offset += excess_theta / max(self.grip_rotation_gain, 1e-4)
                    theta_applied = min_limit

                omega_filtered[self._locked_axis_idx] = theta_applied / self.grip_rotation_gain if self.grip_rotation_gain > 0 else theta_applied

                omega_scaled = np.zeros(3, dtype=np.float64)
                omega_scaled[self._locked_axis_idx] = theta_applied
            else:
                omega_filtered = np.zeros(3, dtype=np.float64)
                omega_scaled = np.zeros(3, dtype=np.float64)

            # 5. Xây dựng Hệ 3 Trục Đích Trực giao Tuyệt đối Neo Động theo Hướng Đầu Máy R_r_anchor (R_tcp_origin):
            if self._clock_u_radial is None or self._clock_u_tangential is None or self._clock_u_up is None:
                if self._R_home is None:
                    self._R_home = R_r_anchor
                    p_anc_xy = self._anchor_tcp[:2] if self._anchor_tcp is not None else np.zeros(2)
                    r_anc_xy = float(np.hypot(p_anc_xy[0], p_anc_xy[1]))
                    if r_anc_xy > 0.03:
                        theta_arm_home = float(np.arctan2(p_anc_xy[1], p_anc_xy[0]))
                    elif self._anchor_theta_arm is not None:
                        theta_arm_home = float(self._anchor_theta_arm)
                    else:
                        theta_arm_home = 0.0
                    a_roll_home = np.array([np.cos(theta_arm_home), np.sin(theta_arm_home), 0.0], dtype=np.float64)
                    a_pitch_home = np.array([-np.sin(theta_arm_home), np.cos(theta_arm_home), 0.0], dtype=np.float64)
                    a_yaw_home = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                    self._e_yaw_local = self._R_home.inv().apply(a_yaw_home)
                    self._e_pitch_local = self._R_home.inv().apply(a_pitch_home)
                    self._e_roll_local = self._R_home.inv().apply(a_roll_home)
                    norm_ey = np.linalg.norm(self._e_yaw_local)
                    if norm_ey > 1e-6:
                        self._e_yaw_local /= norm_ey
                    norm_ep = np.linalg.norm(self._e_pitch_local)
                    if norm_ep > 1e-6:
                        self._e_pitch_local /= norm_ep
                    norm_er = np.linalg.norm(self._e_roll_local)
                    if norm_er > 1e-6:
                        self._e_roll_local /= norm_er

                u_yaw = R_r_anchor.apply(self._e_yaw_local)
                u_pitch = R_r_anchor.apply(self._e_pitch_local)
                u_roll = R_r_anchor.apply(self._e_roll_local)
                self._clock_u_up = u_yaw / max(float(np.linalg.norm(u_yaw)), 1e-6)
                self._clock_u_tangential = u_pitch / max(float(np.linalg.norm(u_pitch)), 1e-6)
                self._clock_u_radial = u_roll / max(float(np.linalg.norm(u_roll)), 1e-6)
                self._clock_u_tool = np.copy(self._clock_u_radial)

            A_yaw = self._clock_u_up             # Trục 1: Gập cổ tay trái / phải (bám động theo mặt phẳng yaw của tool)
            A_roll = self._clock_u_radial        # Trục 2: Vặn chìa khóa (bám động theo hướng mũi roll của tool)
            A_pitch = self._clock_u_tangential    # Trục 3: Uống nước / Đặt cốc (bám động theo bản lề pitch của tool)

            if self._locked_axis_idx == 2:
                # Nhánh Trục 1 — Gập cổ tay trái / phải (yaw_input thắng => Yaw quanh A_yaw):
                # Công thức: R_target = Rot(A_yaw, theta_yaw) * R_tcp_origin (Pre-multiply)
                R_delta_target = R.from_rotvec(theta_applied * A_yaw)
                R_target = R_delta_target * R_r_anchor
                self.locked_rotation_axis = np.copy(A_yaw)

            elif self._locked_axis_idx == 1:
                # Nhánh Trục 2 — Uống nước / Đặt cốc (pitch_input thắng => Pitch quanh A_pitch):
                # Công thức: R_target = Rot(A_pitch, theta_pitch) * R_tcp_origin (Pre-multiply)
                R_delta_target = R.from_rotvec(theta_applied * A_pitch)
                R_target = R_delta_target * R_r_anchor
                self.locked_rotation_axis = np.copy(A_pitch)

            elif self._locked_axis_idx == 0:
                # Nhánh Trục 3 — Vặn chìa khóa (roll_input thắng => Roll Clock-face quanh A_roll):
                # Công thức: R_target = Rot(A_roll, theta_roll) * R_tcp_origin (Pre-multiply)
                R_delta_target = R.from_rotvec(theta_applied * A_roll)
                R_target = R_delta_target * R_r_anchor
                self.locked_rotation_axis = np.copy(A_roll)

            else:
                R_target = R_r_anchor
                self.locked_rotation_axis = None

            # 6. Tích hợp góc bù xoay Base (nếu đang tịnh tiến kết hợp trong cylindrical / clock_face)
            if self.nav_mode in ("cylindrical", "clock_face") and is_translating:
                R_base_rot = R.from_rotvec([0.0, 0.0, self._current_delta_theta])
                R_target_robot = R_base_rot * R_target
            else:
                R_target_robot = R_target

            axis_names = ["VẶN CHÌA KHÓA (Roll)", "UỐNG NƯỚC (Pitch)", "GẬP CỔ TAY (Yaw)"]
            if self._locked_axis_idx is not None and abs(omega_filtered[self._locked_axis_idx]) > 1e-4:
                deg_rot = np.degrees(abs(omega_scaled[self._locked_axis_idx]))
                diag_msg = f"XOAY {axis_names[self._locked_axis_idx]} ({deg_rot:.1f}° | Khóa 1 trục)"
            elif self._locked_axis_idx is not None:
                diag_msg = f"ĐÃ KHÓA {axis_names[self._locked_axis_idx]} (VÙNG CHẾT)"
            else:
                diag_msg = "CHỜ KHÓA TRỤC XOAY (<4°)"
        else:
            self.grip_engaged = False
            self.is_translating = bool(is_translating)
            self.is_pure_rotation = False
            self._locked_axis_idx = None
            self._last_locked_axis_idx = None
            self._last_gesture_angle = None
            self._last_dominant_rot_axis = None
            self.locked_rotation_axis = None
            self._clock_u_radial = None
            self._clock_u_tangential = None
            self._clock_u_up = None
            self._clock_u_tool = None
            if self.nav_mode in ("cylindrical", "clock_face") and is_translating:
                # Khi chỉ bóp Cò tịnh tiến trong chế độ Đồng hồ / Trụ:
                # Ngàm kẹp tự động xoay theo góc quay Base quanh trục Z để giữ nguyên góc đón vật
                R_base_rot = R.from_rotvec([0.0, 0.0, self._current_delta_theta])
                R_target_robot = R_base_rot * R_r_anchor
            else:
                R_target_robot = R_r_anchor
        target_quat_xyzw = R_target_robot.as_quat()
        norm_tq = np.linalg.norm(target_quat_xyzw)
        if norm_tq > 1e-9:
            target_quat_xyzw /= norm_tq

        # Kiểm tra tính liên tục dấu Quaternion mục tiêu thô với frame trước
        if self._prev_raw_target_quat_xyzw is not None:
            if np.dot(target_quat_xyzw, self._prev_raw_target_quat_xyzw) < 0.0:
                target_quat_xyzw = -target_quat_xyzw
        self._prev_raw_target_quat_xyzw = np.copy(target_quat_xyzw)

        # Dynamic Adaptive EMA Filter & Slew-Rate Limiter cho Quaternion (mượt mà, triệt tiêu vi rung)
        if self._ema_quat_xyzw is None:
            self._ema_quat_xyzw = np.copy(target_quat_xyzw)
        else:
            if np.dot(self._ema_quat_xyzw, target_quat_xyzw) < 0.0:
                target_quat_xyzw = -target_quat_xyzw
            
            # Tính khoảng cách góc trắc địa (Geodesic Angle Distance) trên SO(3)
            dot_val = np.clip(abs(float(np.dot(self._ema_quat_xyzw, target_quat_xyzw))), -1.0, 1.0)
            rot_diff = 2.0 * np.arccos(dot_val)
            
            # Angular Slew Rate Limiter (giảm 35-40% so với 0.20 rad)
            max_rot = self.max_rot_step
            if rot_diff > max_rot and rot_diff > 1e-6:
                s_factor = max_rot / rot_diff
                target_quat_xyzw = (1.0 - s_factor) * self._ema_quat_xyzw + s_factor * target_quat_xyzw
                norm_t = np.linalg.norm(target_quat_xyzw)
                if norm_t > 1e-9:
                    target_quat_xyzw /= norm_t
                rot_diff = max_rot

            # Dynamic Adaptive Alpha cho góc xoay
            t_rot = np.clip(rot_diff / 0.25, 0.0, 1.0)
            smooth_rot = t_rot * t_rot * (3.0 - 2.0 * t_rot)
            alpha_rot = 0.20 + (0.50 - 0.20) * smooth_rot
            
            self._ema_quat_xyzw = alpha_rot * target_quat_xyzw + (1.0 - alpha_rot) * self._ema_quat_xyzw
            norm_ema = np.linalg.norm(self._ema_quat_xyzw)
            if norm_ema > 1e-9:
                self._ema_quat_xyzw /= norm_ema
            
        target_quat_wxyz = [
            float(self._ema_quat_xyzw[3]),
            float(self._ema_quat_xyzw[0]),
            float(self._ema_quat_xyzw[1]),
            float(self._ema_quat_xyzw[2])
        ]

        # Kiểm tra tính liên tục dấu Quaternion đầu ra trước khi gửi tới IK
        if hasattr(self, "_last_output_quat_wxyz") and self._last_output_quat_wxyz is not None:
            if np.dot(target_quat_wxyz, self._last_output_quat_wxyz) < 0.0:
                target_quat_wxyz = [-w for w in target_quat_wxyz]
        self._last_output_quat_wxyz = list(target_quat_wxyz)
        self.prev_target_quat = target_quat_wxyz

        return {
            "pos": self._ema_pos.tolist(),
            "quat": target_quat_wxyz
        }

    def get_current_pose(self):
        """Dùng cho lúc init hệ thống"""
        try:
            tcp = self.ctrl.tcp()
            angle = np.linalg.norm(tcp[3:])
            if angle > 1e-6:
                axis = np.array(tcp[3:]) / angle
                quat_wxyz = transforms3d.quaternions.axangle2quat(axis, angle)
            else:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            return {
                "pos": tcp[:3],
                "quat": quat_wxyz.tolist()
            }
        except:
            return None

    def get_delta_translation(self) -> np.ndarray:
        """Trả về vector tịnh tiến [d_forward, d_lateral, d_vertical] (mét) theo thời gian thực"""
        return getattr(self, "last_delta_trans", np.zeros(3, dtype=float))

    def reanchor_position(self, p_actual):
        """
        Chiếu lại anchor vị trí (Target Projection & Re-anchoring):
        Triệt tiêu hiện tượng trôi tích lũy (Windup) khi robot trượt trên bề mặt rào cản.
        Đồng bộ anchor và _ema_pos để khi người dùng đảo chiều kéo ngược ra khỏi bề mặt,
        robot phản hồi ngay lập tức không có độ trễ deadband.
        """
        if self._anchor_tcp is None:
            return
        p_act = np.array(p_actual, dtype=float)[:3]
        if self._ema_pos is not None:
            delta_net = self._ema_pos - self._anchor_tcp[:3]
            dist_net = float(np.linalg.norm(delta_net))
            if dist_net > 0.030:
                delta_net = delta_net * (0.030 / dist_net)
            self._anchor_tcp[:3] = p_act - delta_net
            self._ema_pos = np.copy(p_act)
        else:
            self._anchor_tcp[:3] = np.copy(p_act)

    def reset_anchor(self):
        self._last_button_state = 0
        self.clutch_engaged = False
        self.grip_engaged = False
        self._locked_axis_idx = None
        self._last_locked_axis_idx = None
        self._last_gesture_angle = None
        self.locked_rotation_axis = None
        self._axis_theta_max_pos.clear()
        self._axis_theta_min_neg.clear()
        self._gesture_angle_offset = 0.0
        self._clock_u_radial = None
        self._clock_u_tangential = None
        self._clock_u_up = None
        self._clock_u_tool = None
        self._last_hand_quat_xyzw = None
        self._prev_raw_target_quat_xyzw = None
        self.R_ctrl_origin = None
        self.R_tcp_origin = None
        self.prev_target_quat = None
        self._anchor_pos_quest = None
        self._anchor_quat_quest_xyzw = None
        self._anchor_tcp = None
        self._anchor_quat_robot_xyzw = None
        self._anchor_theta_arm = None
        self._anchor_u_fwd = None
        self._anchor_u_right = None
        self._current_delta_theta = 0.0
        self._ema_pos = None
        self._ema_quat_xyzw = None
        self._last_dominant_rot_axis = None
        self._last_output_quat_wxyz = None
        self.last_delta_trans = np.zeros(3, dtype=float)
        self.last_delta_mag = 0.0
        self.out_of_reach_time = 0.0

    def reset(self):
        """Tương thích ngược cho các module hoặc bài test gọi reset()"""
        self.reset_anchor()

    def clamp_axis_rotation(self, axis_idx: Optional[int], theta_limit: float):
        """
        No-op: Giữ nguyên dải quay tự nhiên của người dùng, không kẹp giảm input.
        """
        pass

    def clamp_rotation_anchor(self, achieved_quat_wxyz):
        """
        Anti-windup re-anchor for rotation:
        When tool rotation hits physical joint limit, clamp the anchor orientation
        along the locked axis WITHOUT corrupting R_tcp_origin with off-axis tilt,
        ensuring zero windup on reverse.
        """
        try:
            # Convert achieved_quat_wxyz [w, x, y, z] to xyzw
            q_achieved_xyzw = np.array([
                achieved_quat_wxyz[1],
                achieved_quat_wxyz[2],
                achieved_quat_wxyz[3],
                achieved_quat_wxyz[0]
            ], dtype=float)
            norm_ach = np.linalg.norm(q_achieved_xyzw)
            if norm_ach > 1e-6:
                q_achieved_xyzw /= norm_ach

            # Chống lật dấu Quaternion giữa q_achieved và anchor robot
            if self._anchor_quat_robot_xyzw is not None:
                if np.dot(q_achieved_xyzw, self._anchor_quat_robot_xyzw) < 0.0:
                    q_achieved_xyzw = -q_achieved_xyzw
            self._anchor_quat_robot_xyzw = q_achieved_xyzw

            # Reset controller anchor to current hand orientation
            if self._last_hand_quat_xyzw is not None:
                self._anchor_quat_quest_xyzw = np.copy(self._last_hand_quat_xyzw)
                self.R_ctrl_origin = R.from_quat(self._anchor_quat_quest_xyzw)
            if self._anchor_quat_robot_xyzw is not None:
                self._ema_quat_xyzw = np.copy(self._anchor_quat_robot_xyzw)
                self.R_tcp_origin = R.from_quat(self._anchor_quat_robot_xyzw)

            self._last_gesture_angle = 0.0
            self._gesture_angle_offset = 0.0

            target_quat_wxyz = [
                float(achieved_quat_wxyz[0]),
                float(achieved_quat_wxyz[1]),
                float(achieved_quat_wxyz[2]),
                float(achieved_quat_wxyz[3])
            ]
            if hasattr(self, "_last_output_quat_wxyz") and self._last_output_quat_wxyz is not None:
                if np.dot(target_quat_wxyz, self._last_output_quat_wxyz) < 0.0:
                    target_quat_wxyz = [-w for w in target_quat_wxyz]
            self.prev_target_quat = list(target_quat_wxyz)
            self._last_output_quat_wxyz = list(target_quat_wxyz)
        except Exception:
            pass
