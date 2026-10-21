import os
import time
import math
import numpy as np
import transforms3d
from scipy.spatial.transform import Rotation as R

class DeltaPoseSource:
    """
    Implements PoseSource contract.
    Chuyển delta tọa độ từ kính Quest sang base_link của UR3e.
    """
    def __init__(self, server, ctrl, scale: float = 0.75, max_delta_m: float = 0.22, anchor_reference_func=None, alpha: float = 0.28):
        self.server = server
        self.ctrl = ctrl
        self.scale = float(os.environ.get("TELEOP_SCALE", str(scale)))
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
        
        self.enable_axis_snapping = os.environ.get("AXIS_SNAPPING", "0") == "1"
        self.nav_mode = os.environ.get("NAV_MODE", "cylindrical").lower()
        self.rotation_mode = os.environ.get("ROTATION_MODE", "decoupled_3axis").lower()
        # Hiệu chuẩn hướng bàn máy (Desk-Aligned):
        # Mặc định khi chưa hiệu chuẩn: [0, 0, -1] và [1, 0, 0]
        self._desk_u_fwd = np.array([0.0, 0.0, -1.0])
        self._desk_u_right = np.array([1.0, 0.0, 0.0])
        self._last_btn_b = False

        raw_heading = os.environ.get("HEADING_MODE", "desk_aligned").lower()
        self.heading_mode = raw_heading
        self.axis_snap_thresh = float(os.environ.get("AXIS_SNAP_THRESH", "0.50"))
        self.rot_scale = float(os.environ.get("ROT_SCALE", "1.1"))
        self._current_delta_theta = 0.0
        self._anchor_u_fwd = np.copy(self._desk_u_fwd)
        self._anchor_u_right = np.copy(self._desk_u_right)
        self._last_dominant_trans_axis = None

        # Cấu hình chống nhiễu chuyên sâu (Anti-Noise & Jitter Suppression)
        self.rot_deadband = float(os.environ.get("ROT_DEADBAND", "0.045"))  # ~2.6 độ (triệt tiêu rung tay khi bóp giữ phím)
        self.pos_deadband = float(os.environ.get("POS_DEADBAND", "0.003"))  # 3mm (triệt tiêu vi rung khi giữ yên tay)
        self.rot_snap_low = float(os.environ.get("ROT_SNAP_LOW", "0.55"))   # < 55%: khóa cứng rãnh đơn trục (Phương án A)
        self.rot_snap_high = float(os.environ.get("ROT_SNAP_HIGH", "0.75")) # > 75%: mở xoay 3D đa trục khi chủ ý bẻ chéo lớn
        self._last_dominant_rot_axis = None

        self._last_button_state = 0
        self.out_of_reach_time = 0.0

        if hasattr(self.server, "set_q0_provider") and hasattr(self.ctrl, "joints"):
            try:
                self.server.set_q0_provider(lambda: self.ctrl.joints()[0] if self.ctrl is not None and hasattr(self.ctrl, "joints") and self.ctrl.joints() is not None else None)
            except Exception:
                pass

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

    def _apply_soft_rotation_snapping(self, rotvec_tcp, rot_deadband=None):
        """
        Khóa trục xoay thông minh (Soft Rotation Snapping) & Bộ lọc triệt tiêu rò rỉ chéo đa tầng:
        - Soft deadband (mặc định 0.045 rad ~ 2.6 độ): Triệt tiêu hoàn toàn vi rung sinh học khi bóp giữ phím Grip.
        - Khóa trục độc lập (< 30%): Khi người dùng chủ ý xoay 1 trục (Pitch, Yaw hoặc Roll),
          các trục phụ nhỏ hơn 30% trục chính bị triệt tiêu hoàn toàn về 0.0.
        - Chống đảo trục chập chờn (Hysteresis Latch): Ưu tiên giữ trục chính đã chọn nếu trục mới chưa vượt trội > 15%.
        - Hòa trộn 3D tự nhiên (30% - 50%): Chuyển tiếp mượt mà bằng đường cong Hermite;
          khi > 50%, mở khóa hoàn toàn cho phép xoay 3D đa trục tự do.
        """
        if rot_deadband is None:
            rot_deadband = self.rot_deadband

        mag = np.linalg.norm(rotvec_tcp)
        if mag < rot_deadband:
            self._last_dominant_rot_axis = None
            return np.zeros(3)

        # Trừ hao deadband để chuyển động bắt đầu từ 0 liên tục, không giật nảy
        scaled_rotvec = rotvec_tcp * ((mag - rot_deadband) / mag)
        abs_v = np.abs(scaled_rotvec)
        curr_max_idx = int(np.argmax(abs_v))
        curr_max_val = abs_v[curr_max_idx]

        if curr_max_val < 1e-4:
            self._last_dominant_rot_axis = None
            return np.zeros(3)

        # Hysteresis Latch: Giữ ổn định trục chính, tránh hiện tượng rung giật lật trục khi 2 trục gần bằng nhau
        dominant_idx = curr_max_idx
        if self._last_dominant_rot_axis is not None and self._last_dominant_rot_axis != curr_max_idx:
            prev_val = abs_v[self._last_dominant_rot_axis]
            if prev_val > 0.85 * curr_max_val:
                dominant_idx = self._last_dominant_rot_axis
                curr_max_val = prev_val
        self._last_dominant_rot_axis = dominant_idx

        snapped = np.copy(scaled_rotvec)
        for i in range(3):
            if i == dominant_idx:
                continue
            ratio = abs_v[i] / curr_max_val
            if ratio < self.rot_snap_low:
                snapped[i] = 0.0
            elif ratio < self.rot_snap_high:
                t = (ratio - self.rot_snap_low) / (self.rot_snap_high - self.rot_snap_low)
                scale = t * t * (3.0 - 2.0 * t) # Hermite smoothstep
                snapped[i] = scaled_rotvec[i] * scale
            else:
                snapped[i] = scaled_rotvec[i]

        return snapped

    def get_pose(self):
        frame = self.server.get_frame()
        if not frame:
            return None

        # 1. Hiệu chuẩn hướng bàn máy khi người dùng bấm button_b (bắt cạnh lên):
        # Lấy yaw của head_quat lúc đó làm _desk_u_fwd / _desk_u_right
        btn_b = bool(frame.get("button_b", False) or frame.get("buttons", {}).get("button_b", False))
        if btn_b and not self._last_btn_b:
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
                print(f"[HIỆU CHUẨN] bàn máy yaw={desk_yaw:+.1f}°")
            else:
                self._desk_u_fwd = np.array([0.0, 0.0, -1.0])
                self._desk_u_right = np.array([1.0, 0.0, 0.0])
                print("[HIỆU CHUẨN] bàn máy yaw=+0.0°")
        self._last_btn_b = btn_b

        # Không sinh delta pose khi đang ở Edit Mode để tránh di chuyển ngoài ý muốn
        if hasattr(self.server, "get_drive_mode") and not self.server.get_drive_mode():
            self._last_button_state = 0
            return None

        # Tách biệt điều khiển Tịnh tiến (Trigger) và Xoay (Grip)
        is_translating = frame.get("trigger", False)
        is_rotating = frame.get("grip", False)
        
        current_button_state = 0
        if is_translating: current_button_state += 1
        if is_rotating: current_button_state += 2
        
        if current_button_state == 0:
            self._last_button_state = 0
            return None

        # TÍNH NĂNG CLUTCH THÔNG MINH:
        # Bất cứ khi nào sếp thay đổi nút bấm (Vd: Đang Tịnh tiến -> Chuyển sang Xoay),
        # Hệ thống sẽ TỰ ĐỘNG CHỐT MỎ NEO MỚI (Reset Anchor). 
        # Điều này loại bỏ hoàn toàn lỗi "nhảy cóc (teleport) tọa độ" do bị ép biến delta về 0.
        if current_button_state != self._last_button_state:
            self._anchor_pos_quest = np.array(frame["pos"])
            
            q_q = frame["quat"] # [w, x, y, z]
            self._anchor_quat_quest_xyzw = np.array([q_q[1], q_q[2], q_q[3], q_q[0]])

            
            # Neo vào vị trí thực tế của robot (hoặc Widget 3D trên màn hình)
            try:
                if self.anchor_reference_func is not None:
                    # Lấy tọa độ từ Widget 3D (x, y, z, qw, qx, qy, qz)
                    ref_pose = self.anchor_reference_func()
                    self._anchor_tcp = np.array(ref_pose)
                else:
                    self._anchor_tcp = np.array(self.ctrl.tcp())
            except Exception:
                self._last_trigger = False
                return None
            
            # Tính toán quaternion neo của robot
            axang = self._anchor_tcp[3:]
            angle = np.linalg.norm(axang)
            if angle > 1e-6:
                axis = axang / angle
                qwxyz = transforms3d.quaternions.axangle2quat(axis, angle)
                self._anchor_quat_robot_xyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])
            else:
                self._anchor_quat_robot_xyzw = np.array([0., 0., 0., 1.])
            
            # Lấy góc quay của mặt phẳng cánh tay cơ học (khớp đế q0)
            try:
                joints = self.ctrl.joints()
                if joints is not None and len(joints) > 0:
                    self._anchor_theta_arm = float(joints[0])
                else:
                    self._anchor_theta_arm = None
            except Exception:
                self._anchor_theta_arm = None
            
            # Xác lập hệ quy chiếu điều khiển (Heading Reference) - Hiệu chuẩn theo bàn máy:
            # Mỗi lần bóp cò: Dùng _desk_u_fwd / _desk_u_right làm _anchor_u_fwd / _anchor_u_right,
            # KHÔNG tính lại từ head_quat.
            self._anchor_u_fwd = np.copy(self._desk_u_fwd)
            self._anchor_u_right = np.copy(self._desk_u_right)

            self._ema_pos = np.copy(self._anchor_tcp[:3])
            self._ema_quat_xyzw = np.copy(self._anchor_quat_robot_xyzw)
            self._last_dominant_rot_axis = None
            self._last_dominant_trans_axis = None
            self._last_button_state = current_button_state
            return None # Bỏ qua frame đầu tiên để tránh giật

        # ===============================
        # TÍNH TOÁN DELTA TỊNH TIẾN (HEAD-CENTRIC & WORLD)
        # ===============================
        diag_msg = "STANDING STILL"
        delta_mag = 0.0
        
        R_q_anchor = R.from_quat(self._anchor_quat_quest_xyzw)
        R_r_anchor = R.from_quat(self._anchor_quat_robot_xyzw)
        
        if is_translating:
            current_pos_quest = np.array(frame["pos"])
            # delta_q_world là sự thay đổi vị trí tay người dùng trong KHÔNG GIAN THỰC
            # (Trái/Phải, Lên/Xuống, Tiến/Lùi) so với lúc vừa bấm cò
            delta_q_world = current_pos_quest - self._anchor_pos_quest
            
            # --- BỘ LỌC CHỐNG NHIỄU & CHẨN ĐOÁN ---
            delta_mag = np.linalg.norm(delta_q_world)
            
            if delta_mag < self.pos_deadband:  # Deadband tĩnh 6mm (loại bỏ vi rung sinh học khi bóp giữ cò)
                delta_q_world = np.zeros(3)
            else:
                # SOFT DEADBAND: Khử độ nhảy vọt khi vượt ngưỡng
                delta_q_world = delta_q_world * ((delta_mag - self.pos_deadband) / delta_mag)
                        
            # Chiếu cử động tay người dùng theo hệ quy chiếu bàn máy / hướng nhìn:
            if self._anchor_u_fwd is not None and self._anchor_u_right is not None:
                d_forward_raw = float(np.dot(delta_q_world, self._anchor_u_fwd))
                d_lateral_raw = float(np.dot(delta_q_world, self._anchor_u_right))
                d_vertical_raw = float(delta_q_world[1])
            else:
                d_forward_raw = -delta_q_world[2]   # Tiến (-Z) / Lùi (+Z)
                d_lateral_raw =  delta_q_world[0]   # Phải (+X) / Trái (-X)
                d_vertical_raw = delta_q_world[1]   # Lên (+Y) / Xuống (-Y)

            # Áp dụng Thuật toán triệt tiêu nhiễu chéo đa trục thông minh (Smooth Axis Isolation):
            if self.enable_axis_snapping:
                d_snapped = self._apply_aggressive_axis_isolation(np.array([d_forward_raw, d_lateral_raw, d_vertical_raw]))
                d_forward = d_snapped[0]
                d_lateral = d_snapped[1]
                d_vertical = d_snapped[2]
            else:
                d_forward = d_forward_raw
                d_lateral = d_lateral_raw
                d_vertical = d_vertical_raw

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
                delta_arc = d_lateral * self.scale
                delta_arc = np.clip(delta_arc, -self.max_delta_m, self.max_delta_m)
                delta_q0 = -delta_arc / max(r_reach, 0.12)
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
                # Vector vươn u_reach nằm trong mặt phẳng tay máy tại góc đế q0_target:
                u_reach = np.array([np.cos(q0_target), np.sin(q0_target), 0.0])
                # Tăng hệ số nhạy hướng Tiến/Lùi lên mức 1:1 (scale_reach = 1.05) để vươn dứt khoát
                scale_reach = float(os.environ.get("REACH_SCALE", "1.05"))
                delta_reach = d_forward * scale_reach
                delta_reach = np.clip(delta_reach, -self.max_delta_m, self.max_delta_m)

                # 3. Lên / Xuống -> Nâng hạ chiều cao Z:
                delta_z = d_vertical * self.scale
                delta_z = np.clip(delta_z, -self.max_delta_m, self.max_delta_m)
                z_target = np.clip(p_anc[2] + delta_z, 0.00, 0.58)

                # Tọa độ đích kết hợp:
                target_pos = p_rot + delta_reach * u_reach
                target_pos[2] = z_target

                # Giới hạn an toàn bán kính cơ học:
                # Dải [0.10m, 0.485m] loại bỏ kẹt lùi, bảo vệ tầm với 500mm của UR3e
                r_target_xy = np.hypot(target_pos[0], target_pos[1])
                if r_target_xy < 0.10:
                    scale_r = 0.10 / max(r_target_xy, 1e-6)
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
                delta_robot_world = np.array([dx_robot, dy_robot, dz_robot]) * self.scale
                delta_robot_world = np.clip(delta_robot_world, -self.max_delta_m, self.max_delta_m)
                target_pos = self._anchor_tcp[:3] + delta_robot_world
                self._current_delta_theta = 0.0
            else:
                delta_robot_world = np.array([
                    d_forward,
                    -d_lateral,
                    d_vertical
                ]) * self.scale
                delta_robot_world = np.clip(delta_robot_world, -self.max_delta_m, self.max_delta_m)
                target_pos = self._anchor_tcp[:3] + delta_robot_world
                self._current_delta_theta = 0.0
        else:
            delta_robot_world = np.zeros(3)
            diag_msg = "CHỈ XOAY (KHÓA TỊNH TIẾN)"
            target_pos = np.copy(self._anchor_tcp[:3])
            self._current_delta_theta = 0.0
            self._last_dominant_trans_axis = None

        # Workspace safety clamp (Chuẩn tầm với UR3e 500mm, Z an toàn tuyệt đối trên mặt bàn)
        target_pos[0] = np.clip(target_pos[0], -0.49, 0.49) # X
        target_pos[1] = np.clip(target_pos[1], -0.49, 0.49) # Y
        target_pos[2] = np.clip(target_pos[2], 0.00, 0.58)  # Z (an toàn tuyệt đối không đâm bàn)

        # Slew Rate Limiter: Tăng trần max_step lên 0.055 (5.5cm / 50ms = 1.1 m/s) giúp phản hồi vươn tay nhanh
        if self._ema_pos is not None:
            pos_step = target_pos - self._ema_pos
            step_mag = np.linalg.norm(pos_step)
            max_step = 0.055  # tối đa 5.5cm / 50ms = 1.1 m/s
            if step_mag > max_step:
                target_pos = self._ema_pos + pos_step * (max_step / step_mag)

        # Dynamic Adaptive EMA Filter cho Position:
        # - Khi đứng yên hoặc vi rung nhẹ (< 1cm): alpha = 0.22 (lọc phẳng vi rung)
        # - Khi di chuyển dứt khoát (> 3cm): alpha = 0.55 (phản hồi tức thì, không bị trễ)
        if self._ema_pos is None:
            self._ema_pos = target_pos
        else:
            pos_diff = np.linalg.norm(target_pos - self._ema_pos)
            t_pos = np.clip(pos_diff / 0.03, 0.0, 1.0)
            smooth_pos = t_pos * t_pos * (3.0 - 2.0 * t_pos)
            alpha_pos = 0.22 + (0.55 - 0.22) * smooth_pos
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
        # TÍNH TOÁN DELTA XOAY (DECOUPLED 3-AXIS / TABLE ROLL / FREE 3D)
        # ===============================
        if is_rotating:
            curr_q = frame["quat"]
            curr_quat_quest_xyzw = np.array([curr_q[1], curr_q[2], curr_q[3], curr_q[0]])
            R_hand_current = R.from_quat(curr_quat_quest_xyzw)

            # 1. Delta xoay trong Không gian Thực (WebXR World Frame):
            # R_delta_world = R_hand_current * R_q_anchor.inv()
            # Trực giao 100% giữa các trục không gian phòng, triệt tiêu hoàn toàn góc nghiêng tay cầm:
            # - omega_world[0]: Gập/ngửa cổ tay lên xuống (Pitch quanh trục X phòng)
            # - omega_world[1]: Quét/lắc cổ tay trái phải quanh trục thẳng đứng (Yaw quanh trục Y phòng)
            # - omega_world[2]: Nghiêng/vặn cổ tay quanh hướng nhìn tới (Roll quanh trục Z phòng)
            R_delta_world = R_hand_current * R_q_anchor.inv()
            omega_world = R_delta_world.as_rotvec()

            # Chiếu vào 3 trục điều khiển tự nhiên của người dùng:
            # - Pitch: Đảo ngược chiều theo yêu cầu: dốc tay về phía mình thì ngửa ra xa thân máy!
            # - Hoán đổi & phân định rõ 3 trục cơ học:
            #   + Lắc cổ tay trái/phải (-omega_world[1]) -> Xoay tròn hai ngón kẹp quanh trục kẹp (Tool Z Roll)
            #   + Vặn chìa khóa (omega_world[2]) -> Nghiêng trái / nghiêng phải (Lateral Tilt quanh u_reach)
            d_pitch =  omega_world[0]
            d_yaw   =  omega_world[2]   # Cử chỉ Vặn chìa khóa quanh cẳng tay: nghiêng trái / phải
            d_roll  = -omega_world[1]   # Cử chỉ Lắc cổ tay trái/phải: xoay ngàm đón vật

            raw_rotvec = np.array([d_pitch, d_yaw, d_roll]) * self.rot_scale

            if self.rotation_mode == "table_roll":
                # Chế độ Proposal 4 (Khóa chúc ngàm, chỉ Roll quanh trục kẹp):
                mag_r = abs(d_roll)
                rot_deadband = self.rot_deadband
                if mag_r < rot_deadband:
                    delta_roll = 0.0
                else:
                    delta_roll = np.sign(d_roll) * (mag_r - rot_deadband)
                if self.nav_mode in ("cylindrical", "clock_face") and is_translating:
                    R_delta_arm = R.from_rotvec([0.0, 0.0, self._current_delta_theta])
                else:
                    R_delta_arm = R.identity()
                R_delta_tool = R.from_rotvec([0.0, 0.0, delta_roll])

                if not is_translating:
                    roll_deg = np.degrees(delta_roll)
                    if abs(roll_deg) > 2.0:
                        direction_str = "PHẢI (CW)" if delta_roll > 0 else "TRÁI (CCW)"
                        diag_msg = f"XOAY NGÀM {direction_str} ({abs(roll_deg):.1f}° | Lắc Cổ Tay)"
                    else:
                        diag_msg = "GIỮ HƯỚNG CŨ"
            elif self.rotation_mode == "free_3d":
                # Chế độ 3D tự do (không khóa trục)
                mag = np.linalg.norm(raw_rotvec)
                rot_deadband = self.rot_deadband
                if mag < rot_deadband:
                    rotvec_tcp = np.zeros(3)
                else:
                    rotvec_tcp = raw_rotvec * ((mag - rot_deadband) / mag)
                if self.nav_mode in ("cylindrical", "clock_face") and is_translating:
                    R_delta_arm = R.from_rotvec([0.0, 0.0, self._current_delta_theta])
                else:
                    R_delta_arm = R.identity()
                R_delta_tool = R.from_rotvec(rotvec_tcp)

                if not is_translating:
                    abs_v = np.abs(rotvec_tcp)
                    max_idx = int(np.argmax(abs_v))
                    max_deg = np.degrees(abs_v[max_idx])
                    axis_names = ["GẬP/CHÚC (Pitch)", "NGHIÊNG TRÁI/PHẢI (Tilt)", "LẮC XOAY NGÀM (Roll)"]
                    if max_deg > 2.0:
                        diag_msg = f"XOAY 3D TỰ DO {axis_names[max_idx]} ({max_deg:.1f}°)"
                    else:
                        diag_msg = "GIỮ HƯỚNG CŨ"
            else:
                # Chế độ chuẩn DECOUPLED 3-AXIS (Mặc định: Kết hợp Phương án A + B):
                # 1. Phương án A: Khóa rãnh đơn trục (Soft Rotation Snapping với ngưỡng 55%) triệt tiêu 100% rung lắc
                if self.enable_axis_snapping:
                    rotvec_filtered = self._apply_soft_rotation_snapping(raw_rotvec)
                else:
                    mag = np.linalg.norm(raw_rotvec)
                    rotvec_filtered = np.zeros(3) if mag < self.rot_deadband else raw_rotvec * ((mag - self.rot_deadband) / mag)

                pitch_val = rotvec_filtered[0]
                yaw_val   = rotvec_filtered[1]
                roll_val  = rotvec_filtered[2]

                # 2. Phương án B: Đồng bộ xoay theo Mặt phẳng cánh tay robot (Arm-Plane / Cylindrical Rotation)
                if self.nav_mode == "arm_plane":
                    theta_arm = self._anchor_theta_arm
                    if theta_arm is None:
                        x_anc = self._anchor_tcp[0]
                        y_anc = self._anchor_tcp[1]
                        r_anc = np.hypot(x_anc, y_anc)
                        theta_arm = np.arctan2(y_anc, x_anc) if r_anc > 0.03 else 0.0

                    # Vector đơn vị vươn dọc theo cánh tay trong Base:
                    u_reach = np.array([np.cos(theta_arm), np.sin(theta_arm), 0.0])
                    # Trục ngang tiếp tuyến vuông góc với phương duỗi cánh tay trong Base:
                    u_lateral = np.array([np.sin(theta_arm), -np.cos(theta_arm), 0.0])

                    # Pitch gập dọc mặt phẳng cánh tay vươn (xoay quanh u_lateral)
                    # Tilt nghiêng trái/phải dọc theo trục vươn của cánh tay (xoay quanh u_reach)
                    rotvec_arm = u_lateral * pitch_val + u_reach * yaw_val
                    R_delta_arm = R.from_rotvec(rotvec_arm)
                    # Roll vặn quanh trục kẹp Tool Z (Lắc cổ tay trái/phải)
                    R_delta_tool = R.from_rotvec([0.0, 0.0, roll_val])
                elif self.nav_mode in ("cylindrical", "clock_face"):
                    # Đồng bộ hóa góc xoay ngàm kẹp theo chuyển động quay kim đồng hồ của Base (quanh trục Z)
                    if is_translating:
                        R_delta_arm = R.from_rotvec([0.0, 0.0, self._current_delta_theta])
                    else:
                        R_delta_arm = R.identity()
                    R_delta_tool = R.from_rotvec(rotvec_filtered)
                else:
                    R_delta_arm = R.identity()
                    R_delta_tool = R.from_rotvec(rotvec_filtered)

                if not is_translating:
                    abs_v = np.abs(rotvec_filtered)
                    max_idx = int(np.argmax(abs_v))
                    max_deg = np.degrees(abs_v[max_idx])
                    axis_names = ["GẬP/CHÚC (Pitch)", "NGHIÊNG TRÁI/PHẢI (Tilt)", "LẮC XOAY NGÀM (Roll)"]
                    if max_deg > 2.0:
                        ratios = [abs_v[i] / (abs_v[max_idx] + 1e-6) for i in range(3)]
                        secondary = [i for i in range(3) if i != max_idx and ratios[i] > 0.50]
                        if secondary:
                            diag_msg = f"XOAY 3D ĐA TRỤC ({max_deg:.1f}° | {axis_names[max_idx]})"
                        else:
                            diag_msg = f"XOAY {axis_names[max_idx]} ({max_deg:.1f}° | Lọc độc lập)"
                    else:
                        diag_msg = "GIỮ HƯỚNG CŨ"

            # Áp dụng góc xoay vào Robot:
            # - R_delta_arm tác động trong hệ tọa độ Base / Arm-Plane (Gập và Quét theo mặt phẳng tay)
            # - R_delta_tool tác động trong hệ tọa độ Tool TCP (Vặn chìa khóa Roll)
            R_target_robot = R_delta_arm * R_r_anchor * R_delta_tool
        else:
            self._last_dominant_rot_axis = None
            if self.nav_mode in ("cylindrical", "clock_face") and is_translating:
                # Khi chỉ bóp Cò tịnh tiến trong chế độ Đồng hồ / Trụ:
                # Ngàm kẹp tự động xoay theo góc quay Base quanh trục Z để giữ nguyên góc đón vật
                R_base_rot = R.from_rotvec([0.0, 0.0, self._current_delta_theta])
                R_target_robot = R_base_rot * R_r_anchor
            else:
                R_target_robot = R_r_anchor
        target_quat_xyzw = R_target_robot.as_quat()
        
        # Dynamic Adaptive EMA Filter cho Quaternion (mượt mà, triệt tiêu vi rung)
        if self._ema_quat_xyzw is None:
            self._ema_quat_xyzw = target_quat_xyzw
        else:
            if np.dot(self._ema_quat_xyzw, target_quat_xyzw) < 0:
                target_quat_xyzw = -target_quat_xyzw
            
            # Tính khoảng cách góc trắc địa (Geodesic Angle Distance) trên SO(3)
            dot_val = np.clip(abs(np.dot(self._ema_quat_xyzw, target_quat_xyzw)), -1.0, 1.0)
            rot_diff = 2.0 * np.arccos(dot_val)
            
            # Slew Rate Limiter cho góc xoay (chống giật đột biến nếu tracking bị vấp)
            max_rot_step = 0.20  # ~11.5 độ / 50ms
            if rot_diff > max_rot_step:
                s_factor = max_rot_step / rot_diff
                target_quat_xyzw = (1.0 - s_factor) * self._ema_quat_xyzw + s_factor * target_quat_xyzw
                target_quat_xyzw /= np.linalg.norm(target_quat_xyzw)
                rot_diff = max_rot_step

            # Dynamic Adaptive Alpha cho góc xoay:
            # - Khi đứng yên hoặc vi rung chậm (< 4 độ): alpha = 0.18 (lọc phẳng 100% rung lắc)
            # - Khi xoay dứt khoát (> 15 độ): alpha = 0.48 (phản hồi tức thì, không trễ)
            t_rot = np.clip(rot_diff / 0.25, 0.0, 1.0)
            smooth_rot = t_rot * t_rot * (3.0 - 2.0 * t_rot)
            alpha_rot = 0.18 + (0.48 - 0.18) * smooth_rot
            
            self._ema_quat_xyzw = alpha_rot * target_quat_xyzw + (1.0 - alpha_rot) * self._ema_quat_xyzw
            self._ema_quat_xyzw /= np.linalg.norm(self._ema_quat_xyzw)
            
        target_quat_wxyz = [self._ema_quat_xyzw[3], self._ema_quat_xyzw[0], self._ema_quat_xyzw[1], self._ema_quat_xyzw[2]]

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

    def reset_anchor(self):
        self._last_button_state = 0
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
        self.out_of_reach_time = 0.0
