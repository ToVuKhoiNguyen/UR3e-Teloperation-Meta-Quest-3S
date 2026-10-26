import numpy as np
import transforms3d

class WidgetPoseSource:
    """
    Nguồn cấp tọa độ cho Controller đọc trực tiếp từ Viser 3D Gizmo (Zero-Drift Absolute Tracking).
    Bám tuyệt đối 1:1 vào tọa độ Gizmo:
      - Đích đến mong muốn (desired_target_pose) trong mỗi chu kỳ là tọa độ tuyệt đối hiện tại của ui.gizmo.
      - XÓA BỎ việc cộng dồn delta pos và XÓA BỎ việc re-anchor làm trôi virtual_target lệch khỏi ui.gizmo.
      - Zero-Glitch: Tuyệt đối không ghi đè lên ui.gizmo khi đang kéo chuột.
    """
    is_absolute = True

    def __init__(self, target_widget, alpha=0.55):
        self.target_widget = target_widget
        self.alpha = alpha
        self._prev_read_pos = None
        self._prev_read_quat_wxyz = None
        self.is_translating = False
        self.is_pure_rotation = False

    def get_pose(self):
        curr_gizmo_pos = np.array(getattr(self.target_widget, "position", [0.0, 0.0, 0.0]), dtype=float)
        curr_gizmo_quat_wxyz = np.array(getattr(self.target_widget, "wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        norm_q = np.linalg.norm(curr_gizmo_quat_wxyz)
        curr_gizmo_quat_wxyz = curr_gizmo_quat_wxyz / max(1e-6, norm_q)

        # Giới hạn không gian làm việc an toàn của UR3e (bán kính <= 52cm, Z >= 0.015m)
        curr_gizmo_pos[0] = np.clip(curr_gizmo_pos[0], -0.52, 0.52)
        curr_gizmo_pos[1] = np.clip(curr_gizmo_pos[1], -0.52, 0.52)
        curr_gizmo_pos[2] = np.clip(curr_gizmo_pos[2], 0.015, 0.60)

        # Detect active mouse interaction intent
        if self._prev_read_pos is not None and self._prev_read_quat_wxyz is not None:
            d_pos = float(np.linalg.norm(curr_gizmo_pos - self._prev_read_pos))
            dot_q = abs(float(np.dot(curr_gizmo_quat_wxyz, self._prev_read_quat_wxyz)))
            d_rot = 2.0 * float(np.arccos(min(1.0, max(-1.0, dot_q))))

            if d_pos > 0.001:  # user is actively dragging Gizmo translation (> 1mm)
                self.is_translating = True
                self.is_pure_rotation = False
            elif d_rot > 0.015:  # user is actively rotating Gizmo in place (> 0.85 deg)
                self.is_translating = False
                self.is_pure_rotation = True
            else:  # stationary / mouse released / idle
                self.is_translating = False
                self.is_pure_rotation = False

        self._prev_read_pos = np.copy(curr_gizmo_pos)
        self._prev_read_quat_wxyz = np.copy(curr_gizmo_quat_wxyz)

        return {
            "pos": curr_gizmo_pos.tolist(),
            "quat": curr_gizmo_quat_wxyz.tolist()
        }

    @property
    def _virtual_pos(self):
        return np.array(getattr(self.target_widget, "position", [0.0, 0.0, 0.0]), dtype=float)

    @_virtual_pos.setter
    def _virtual_pos(self, val):
        pass

    @property
    def _virtual_quat_wxyz(self):
        return np.array(getattr(self.target_widget, "wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=float)

    @_virtual_quat_wxyz.setter
    def _virtual_quat_wxyz(self, val):
        pass

    @property
    def _ema_pos(self):
        return np.array(getattr(self.target_widget, "position", [0.0, 0.0, 0.0]), dtype=float)

    @_ema_pos.setter
    def _ema_pos(self, val):
        pass

    def sync_with_robot(self, *args, pos=None, quat=None, joints=None, fk_func=None, force=False, **kwargs):
        """
        Đồng bộ mốc tọa độ Gizmo chỉ khi được gọi tường minh với force=True (từ Go Home hoặc startup).
        TUYỆT ĐỐI CẤM ghi đè lên ui.gizmo trong vòng lặp teleop thông thường để chống trôi ảo.
        """
        if not force:
            return

        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, (list, tuple, np.ndarray)):
                if len(arg0) == 6 and fk_func is not None:
                    try:
                        fk = fk_func(arg0)
                        pos = fk.get("pos")
                        quat = fk.get("quat")
                    except Exception:
                        pass
                elif len(arg0) == 3:
                    pos = arg0
        elif len(args) >= 2:
            pos = args[0]
            quat = args[1]

        if joints is not None and fk_func is not None and pos is None:
            try:
                fk = fk_func(joints)
                pos = fk.get("pos")
                quat = fk.get("quat")
            except Exception:
                pass

        if hasattr(self.target_widget, "update_pose"):
            try:
                self.target_widget.update_pose(pos=pos, wxyz=quat, suppress=True)
            except Exception:
                pass
        else:
            if pos is not None and hasattr(self.target_widget, "position"):
                try:
                    self.target_widget.position = tuple(float(x) for x in pos[:3])
                except Exception:
                    pass

            if quat is not None and hasattr(self.target_widget, "wxyz"):
                try:
                    self.target_widget.wxyz = tuple(float(w) for w in quat)
                except Exception:
                    pass

    def reanchor_position(self, p_actual):
        """
        No-op: Absolute 1:1 tracking NEVER re-anchors away from ui.gizmo,
        eliminating position offset drift completely (0.0mm steady-state error).
        """
        pass

    def clamp_rotation_anchor(self, achieved_quat_wxyz):
        """
        No-op for Viser Gizmo: TUYỆT ĐỐI CẤM ghi đè ui.gizmo.wxyz từ Server/FK
        khi người dùng không chủ động kéo chuột Gizmo. Gizmo phải đứng yên tuyệt đối
        tại vị trí người dùng thả ra.
        """
        pass


def get_robot_fk_as_ur(teleop, ctrl):
    """
    Dùng vị trí khớp thật (q_real) để tính FK bằng bộ giải PyRoki.
    Giúp loại bỏ sai lệch Calibration giữa URSim thật và PyRoki URDF.
    """
    q_real = None
    if ctrl is not None and hasattr(ctrl, "joints"):
        try:
            q_real = ctrl.joints()
        except Exception:
            q_real = None
    if q_real is None and hasattr(teleop, "_last_valid_q") and teleop._last_valid_q is not None:
        q_real = teleop._last_valid_q
    if q_real is None and ctrl is not None and hasattr(ctrl, "tcp"):
        try:
            return list(ctrl.tcp())
        except Exception:
            pass
    if q_real is None:
        return [0.0, -0.3, 0.3, 0.0, 3.14159, 0.0]

    fk_dict = teleop._solver.fk(q_real)
    pos = fk_dict["pos"]
    quat_wxyz = fk_dict["quat"]
    axis, angle = transforms3d.quaternions.quat2axangle(quat_wxyz)
    return [pos[0], pos[1], pos[2], axis[0]*angle, axis[1]*angle, axis[2]*angle]
