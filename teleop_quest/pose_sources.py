import numpy as np
import transforms3d

class WidgetPoseSource:
    """Nguồn cấp tọa độ cho Controller (Đọc từ Widget)"""
    def __init__(self, target_widget, alpha=0.55):
        self.target_widget = target_widget
        self.alpha = alpha
        self._ema_pos = None
        self._ema_quat_xyzw = None
        
    def get_pose(self):
        target_pos = np.array(self.target_widget.position)
        # Giới hạn không gian làm việc an toàn trong tầm với của UR3e (bán kính <= 49cm, Z >= 0.00m)
        target_pos[0] = np.clip(target_pos[0], -0.49, 0.49)
        target_pos[1] = np.clip(target_pos[1], -0.49, 0.49)
        target_pos[2] = np.clip(target_pos[2], 0.00, 0.58)
        target_quat_wxyz = self.target_widget.wxyz
        target_quat_xyzw = np.array([target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]])
        
        if self._ema_pos is None:
            self._ema_pos = target_pos
            self._ema_quat_xyzw = target_quat_xyzw
        else:
            self._ema_pos = self.alpha * target_pos + (1.0 - self.alpha) * self._ema_pos
            
            # Slerp/EMA cho Quaternion
            if np.dot(self._ema_quat_xyzw, target_quat_xyzw) < 0:
                target_quat_xyzw = -target_quat_xyzw
            self._ema_quat_xyzw = self.alpha * target_quat_xyzw + (1.0 - self.alpha) * self._ema_quat_xyzw
            self._ema_quat_xyzw /= np.linalg.norm(self._ema_quat_xyzw)
            
        ema_quat_wxyz = [self._ema_quat_xyzw[3], self._ema_quat_xyzw[0], self._ema_quat_xyzw[1], self._ema_quat_xyzw[2]]
        return {"pos": self._ema_pos.tolist(), "quat": ema_quat_wxyz}

    def sync_with_robot(self, joints=None, fk_func=None):
        """Đồng bộ lại bộ lọc EMA với trạng thái mới để tránh giật tọa độ."""
        self._ema_pos = None
        self._ema_quat_xyzw = None

def get_robot_fk_as_ur(teleop, ctrl):
    """
    Dùng vị trí khớp thật (q_real) để tính FK bằng bộ giải PyRoki.
    Giúp loại bỏ sai lệch Calibration giữa URSim thật và PyRoki URDF.
    """
    q_real = ctrl.joints()
    fk_dict = teleop._solver.fk(q_real)
    pos = fk_dict["pos"]
    quat_wxyz = fk_dict["quat"]
    axis, angle = transforms3d.quaternions.quat2axangle(quat_wxyz)
    return [pos[0], pos[1], pos[2], axis[0]*angle, axis[1]*angle, axis[2]*angle]
