import math
import cv2
import numpy as np
import transforms3d
from typing import Optional, Union

class SyntheticWristCamera:
    """
    Camera ảo gắn trên cổ tay robot UR3e (Eye-in-Hand Wrist Camera - RealSense D435i).
    - Render trực tiếp hình ảnh 3D từ góc nhìn của đầu gắp robot nhìn xuống mặt bàn.
    - Nhìn thấy quả cà chua đỏ, khay chứa xanh, và hai ngón tay kẹp của robot.
    - Cung cấp frame RGB chuẩn (480, 640, 3) cho DataLogger (ACT / Diffusion Policy).
    """
    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 65.0):
        self.width = width
        self.height = height
        self.fx = (self.width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def render(
        self,
        tool_pos: Optional[Union[list, np.ndarray]] = None,
        tool_quat: Optional[Union[list, np.ndarray]] = None,
        tomato_pos: Optional[np.ndarray] = None,
        basket_pos: Optional[np.ndarray] = None,
        gripper_closed: bool = False,
        tcp_pose: Optional[Union[list, np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Render khung cảnh 3D từ góc nhìn của TCP sang ảnh RGB (uint8).
        tool_pos/tcp_pose: [x, y, z] trong hệ tọa độ robot world (hoặc [x, y, z, rx, ry, rz] nếu cũ)
        tool_quat: [w, x, y, z] Quaternion từ PyRoki FK (hoặc None nếu truyền tcp_pose cũ)
        """
        if tool_pos is None and tcp_pose is not None:
            tool_pos = tcp_pose

        # Xử lý tương thích ngược nếu truyền kiểu cũ render(tcp_pose, tomato_pos, basket_pos, ...)
        if tool_quat is not None and not isinstance(tool_quat, (list, tuple, np.ndarray)):
            # Trường hợp gọi theo cú pháp cũ: (tcp_pose, tomato_pos, basket_pos, gripper_closed)
            gripper_closed = bool(basket_pos) if basket_pos is not None else False
            basket_pos = tomato_pos
            tomato_pos = tool_quat
            tool_quat = None

        pos_cam = np.array(tool_pos[:3], dtype=np.float32) if tool_pos is not None else np.array([0.0, 0.3, 0.35], dtype=np.float32)

        if tool_quat is not None and len(tool_quat) == 4:
            # PyRoki FK Quaternion: [w, x, y, z]
            R_world_tool = transforms3d.quaternions.quat2mat(tool_quat)
        elif tool_quat is not None and len(tool_quat) == 3:
            rot_axis = np.array(tool_quat, dtype=np.float32)
            angle = np.linalg.norm(rot_axis)
            R_world_tool = transforms3d.axangles.axangle2mat(rot_axis / angle, angle) if angle > 1e-6 else np.eye(3)
        elif tool_pos is not None and len(tool_pos) >= 6:
            rot_axis = np.array(tool_pos[3:], dtype=np.float32)
            angle = np.linalg.norm(rot_axis)
            R_world_tool = transforms3d.axangles.axangle2mat(rot_axis / angle, angle) if angle > 1e-6 else np.eye(3)
        else:
            R_world_tool = np.eye(3)

        # Trong hệ robot UR3e: Trục Z của tool0 chúc thẳng xuống mặt bàn (-Z_world).
        # Camera quang học (OpenCV): +Z là tia nhìn thẳng ra ngoài thấu kính,
        # +X sang phải ảnh, +Y hướng xuống ảnh (về phía thân robot).
        # R_cam_world = R_world_tool^T đưa vector từ World sang Camera frame.
        R_cam_world = R_world_tool.T

        # 1. Tạo ảnh nền (Mặt bàn xám nhạt với lưới ô vuông)
        img = np.full((self.height, self.width, 3), (218, 220, 224), dtype=np.uint8)

        def project_point(p_world):
            """Chiếu điểm 3D world sang pixel 2D (u, v, z_cam)"""
            if p_world is None:
                return None, None, 1.0
            p_rel = np.array(p_world, dtype=np.float32) - pos_cam
            p_cam = R_cam_world @ p_rel
            z = float(p_cam[2])
            if z <= 0.02 or math.isnan(z):  # Phía sau camera hoặc quá sát thấu kính
                return None, None, z
            u = int(self.fx * (p_cam[0] / z) + self.cx)
            v = int(self.fy * (p_cam[1] / z) + self.cy)
            return u, v, z

        # Vẽ lưới ô vuông trên mặt bàn (Z = 0) giúp người dùng nhận biết chiều sâu và chuyển động
        for gx in np.linspace(-0.4, 0.4, 9):
            p1 = project_point(np.array([gx, 0.05, 0.0]))
            p2 = project_point(np.array([gx, 0.65, 0.0]))
            if p1[0] is not None and p2[0] is not None:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (202, 206, 212), 1)
        for gy in np.linspace(0.05, 0.65, 7):
            p1 = project_point(np.array([-0.4, gy, 0.0]))
            p2 = project_point(np.array([0.4, gy, 0.0]))
            if p1[0] is not None and p2[0] is not None:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (202, 206, 212), 1)

        # 2. Vẽ khay đựng giỏ (Khối hình chữ nhật xanh dương)
        u_b, v_b, z_b = project_point(basket_pos)
        if u_b is not None and 0 <= u_b < self.width and 0 <= v_b < self.height:
            w_box = max(20, int(self.fx * 0.15 / z_b))
            h_box = max(15, int(self.fy * 0.15 / z_b))
            cv2.rectangle(
                img,
                (u_b - w_box // 2, v_b - h_box // 2),
                (u_b + w_box // 2, v_b + h_box // 2),
                (60, 150, 230),
                -1
            )
            cv2.putText(img, "Basket", (u_b - 25, v_b - h_box // 2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 100, 180), 1)

        # 3. Vẽ quả cà chua đỏ
        u_t, v_t, z_t = project_point(tomato_pos)
        if u_t is not None and -100 <= u_t < self.width + 100 and -100 <= v_t < self.height + 100:
            rad = max(8, int(self.fx * 0.032 / z_t))
            # Bóng đổ trên mặt bàn
            cv2.ellipse(img, (u_t, v_t + int(rad * 0.8)), (rad, int(rad * 0.4)), 0, 0, 360, (160, 160, 165), -1)
            # Thân quả cà chua đỏ tươi
            cv2.circle(img, (u_t, v_t), rad, (235, 45, 45), -1)
            # Điểm sáng phản quang (Highlight)
            cv2.circle(img, (u_t - int(rad * 0.3), v_t - int(rad * 0.3)), int(rad * 0.25), (255, 130, 130), -1)
            # Cuống lá xanh
            stem_r = max(2, int(rad * 0.25))
            cv2.circle(img, (u_t, v_t - int(rad * 0.7)), stem_r, (40, 180, 50), -1)

        # 4. Vẽ ngón tay kẹp robot (Gripper fingers) cố định ở hai bên mép dưới khung nhìn
        finger_w = 45
        finger_h = 120
        # Nếu kẹp đóng: hai ngón tay khép vào giữa
        offset_x = 100 if gripper_closed else 150
        # Ngón trái
        cv2.rectangle(img, (int(self.cx - offset_x - finger_w), self.height - finger_h),
                      (int(self.cx - offset_x), self.height), (70, 75, 80), -1)
        # Ngón phải
        cv2.rectangle(img, (int(self.cx + offset_x), self.height - finger_h),
                      (int(self.cx + offset_x + finger_w), self.height), (70, 75, 80), -1)

        return img

    def render_depth(
        self,
        tool_pos: Optional[Union[list, np.ndarray]] = None,
        tool_quat: Optional[Union[list, np.ndarray]] = None,
        tomato_pos: Optional[np.ndarray] = None,
        basket_pos: Optional[np.ndarray] = None,
        gripper_closed: bool = False,
        tcp_pose: Optional[Union[list, np.ndarray]] = None,
        **kwargs
    ):
        """
        Render bản đồ độ sâu mô phỏng (Synthetic Depth Stream):
        - Trả về tuple: (depth_colored_rgb, depth_raw_uint16_mm)
        - depth_colored_rgb: shape (H, W, 3) hệ màu RGB qua RealSense Turbo Colorizer
        - depth_raw_uint16_mm: shape (H, W) đơn vị mm (chuẩn format z16)
        """
        if tool_pos is None and tcp_pose is not None:
            tool_pos = tcp_pose

        if tool_quat is not None and not isinstance(tool_quat, (list, tuple, np.ndarray)):
            gripper_closed = bool(basket_pos) if basket_pos is not None else False
            basket_pos = tomato_pos
            tomato_pos = tool_quat
            tool_quat = None

        pos_cam = np.array(tool_pos[:3], dtype=np.float32) if tool_pos is not None else np.array([0.0, 0.3, 0.35], dtype=np.float32)

        if tool_quat is not None and len(tool_quat) == 4:
            R_world_tool = transforms3d.quaternions.quat2mat(tool_quat)
        elif tool_quat is not None and len(tool_quat) == 3:
            rot_axis = np.array(tool_quat, dtype=np.float32)
            angle = np.linalg.norm(rot_axis)
            R_world_tool = transforms3d.axangles.axangle2mat(rot_axis / angle, angle) if angle > 1e-6 else np.eye(3)
        elif tool_pos is not None and len(tool_pos) >= 6:
            rot_axis = np.array(tool_pos[3:], dtype=np.float32)
            angle = np.linalg.norm(rot_axis)
            R_world_tool = transforms3d.axangles.axangle2mat(rot_axis / angle, angle) if angle > 1e-6 else np.eye(3)
        else:
            R_world_tool = np.eye(3)

        R_cam_world = R_world_tool.T

        def project_point(p_world):
            if p_world is None:
                return None, None, 1.0
            p_rel = np.array(p_world, dtype=np.float32) - pos_cam
            p_cam = R_cam_world @ p_rel
            z = float(p_cam[2])
            if z <= 0.02:
                return None, None, z
            u = int(self.fx * (p_cam[0] / z) + self.cx)
            v = int(self.fy * (p_cam[1] / z) + self.cy)
            return u, v, z

        # 1. Nền và mặt bàn (Độ sâu mô phỏng từ camera xuống mặt bàn z=0)
        z_table_est = max(0.12, float(pos_cam[2]))
        z_table_mm = int(z_table_est * 1000)

        # Gradient theo chiều cao ảnh (mép dưới gần camera hơn mép trên)
        depth_raw = np.zeros((self.height, self.width), dtype=np.uint16)
        y_indices = np.arange(self.height, dtype=np.float32)
        table_rows = np.clip(z_table_mm + (self.height - y_indices - self.height * 0.35) * 0.45, 120, 1500).astype(np.uint16)
        depth_raw[:, :] = table_rows[:, None]

        # 2. Khay giỏ
        u_b, v_b, z_b = project_point(basket_pos)
        if u_b is not None and 0 <= u_b < self.width and 0 <= v_b < self.height:
            w_box = max(20, int(self.fx * 0.15 / z_b))
            h_box = max(15, int(self.fy * 0.15 / z_b))
            z_box_mm = np.uint16(max(100, int(z_b * 1000 - 30)))
            cv2.rectangle(
                depth_raw,
                (max(0, u_b - w_box // 2), max(0, v_b - h_box // 2)),
                (min(self.width - 1, u_b + w_box // 2), min(self.height - 1, v_b + h_box // 2)),
                int(z_box_mm),
                -1
            )

        # 3. Quả cà chua
        u_t, v_t, z_t = project_point(tomato_pos)
        if u_t is not None and -50 <= u_t < self.width + 50 and -50 <= v_t < self.height + 50:
            rad = max(8, int(self.fx * 0.032 / z_t))
            z_tom_mm = np.uint16(max(80, int(z_t * 1000 - 35)))
            cv2.circle(depth_raw, (u_t, v_t), rad, int(z_tom_mm), -1)

        # 4. Ngón tay kẹp (Gripper) - sát thấu kính nhất (~85mm)
        finger_w = 45
        finger_h = 120
        offset_x = 100 if gripper_closed else 150
        z_grip_mm = np.uint16(85)
        cv2.rectangle(depth_raw, (int(self.cx - offset_x - finger_w), self.height - finger_h),
                      (int(self.cx - offset_x), self.height), int(z_grip_mm), -1)
        cv2.rectangle(depth_raw, (int(self.cx + offset_x), self.height - finger_h),
                      (int(self.cx + offset_x + finger_w), self.height), int(z_grip_mm), -1)

        # 5. Colorizer: Chuyển depth mm thành RGB colormap (TURBO)
        d_min, d_max = 80.0, 750.0
        depth_clipped = np.clip(depth_raw.astype(np.float32), d_min, d_max)
        depth_norm = ((depth_clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        depth_colored_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        depth_colored_rgb = cv2.cvtColor(depth_colored_bgr, cv2.COLOR_BGR2RGB)

        return depth_colored_rgb, depth_raw

    def add_osd_preview(self, img_rgb: np.ndarray, gripper_closed: bool = False) -> np.ndarray:
        """
        Trả về khung hình video stream thuần túy sạch 100% (không chèn text hay trạng thái ngàm).
        """
        return img_rgb.copy()

