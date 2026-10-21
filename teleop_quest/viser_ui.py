import os
import time
import subprocess
import cv2
import numpy as np
import torch
import transforms3d
import viser
from viser.extras import ViserUrdf
import yourdfpy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

def _urdf_filename_handler(fname: str) -> str:
    base_name = os.path.basename(fname)
    direct_asset = os.path.join(BASE_DIR, "assets", base_name)
    if os.path.exists(direct_asset):
        return direct_asset

    if fname.startswith("package://ur_description/"):
        rel_path = fname[len("package://ur_description/"):]
        candidates = [
            os.path.join(BASE_DIR, "assets", "ur_description", rel_path),
            os.path.join(BASE_DIR, "assets", rel_path),
            os.path.join(os.path.dirname(BASE_DIR), "src", "Universal_Robots_ROS2_Description", rel_path),
        ]

        for cand in candidates:
            if os.path.exists(cand):
                return cand
    return fname


class StatusProxy:
    """Proxy hỗ trợ truy cập .value hai chiều tương thích với Viser GUI text input cũ"""
    def __init__(self, refresh_cb, initial_value=""):
        self._val = str(initial_value)
        self._refresh_cb = refresh_cb
        self._callbacks = []

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, new_val):
        new_val_str = str(new_val)
        if self._val == new_val_str:
            return  # Chống spam cập nhật Markdown WebSocket khi trạng thái không đổi
        self._val = new_val_str
        if self._refresh_cb:
            self._refresh_cb()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception as e:
                print(f"[StatusProxy Cảnh báo] Callback lỗi: {e}")

    def on_update(self, callback):
        """Hỗ trợ đăng ký callback event handler như một Viser GUI handle thực thụ"""
        self._callbacks.append(callback)
        return callback

    def __str__(self):
        return self._val



class TargetWidgetWrapper:
    """
    Wrapper đồng bộ tọa độ giữa Gizmo và Mô hình tay cầm VR:
      - Vị trí: Tay cầm đặt phía trước TCP (offset 8.5cm theo phương cánh tay/chuyển động), không bị trùng sát vào đầu tay máy.
      - Góc quay: Đồng bộ 1:1 với Gizmo (R_ctrl = R_gizmo * R_align), tại Home đứng thẳng tự nhiên và chĩa mũi theo đúng hướng di chuyển.
    """
    def __init__(self, gizmo, ctrl_frame, offset_dist=0.085, offset_right_dist=0.075, r_align=None, get_arm_theta=None):
        self._gizmo = gizmo
        self._ctrl_frame = ctrl_frame
        self._offset_dist = float(offset_dist)
        self._offset_right_dist = float(offset_right_dist)
        self._r_align = r_align
        self._get_arm_theta = get_arm_theta

        @self._gizmo.on_update
        def _(_):
            self._update_ctrl_pose(self._gizmo.position, self._gizmo.wxyz)

    def _calc_arm_vectors(self, pos):
        theta = None
        if self._get_arm_theta is not None:
            try:
                theta = self._get_arm_theta()
            except Exception:
                pass
        if theta is None:
            x, y = pos[0], pos[1]
            r = np.hypot(x, y)
            theta = np.arctan2(y, x) if r > 0.03 else -np.pi / 2

        c, s = np.cos(theta), np.sin(theta)
        v_fwd = np.array([c, s, 0.0], dtype=float)
        v_right = np.array([s, -c, 0.0], dtype=float)
        return v_fwd, v_right

    def _update_ctrl_pose(self, pos, wxyz):
        v_fwd, v_right = self._calc_arm_vectors(pos)
        ctrl_pos = np.array(pos, dtype=float) + self._offset_dist * v_fwd + self._offset_right_dist * v_right
        self._ctrl_frame.position = tuple(ctrl_pos)
        self._sync_orientation(wxyz)

    def _sync_orientation(self, gizmo_wxyz):
        if self._r_align is not None:
            try:
                r_gizmo = transforms3d.quaternions.quat2mat(gizmo_wxyz)
                r_ctrl = r_gizmo @ self._r_align
                q_ctrl = transforms3d.quaternions.mat2quat(r_ctrl)
                self._ctrl_frame.wxyz = tuple(q_ctrl)
            except Exception:
                pass

    @property
    def position(self):
        return self._gizmo.position

    @position.setter
    def position(self, val):
        self._gizmo.position = val
        self._update_ctrl_pose(val, self._gizmo.wxyz)

    @property
    def wxyz(self):
        return self._gizmo.wxyz

    @wxyz.setter
    def wxyz(self, val):
        self._gizmo.wxyz = val
        self._sync_orientation(val)

    @property
    def visible(self):
        return self._gizmo.visible

    @visible.setter
    def visible(self, val):
        self._gizmo.visible = val

    def __getattr__(self, name):
        return getattr(self._gizmo, name)





class TeleopViserUI:
    """
    Quản lý giao diện Viser 3D:
      - Mô hình robot ảo & đồng bộ trạng thái khớp
      - Trục tọa độ 3D Transform Controls (Target Widget)
      - Bảng điều khiển nút bấm GUI (Teleop, Record, Discard, Home, Status)
    """
    def __init__(
        self,
        init_pos: list,
        init_quat: list,
        init_joints: list,
        init_tomato_pos: tuple = (0.15, 0.35),
        port: int = 8080
    ):
        # Dọn dẹp port cũ nếu có
        subprocess.run(f"fuser -k {port}/tcp 2>/dev/null", shell=True)
        time.sleep(0.5)

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        urdf_path = os.path.join(BASE_DIR, "assets", "ur3e_custom.urdf")
        self.urdf_model = yourdfpy.URDF.load(urdf_path, filename_handler=_urdf_filename_handler)
        self.viser_urdf = ViserUrdf(self.server, self.urdf_model, root_node_name="/robot")
        self.viz = self.viser_urdf

        # Đồng bộ khớp ban đầu
        self.update_robot_state(init_joints)

        # Tạo Widget 3D tại tọa độ TCP ban đầu của robot (chuẩn PyRoki world)
        pos = init_pos[:3]
        if len(init_quat) == 4:
            quat = init_quat
        else:
            axis = np.array(init_quat)
            angle = np.linalg.norm(axis)
            quat = transforms3d.quaternions.axangle2quat(axis / angle, angle) if angle > 1e-6 else [1.0, 0.0, 0.0, 0.0]

        # Lấy góc quay của khớp đế ban đầu q0
        init_q0 = float(init_joints[0]) if (init_joints is not None and len(init_joints) > 0) else 0.0
        self._current_q0 = init_q0

        # Tính toán ma trận căn chỉnh R_align để tay cầm đồng bộ 1:1 với Gizmo:
        # Tại tư thế Home, tay cầm đứng thẳng tự nhiên (trục cán hướng xuống -Z)
        # và mũi tay cầm chĩa theo đúng phương vươn của cánh tay robot (v_fwd = [cos(q0), sin(q0), 0]):
        R_home = transforms3d.quaternions.quat2mat(quat)
        c0, s0 = np.cos(init_q0), np.sin(init_q0)
        R_upright = np.array([
            [-s0,   c0,  0.0],
            [ c0,   s0,  0.0],
            [ 0.0,  0.0, -1.0]
        ], dtype=float)
        self._r_align = R_home.T @ R_upright
        init_ctrl_wxyz = tuple(transforms3d.quaternions.mat2quat(R_upright))

        # Vector hướng vươn ban đầu và hướng bên phải của cánh tay robot
        v_fwd_init = np.array([c0, s0, 0.0], dtype=float)
        v_right_init = np.array([s0, -c0, 0.0], dtype=float)
        self._ctrl_offset_dist = 0.085       # Khoảng cách an toàn 8.5cm phía trước TCP
        self._ctrl_offset_right = 0.075      # Dịch sang phải 7.5cm để quan sát song song cùng Gizmo

        # 1. Tạo Gizmo transform controls độc lập tại root level (mặc định hiển thị đồng thời với tay cầm)
        self._raw_gizmo = self.server.scene.add_transform_controls(
            "target_pose",
            depth_test=False,
            position=pos,
            wxyz=quat,
            scale=0.18,
            disable_sliders=False,
            visible=True
        )

        # 2. Tạo Frame điều khiển tay cầm VR độc lập tại root level (đặt trước đầu tay máy 8.5cm, lệch phải 7.5cm):
        ctrl_init_pos = tuple(np.array(pos) + self._ctrl_offset_dist * v_fwd_init + self._ctrl_offset_right * v_right_init)
        self._ctrl_frame = self.server.scene.add_frame(
            "vr_controller_frame",
            position=ctrl_init_pos,
            wxyz=init_ctrl_wxyz,
            show_axes=False
        )

        # 3. Target widget proxy đồng bộ tọa độ và góc xoay 1:1 cho main_quest và các module khác
        self.target_widget = TargetWidgetWrapper(
            self._raw_gizmo,
            self._ctrl_frame,
            offset_dist=self._ctrl_offset_dist,
            offset_right_dist=self._ctrl_offset_right,
            r_align=self._r_align,
            get_arm_theta=lambda: self._current_q0
        )

        # 4. Nạp mô hình 3D tay cầm Meta Quest 3/3S gắn vào vr_controller_frame:
        # - Trùng vị trí Gizmo, xoay bám sát cử động cổ tay thực tế của người dùng từ kính VR
        # - Độ trong suốt 60% (baseColorFactor alpha = 0.60, alphaMode = BLEND)
        # - 5 biến thể màu sắc phản hồi trực quan phím bấm:
        #   + idle: trắng/xám (trạng thái nghỉ)
        #   + translate: xanh lá (bóp Cò)
        #   + rotate: cam/vàng (bóp Grip)
        #   + both: xanh dương ngọc (giữ cả Cò + Grip)
        #   + gripper: tím hồng (bấm nút A hút/kẹp)
        self._ctrl_handles = {}
        self._current_ctrl_mode = "idle"
        ctrl_variants = {
            "idle": os.path.join(BASE_DIR, "assets", "quest3_ctrl_idle.glb"),
            "translate": os.path.join(BASE_DIR, "assets", "quest3_ctrl_translate.glb"),
            "rotate": os.path.join(BASE_DIR, "assets", "quest3_ctrl_rotate.glb"),
            "both": os.path.join(BASE_DIR, "assets", "quest3_ctrl_both.glb"),
            "gripper": os.path.join(BASE_DIR, "assets", "quest3_ctrl_gripper.glb"),
            "btn_b": os.path.join(BASE_DIR, "assets", "quest3_ctrl_btn_b.glb"),
            "joystick": os.path.join(BASE_DIR, "assets", "quest3_ctrl_joystick.glb"),
            "home": os.path.join(BASE_DIR, "assets", "quest3_ctrl_home.glb"),
        }
        fallback_glb = os.path.join(BASE_DIR, "assets", "quest3_controller_right.glb")
        for mode, path in ctrl_variants.items():
            target_file = path if os.path.exists(path) else fallback_glb
            if os.path.exists(target_file):
                try:
                    with open(target_file, "rb") as f:
                        glb_bytes = f.read()
                    self._ctrl_handles[mode] = self.server.scene.add_glb(
                        f"vr_controller_frame/controller_{mode}",
                        glb_data=glb_bytes,
                        position=(0.0, 0.0, 0.0),
                        wxyz=(1.0, 0.0, 0.0, 0.0),
                        scale=1.0,
                        visible=(mode == "idle")
                    )
                except Exception as e:
                    print(f"[Viser UI] Cảnh báo không thể nạp mô hình tay cầm {mode}: {e}")

        # Trạng thái hệ thống hiển thị dạng Dashboard Markdown compact
        self._val_status = "Waiting..."
        self._val_conn = "No"
        self._val_episodes = "0"
        self._val_grasp = "Empty"

        self.ui_status = StatusProxy(self._on_status_change, "Waiting...")
        self.ui_conn = StatusProxy(self._on_conn_change, "No")
        self.ui_episodes = StatusProxy(self._on_episodes_change, "0")
        self.ui_grasp_state = StatusProxy(self._on_grasp_change, "Empty")

        # Bảng điều khiển GUI tinh gọn
        with self.server.gui.add_folder("Teleoperation"):
            # 1. Bảng trạng thái gọn gàng (Thay thế hoàn toàn 4 ô input xám)
            self._md_status = self.server.gui.add_markdown(self._build_status_md())
            self.server.gui.add_divider()

            # 2. Cụm công tắc 2x2 (Hàng 1: Teleop & Record AI | Hàng 2: Gripper & Discard)
            self.server.gui.add_html("""
            <div id="teleop-2x2-anchor" style="display:none;"></div>
            <style>
            div:has(> div > #teleop-2x2-anchor) {
                display: flex !important;
                flex-wrap: wrap !important;
                row-gap: 4px !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:has(#teleop-2x2-anchor) {
                display: none !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(1),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(2),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(8),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(9),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(10),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(11) {
                width: 100% !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(4),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(5),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(6),
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(7) {
                width: 50% !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(4) > div,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(5) > div,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(6) > div,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(7) > div {
                width: 100% !important;
                padding-bottom: 0 !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(4) .mantine-Flex-root,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(5) .mantine-Flex-root,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(6) .mantine-Flex-root,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(7) .mantine-Flex-root {
                justify-content: space-between !important;
                width: 100% !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(4) .mantine-Flex-root > div:first-child,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(5) .mantine-Flex-root > div:first-child,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(6) .mantine-Flex-root > div:first-child,
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(7) .mantine-Flex-root > div:first-child {
                width: auto !important;
                padding-right: 4px !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(7) button {
                height: 28px !important;
                padding: 0 4px !important;
                font-size: 0.85em !important;
            }
            div:has(> div > #teleop-2x2-anchor) > div:nth-child(11) .mantine-Slider-root {
                flex-grow: 1 !important;
                min-width: 80px !important;
            }
            </style>
            """)

            # Hàng 1: Teleop & Record AI
            self.cb_teleop = self.server.gui.add_checkbox("Bật Teleop", initial_value=False)
            self.cb_record = self.server.gui.add_checkbox("Record AI", initial_value=False)

            # Hàng 2: Gripper & Discard
            self.cb_gripper = self.server.gui.add_checkbox("Gripper", initial_value=False)
            self.cb_discard = self.server.gui.add_checkbox("Discard", initial_value=False)

            self.server.gui.add_divider()

            # 3. Nút khôi phục an toàn (Full width bên dưới cụm 2x2)
            self.btn_home = self.server.gui.add_button("Go to Home")
            self.server.gui.add_divider()

            # 4. Thanh trượt tinh chỉnh độ nhạy di chuyển (Scale) theo thời gian thực
            self.slider_sensitivity = self.server.gui.add_slider(
                "Độ nhạy (Scale)",
                min=0.25,
                max=3.0,
                step=0.05,
                initial_value=1.0
            )

        # Thư mục điều khiển hiển thị 3D: Gizmo và Tay Cầm trên cùng một hàng ngang
        with self.server.gui.add_folder("Hiển thị 3D", expand_by_default=True):
            self.server.gui.add_html("""
            <div id="vis-toggle-anchor" style="display:none;"></div>
            <style>
            div:has(> div > #vis-toggle-anchor) {
                display: flex !important;
                flex-wrap: wrap !important;
                row-gap: 4px !important;
            }
            div:has(> div > #vis-toggle-anchor) > div:has(#vis-toggle-anchor) {
                display: none !important;
            }
            div:has(> div > #vis-toggle-anchor) > div:nth-child(2),
            div:has(> div > #vis-toggle-anchor) > div:nth-child(3) {
                width: 50% !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
            }
            div:has(> div > #vis-toggle-anchor) > div:nth-child(2) > div,
            div:has(> div > #vis-toggle-anchor) > div:nth-child(3) > div {
                width: 100% !important;
                padding-bottom: 0 !important;
            }
            div:has(> div > #vis-toggle-anchor) .mantine-Flex-root {
                justify-content: space-between !important;
                width: 100% !important;
            }
            </style>
            """)
            self.cb_show_gizmo = self.server.gui.add_checkbox("Gizmo", initial_value=True)
            self.cb_show_controller = self.server.gui.add_checkbox("Tay Cầm", initial_value=True)

            @self.cb_show_gizmo.on_update
            def _(_):
                self._raw_gizmo.visible = self.cb_show_gizmo.value

            @self.cb_show_controller.on_update
            def _(_):
                self.update_controller_visual()


    def get_camera_view_mode(self) -> str:
        """Lấy chế độ xem camera (Mặc định 'rgb' do đã gỡ bỏ UI điều khiển trên Viser)"""
        return "rgb"

    def _on_status_change(self):
        self._val_status = self.ui_status.value
        self._refresh_status_md()

    def _on_conn_change(self):
        self._val_conn = self.ui_conn.value
        self._refresh_status_md()

    def _on_episodes_change(self):
        self._val_episodes = self.ui_episodes.value
        self._refresh_status_md()

    def _on_grasp_change(self):
        self._val_grasp = self.ui_grasp_state.value
        self._refresh_status_md()

    def _build_status_md(self) -> str:
        return (
            f"**Status:** `{self._val_status}` &nbsp;|&nbsp; **VR:** `{self._val_conn}`  \n"
            f"**Grasp:** `{self._val_grasp}` &nbsp;|&nbsp; **Episodes:** `{self._val_episodes}`"
        )

    def _refresh_status_md(self):
        self._md_status.content = self._build_status_md()

    def update_robot_state(self, q: list):
        """Cập nhật dáng robot thực tế lên Viser 3D (CPU-only)"""
        if q is not None and len(q) > 0:
            self._current_q0 = float(q[0])
            self.viser_urdf.update_cfg(np.array(q, dtype=float))

    def update_cam_preview(self, frame_rgb: np.ndarray):
        """Cập nhật ảnh xem trực tiếp camera cổ tay (Đã vô hiệu hóa giao diện Wrist Camera trên Viser)"""
        pass

    def set_status(self, text: str):
        self.ui_status.value = text

    def set_connected(self, is_connected: bool):
        self.ui_conn.value = "Yes" if is_connected else "No"

    def set_episodes(self, count: int):
        self.ui_episodes.value = str(count)

    def set_grasp_state(self, is_grasped: bool):
        self.ui_grasp_state.value = "Holding" if is_grasped else "Empty"

    def update_controller_visual(
        self,
        is_translating: bool = False,
        is_rotating: bool = False,
        is_gripper_btn: bool = False,
        is_btn_b: bool = False,
        is_joystick: bool = False,
        is_home: bool = False
    ):
        """
        Cập nhật màu sắc phím bấm của mô hình tay cầm VR theo thời gian thực (8 trạng thái màu độc lập):
          - Bóp Cò (Trigger): Sáng xanh lá (Tịnh tiến).
          - Bóp Grip: Sáng cam (Xoay 3 trục).
          - Giữ cả Cò + Grip: Sáng xanh dương ngọc (Phối hợp 6-DOF).
          - Bấm nút A (Gripper Toggle): Sáng tím hồng (Đóng/mở hút kẹp).
          - Bấm nút B / Y: Sáng vàng kim (Đổi Drive Mode).
          - Gạt cần Joystick: Sáng tím Violet (Điều hướng / Analog tilt).
          - Nhấn Joystick / Về Home: Sáng xanh dương đậm Royal Blue (Lệnh Go Home).
          - Trạng thái nghỉ: Xám trắng tự nhiên (60% độ trong suốt).
        """
        if not hasattr(self, "cb_show_controller") or not self.cb_show_controller.value:
            for h in self._ctrl_handles.values():
                h.visible = False
            return

        if is_home:
            target_mode = "home"
        elif is_joystick:
            target_mode = "joystick"
        elif is_btn_b:
            target_mode = "btn_b"
        elif is_gripper_btn:
            target_mode = "gripper"
        elif is_translating and is_rotating:
            target_mode = "both"
        elif is_translating:
            target_mode = "translate"
        elif is_rotating:
            target_mode = "rotate"
        else:
            target_mode = "idle"

        if target_mode != self._current_ctrl_mode or not any(h.visible for h in self._ctrl_handles.values()):
            self._current_ctrl_mode = target_mode
            for mode, h in self._ctrl_handles.items():
                h.visible = (mode == target_mode)

