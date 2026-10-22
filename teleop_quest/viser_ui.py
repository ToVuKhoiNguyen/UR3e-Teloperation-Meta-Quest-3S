import os
import time
import subprocess
import numpy as np
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
    Wrapper độc lập điều khiển Target Gizmo tại vị trí TCP cho giải nghịch học IK.
    Tay cầm VR 3D đã được tách thành node 3D HUD riêng biệt, không còn bám theo TCP.
    """
    def __init__(self, gizmo, ctrl_frame=None, offset_dist=0.085, offset_right_dist=0.075, r_align=None, get_arm_theta=None):
        self._gizmo = gizmo

    @property
    def position(self):
        return self._gizmo.position

    @position.setter
    def position(self, val):
        self._gizmo.position = val

    @property
    def wxyz(self):
        return self._gizmo.wxyz

    @wxyz.setter
    def wxyz(self, val):
        self._gizmo.wxyz = val

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

        # 1. Tạo Gizmo transform controls độc lập tại root level cho bộ giải IK
        self._raw_gizmo = self.server.scene.add_transform_controls(
            "target_pose",
            depth_test=False,
            position=pos,
            wxyz=quat,
            scale=0.18,
            disable_sliders=False,
            visible=True
        )
        self.server.scene.world_axes.visible = False

        # 2. Target widget wrapper điều khiển Gizmo (cách ly 100% với 3D HUD)
        self.target_widget = TargetWidgetWrapper(self._raw_gizmo)

        # 3. Tạo Node 3D HUD (Camera Space Viewport HUD cố định góc màn hình)
        self._hud_mode = "viewport"
        self._active_camera = None
        self._hud_base_pos = np.array([0.45, 0.20, 0.14], dtype=float)

        # Ma trận xoay upright dựng đứng thẳng đối diện người dùng trong Camera Space (Pitch 0 deg, mặt nút bấm hướng ra ngoài)
        self._r_hud_upright = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0]
        ], dtype=float)
        self._q_hud_upright = (0.7071068, -0.7071068, 0.0, 0.0)
        self._hud_base_wxyz = self._q_hud_upright
        self._init_ctrl_wxyz = self._hud_base_wxyz

        self._hud_frame = self.server.scene.add_frame(
            "vr_controller_hud",
            position=tuple(self._hud_base_pos),
            wxyz=self._hud_base_wxyz,
            show_axes=False
        )
        self._ctrl_frame = self._hud_frame  # Giữ alias cho backward-compatibility

        self._client_cameras = {}

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            self._client_cameras[client.client_id] = client.camera
            self._active_camera = client.camera
            self._sync_hud_to_camera(client.camera)
            @client.camera.on_update
            def _(cam):
                if self._hud_mode == "viewport":
                    self._active_camera = cam
                    self._sync_hud_to_camera(cam)

        # 4. Nạp mô hình 3D tay cầm Meta Quest 3/3S gắn vào vr_controller_hud:
        # - Neo độc lập tại HUD góc màn hình / góc bàn máy
        # - Độ trong suốt 60% (baseColorFactor alpha = 0.60, alphaMode = BLEND)
        # - 8 biến thể màu sắc phản hồi trực quan phím bấm:
        #   + idle: trắng/xám (trạng thái nghỉ)
        #   + translate: xanh lá (bóp Cò)
        #   + rotate: cam/vàng (bóp Grip)
        #   + both: xanh dương ngọc (giữ cả Cò + Grip)
        #   + gripper: tím hồng (bấm nút A hút/kẹp)
        #   + btn_b: vàng kim (nút B/Y)
        #   + joystick: tím violet (analog tilt)
        #   + home: xanh dương đậm (lệnh Home)
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
                        f"vr_controller_hud/controller_{mode}",
                        glb_data=glb_bytes,
                        position=(0.0, 0.0, 0.0),
                        wxyz=(1.0, 0.0, 0.0, 0.0),
                        scale=(-1.0, 1.0, 1.0),
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
            div:has(> div > #teleop-2x2-anchor) .mantine-Slider-root {
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

            # 3. Thanh trượt tinh chỉnh tỉ lệ di chuyển (Scale) theo thời gian thực
            self.slider_sensitivity = self.server.gui.add_slider(
                "Scale",
                min=0.25,
                max=3.0,
                step=0.05,
                initial_value=1.0
            )
            self.server.gui.add_divider()

            # 4. Nút khôi phục an toàn (Full width bên dưới Scale)
            self.btn_home = self.server.gui.add_button("Go to Home")



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

    def _calc_hud_pose_from_cam(self, cam_pos, cam_wxyz):
        """Tính toán neo vị trí và hướng quay của 3D HUD theo Camera Space"""
        try:
            r_cam = transforms3d.quaternions.quat2mat(cam_wxyz)
            # OpenCV camera convention: +X right, +Y down, +Z fwd
            u_right = r_cam[:, 0]
            u_down = r_cam[:, 1]
            u_fwd = r_cam[:, 2]

            d_fwd = 0.45
            s_right = 0.20
            s_down = 0.14
            self._hud_base_pos = np.array(cam_pos) + d_fwd * u_fwd + s_right * u_right + s_down * u_down
            r_base = r_cam @ self._r_hud_upright
            self._hud_base_wxyz = tuple(transforms3d.quaternions.mat2quat(r_base))
            self._r_cam_current = r_cam
        except Exception:
            pass

    def _sync_hud_to_camera(self, cam):
        """Tính toán neo 3D HUD cố định tại góc dưới bên phải màn hình theo camera hiện tại"""
        try:
            self._calc_hud_pose_from_cam(cam.position, cam.wxyz)
            if not hasattr(self, "_current_ctrl_mode") or self._current_ctrl_mode == "idle":
                self._hud_frame.position = tuple(self._hud_base_pos)
                self._hud_frame.wxyz = self._hud_base_wxyz
        except Exception:
            pass

    def update_controller_visual(
        self,
        is_translating: bool = False,
        is_rotating: bool = False,
        is_gripper_btn: bool = False,
        is_btn_b: bool = False,
        is_joystick: bool = False,
        is_home: bool = False,
        delta_trans: np.ndarray = None,
        vr_quat: list = None
    ):
        """
        Cập nhật hiển thị toàn diện của 3D HUD:
          1. Nút bấm: Đổi màu sắc model GLB theo 8 biến thể.
          2. Hướng xoay: Đồng bộ góc xoay cổ tay qua vr_quat CHỈ KHI BÓP CÒ / GRIP.
          3. Dịch chuyển tịnh tiến: Nhích vi mô (Micro-translation) CHỈ KHI BÓP CÒ TIẾN/LÙI.
          4. Khi nhả cò: Đàn hồi giật về vị trí Base Position và góc đứng thẳng ban đầu.
        """
        if hasattr(self, "cb_show_controller") and not self.cb_show_controller.value:
            for h in self._ctrl_handles.values():
                h.visible = False
            return

        # 1. Quản lý biến thể màu sắc nút bấm
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

        # Luôn đảm bảo đồng bộ góc nhìn camera client mới nhất nếu có
        if self._hud_mode == "viewport":
            if self._active_camera is not None:
                self._calc_hud_pose_from_cam(self._active_camera.position, self._active_camera.wxyz)
            else:
                clients = self.server.get_clients()
                if clients:
                    c = next(iter(clients.values()))
                    self._active_camera = c.camera
                    self._calc_hud_pose_from_cam(c.camera.position, c.camera.wxyz)

        # 2. Quản lý góc xoay tay cầm (Chỉ xoay khi bóp cò / grip; khi nhả cò thì đứng thẳng hướng mặt về màn hình)
        is_active = is_translating or is_rotating
        if is_active and vr_quat is not None and len(vr_quat) == 4:
            try:
                q = np.array(vr_quat, dtype=float)
                norm_q = np.linalg.norm(q)
                if norm_q > 1e-4:
                    q /= norm_q
                    r_wrist = transforms3d.quaternions.quat2mat(q)
                    if hasattr(self, "_r_cam_current"):
                        r_hud = self._r_cam_current @ self._r_hud_upright @ r_wrist
                    elif hasattr(self, "_hud_base_wxyz"):
                        r_base = transforms3d.quaternions.quat2mat(self._hud_base_wxyz)
                        r_hud = r_base @ r_wrist
                    else:
                        r_hud = r_wrist
                    self._hud_frame.wxyz = tuple(transforms3d.quaternions.mat2quat(r_hud))
            except Exception:
                if hasattr(self, "_hud_base_wxyz"):
                    self._hud_frame.wxyz = self._hud_base_wxyz
        elif hasattr(self, "_hud_base_wxyz"):
            # Khi nhả cò: Giật đàn hồi ngay lập tức về tư thế đứng thẳng tĩnh ban đầu
            self._hud_frame.wxyz = self._hud_base_wxyz

        # 3. Quản lý phản hồi tịnh tiến (Độ đàn hồi vi mô - Micro-translation)
        # - KHI BÓP CÒ TIẾN/LÙI/TRÁI/PHẢI/LÊN/XUỐNG: Nhích vi mô tối đa đúng 2.0cm
        # - KHI NHẢ CÒ: Đàn hồi lập tức giật về vị trí gốc Base Position
        if hasattr(self, "_hud_frame") and hasattr(self, "_hud_base_pos"):
            if is_translating and delta_trans is not None and len(delta_trans) == 3:
                d_fwd = float(delta_trans[0])
                d_lat = float(delta_trans[1])
                d_vert = float(delta_trans[2])
                alpha_micro = 0.15
                if hasattr(self, "_r_cam_current"):
                    u_right = self._r_cam_current[:, 0]
                    u_down = self._r_cam_current[:, 1]
                    u_fwd = self._r_cam_current[:, 2]
                    # Phản hồi theo hệ trục Camera: Tiến (+fwd), Sang Phải (+right), Lên Trên (-down)
                    disp = (d_lat * u_right - d_vert * u_down + d_fwd * u_fwd) * alpha_micro
                else:
                    disp = np.array([d_lat, -d_vert, d_fwd], dtype=float) * alpha_micro

                disp_norm = np.linalg.norm(disp)
                max_disp = 0.020  # Khống chế dịch chuyển tối đa đúng 2.0cm
                if disp_norm > max_disp:
                    disp = disp * (max_disp / disp_norm)
                self._hud_frame.position = tuple(self._hud_base_pos + disp)
            else:
                # Trạng thái tĩnh: Neo cố định tại Base Position
                self._hud_frame.position = tuple(self._hud_base_pos)

