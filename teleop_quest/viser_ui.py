"""
viser_ui.py - 3D Visualizer and Teleoperation GUI for UR3e using Viser.
Runs entirely on CPU with zero GPU / NVIDIA CUDA dependencies.
"""

import os
import time
import subprocess
from typing import Optional, Dict, Any, List
import numpy as np
import transforms3d
import viser
from viser.extras import ViserUrdf
import yourdfpy
import trimesh

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
    """Proxy supporting bidirectional .value access compatible with Viser GUI."""
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
            return  # Prevent redundant Markdown WebSocket updates
        self._val = new_val_str
        if self._refresh_cb:
            self._refresh_cb()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception as e:
                print(f"[StatusProxy Warning] Callback error: {e}")

    def on_update(self, callback):
        """Register update event callback."""
        self._callbacks.append(callback)
        return callback

    def __str__(self):
        return self._val


class _DummyGuiHandle:
    """Backward-compatible proxy for internal boolean state variables."""
    def __init__(self, initial_value: bool = False):
        self._val = bool(initial_value)
        self._callbacks = []

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, new_val):
        self._val = bool(new_val)
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def on_update(self, callback):
        self._callbacks.append(callback)
        return callback

    def __bool__(self):
        return bool(self._val)


class TargetWidgetWrapper:
    """
    Independent wrapper for Target Gizmo at TCP for IK solving.
    The 3D VR controller HUD is separated into an independent camera-space viewport node.
    Provides:
      - Deadband filtering for position (Delta p > 1e-4 m) and orientation (Delta theta > 1e-4 rad)
      - Quaternion hemisphere sign continuity (ensures dot(new, curr) >= 0 to avoid 360-deg visual flips)
      - Programmatic write suppression flag to prevent feedback loops with mouse drag callbacks
    """
    def __init__(self, gizmo, ctrl_frame=None, offset_dist=0.085, offset_right_dist=0.075, r_align=None, get_arm_theta=None):
        self._gizmo = gizmo
        self._suppress_callback = False
        self._last_programmatic_time = 0.0
        self._suppress_duration = 0.12  # 120ms window for WebSocket client echo absorption
        self._pos_epsilon = 1e-4  # 0.1 mm
        self._rot_epsilon = 1e-4  # 0.0001 rad

    @property
    def is_suppressed(self) -> bool:
        return self._suppress_callback

    @is_suppressed.setter
    def is_suppressed(self, val: bool):
        self._suppress_callback = bool(val)

    def is_suppression_active(self) -> bool:
        """Returns True if explicitly suppressed or within echo absorption window."""
        return self._suppress_callback or ((time.time() - self._last_programmatic_time) < self._suppress_duration)

    @property
    def position(self):
        return self._gizmo.position

    @position.setter
    def position(self, val):
        self.update_pose(pos=val, suppress=True)

    @property
    def wxyz(self):
        return self._gizmo.wxyz

    @wxyz.setter
    def wxyz(self, val):
        self.update_pose(wxyz=val, suppress=True)

    def update_pose(self, pos=None, wxyz=None, suppress=True):
        """
        Updates gizmo position and/or orientation with:
          1. Deadband filtering (Delta p > 1e-4 m, Delta theta > 1e-4 rad)
          2. Quaternion hemisphere sign continuity (dot(new, curr) >= 0)
          3. Callback echo suppression
        """
        if suppress:
            self._last_programmatic_time = time.time()
            self._suppress_callback = True

        try:
            if pos is not None:
                new_pos = np.asarray(pos, dtype=float)[:3]
                try:
                    curr_pos = np.asarray(self._gizmo.position, dtype=float)[:3]
                    dist = float(np.linalg.norm(new_pos - curr_pos))
                except Exception:
                    dist = 1.0

                if dist > self._pos_epsilon:
                    self._gizmo.position = tuple(float(x) for x in new_pos)

            if wxyz is not None:
                new_w = np.asarray(wxyz, dtype=float)[:4]
                norm_w = float(np.linalg.norm(new_w))
                if norm_w > 1e-9:
                    new_w = new_w / norm_w
                else:
                    new_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

                try:
                    curr_w = np.asarray(self._gizmo.wxyz, dtype=float)[:4]
                    curr_norm = float(np.linalg.norm(curr_w))
                    if curr_norm > 1e-9:
                        curr_w = curr_w / curr_norm
                    dot = float(np.dot(new_w, curr_w))
                except Exception:
                    dot = 1.0

                if dot < 0.0:
                    new_w = -new_w
                    dot = -dot

                # Angular deadband check: delta_theta = 2 * arccos(min(1.0, dot))
                delta_theta = 2.0 * np.arccos(min(1.0, max(-1.0, dot)))
                if delta_theta > self._rot_epsilon:
                    self._gizmo.wxyz = tuple(float(x) for x in new_w)

        finally:
            if suppress:
                self._suppress_callback = False

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
    Manages Viser 3D interface:
      - Virtual robot model and joint state synchronization
      - 3D Transform Controls (Target Gizmo)
      - GUI control panel (Teleop, Record, Discard, Home, Telemetry Status)
      - 0.5cm (5mm) Inflated Mesh Armor Visualization (WYSIWYG)
    """
    def __init__(
        self,
        init_pos: list,
        init_quat: list,
        init_joints: list,
        init_tomato_pos: tuple = (0.15, 0.35),
        port: int = 8080
    ):
        # Clean up previous port binding if necessary
        subprocess.run(f"fuser -k {port}/tcp 2>/dev/null", shell=True)
        time.sleep(0.5)

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        # Optimal camera perspective for UR3e workspace
        self.server.initial_camera.position = (0.8, -0.6, 0.7)
        self.server.initial_camera.look_at = (0.15, 0.1, 0.15)
        urdf_path = os.path.join(BASE_DIR, "assets", "ur3e_custom.urdf")
        self.urdf_model = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=_urdf_filename_handler,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        self.viser_urdf = ViserUrdf(
            self.server,
            self.urdf_model,
            root_node_name="/robot",
            load_collision_meshes=False,
        )
        self.viz = self.viser_urdf

        # Uniform +1.0cm (10mm) Red Armor Envelope (True 10mm Inflated Mesh Envelope)
        self._robot_armor_handles = []
        link_to_frame = {}
        for f in self.viser_urdf._joint_frames:
            link_name = f.name.split("/")[-1]
            link_to_frame[link_name] = f

        armor_mesh_map = {
            "base_link_inertia": "assets/ur3e_inflated_1cm/base_inflated.stl",
            "shoulder_link": "assets/ur3e_inflated_1cm/shoulder_inflated.stl",
            "upper_arm_link": "assets/ur3e_inflated_1cm/upperarm_inflated.stl",
            "forearm_link": "assets/ur3e_inflated_1cm/forearm_inflated.stl",
            "wrist_1_link": "assets/ur3e_inflated_1cm/wrist1_inflated.stl",
            "wrist_2_link": "assets/ur3e_inflated_1cm/wrist2_inflated.stl",
            "wrist_3_link": "assets/ur3e_inflated_1cm/wrist3_inflated.stl",
            "camera_mount_link": "assets/ur3e_inflated_1cm/mount_inflated.stl",
            "d455_link": "assets/ur3e_inflated_1cm/d455_inflated.stl",
        }

        for link_name, rel_path in armor_mesh_map.items():
            full_path = os.path.join(BASE_DIR, rel_path)
            if link_name in link_to_frame and os.path.exists(full_path):
                frame = link_to_frame[link_name]
                m = trimesh.load(full_path, force="mesh").copy()
                link = self.urdf_model.link_map[link_name]
                if link.visuals and link.visuals[0].origin is not None:
                    m.apply_transform(link.visuals[0].origin)
                handle = self.server.scene.add_mesh_simple(
                    name=f"{frame.name}/armor_mesh",
                    vertices=m.vertices,
                    faces=m.faces,
                    color=(255, 60, 60),
                    opacity=0.35,
                    visible=True,
                )
                self._robot_armor_handles.append(handle)
        print(f"[Viser UI] Successfully attached {len(self._robot_armor_handles)}/9 URDF +1.0cm inflated armor meshes.")

        # Parent frame for environmental collision barriers (WYSIWYG)
        self._collision_frame = self.server.scene.add_frame(
            "/collision_envelopes",
            show_axes=False,
            visible=True
        )
        self._collision_handles = {}

        # Obstacle configuration from obstacle_scene.yml
        scene_path = os.path.join(BASE_DIR, "assets", "obstacle_scene.yml")
        table_dims = (1.2, 1.2, 0.1)
        table_pos = (0.15, 0.0, -0.04)
        if os.path.exists(scene_path):
            try:
                import yaml
                with open(scene_path, "r") as f:
                    data = yaml.safe_load(f)
                t = data.get("cuboid", {}).get("table", {})
                if "dims" in t and "pose" in t:
                    table_dims = tuple(t["dims"])
                    table_pos = tuple(t["pose"][:3])
            except Exception as e:
                print(f"[Viser UI] Could not parse obstacle_scene.yml: {e}")

        # Table Armor Envelope: +1.0cm armor, top surface exactly at Z=+0.020m
        self._collision_handles["table_barrier"] = self.server.scene.add_box(
            "/collision_envelopes/table_barrier",
            dimensions=(table_dims[0] + 0.020, table_dims[1] + 0.020, table_dims[2] + 0.010),
            position=(table_pos[0], table_pos[1], table_pos[2] + 0.005),
            color=(255, 70, 70),
            opacity=0.20,
            visible=True,
        )

        # Workspace Bounding Box: XY ±0.52m, Z 0.015 - 0.60m
        self._collision_handles["workspace_bbox"] = self.server.scene.add_box(
            "/collision_envelopes/workspace_bbox",
            dimensions=(1.04, 1.04, 0.585),
            position=(0.0, 0.0, 0.3075),
            color=(0, 200, 255),
            wireframe=True,
            opacity=0.30,
            visible=False,
        )

        # Synchronize initial joint state
        self.update_robot_state(init_joints)

        # Target Gizmo at initial TCP position
        pos = init_pos[:3]
        if len(init_quat) == 4:
            quat = init_quat
        else:
            axis = np.array(init_quat)
            angle = np.linalg.norm(axis)
            quat = transforms3d.quaternions.axangle2quat(axis / angle, angle) if angle > 1e-6 else [1.0, 0.0, 0.0, 0.0]

        init_q0 = float(init_joints[0]) if (init_joints is not None and len(init_joints) > 0) else 0.0
        self._current_q0 = init_q0

        R_home = transforms3d.quaternions.quat2mat(quat)
        c0, s0 = np.cos(init_q0), np.sin(init_q0)
        R_upright = np.array([
            [-s0,   c0,  0.0],
            [ c0,   s0,  0.0],
            [ 0.0,  0.0, -1.0]
        ], dtype=float)
        self._r_align = R_home.T @ R_upright

        self._ctrl_offset_dist = 0.085
        self._ctrl_offset_right = 0.075

        # 1. Target Gizmo for IK solving
        self._raw_gizmo = self.server.scene.add_transform_controls(
            "target_pose",
            depth_test=False,
            position=pos,
            wxyz=quat,
            scale=0.18,
            disable_sliders=False,
            disable_rotations=False,
            visible=True
        )

        # 2. Target widget wrapper
        self.target_widget = TargetWidgetWrapper(self._raw_gizmo)

        # 3. 3D Controller HUD (Camera Space Viewport HUD)
        self._hud_mode = "viewport"
        self._active_camera = None

        self._r_hud_upright = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=float)

        self._hud_base_pos = np.array([0.25, -0.20, 0.35], dtype=float)
        self._hud_base_wxyz = (1.0, 0.0, 0.0, 0.0)

        self._hud_frame = self.server.scene.add_frame(
            "vr_controller_hud",
            show_axes=False,
            position=tuple(self._hud_base_pos),
            wxyz=self._hud_base_wxyz
        )
        self._ctrl_frame = self._hud_frame

        # Camera sync listener
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            self._active_camera = client.camera
            self._sync_hud_to_camera(client.camera)

            @client.camera.on_update
            def _(cam: viser.CameraHandle):
                self._sync_hud_to_camera(cam)

        # 4. 3D Controller GLB Models for button feedback
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
                    print(f"[Viser UI] Warning loading controller model {mode}: {e}")

        # Dashboard status state variables
        self._val_status = "OK"
        self._val_conn = "No"
        self._val_episodes = "0"
        self._val_grasp = "Empty"

        self.ui_status = StatusProxy(self._on_status_change, "OK")
        self.ui_conn = StatusProxy(self._on_conn_change, "No")
        self.ui_episodes = StatusProxy(self._on_episodes_change, "0")
        self.ui_grasp_state = StatusProxy(self._on_grasp_change, "Empty")

        # Teleoperation GUI folder
        with self.server.gui.add_folder("Teleoperation"):
            self._md_status = self.server.gui.add_markdown(self._build_status_md())
            self.server.gui.add_divider()

            self.server.gui.add_html("""
            <div id="teleop-horizontal-anchor" style="display:none;"></div>
            <style>
            div:has(> div > #teleop-horizontal-anchor) {
                display: flex !important;
                flex-wrap: wrap !important;
                row-gap: 4px !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:has(#teleop-horizontal-anchor) {
                display: none !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(1),
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(2) {
                width: 100% !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(4),
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(5) {
                width: 50% !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                padding-right: 4px !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(4) > div,
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(5) > div {
                width: 100% !important;
                padding-bottom: 0 !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(4) .mantine-Flex-root,
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(5) .mantine-Flex-root {
                justify-content: space-between !important;
                width: 100% !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(4) .mantine-Flex-root > div:first-child,
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(5) .mantine-Flex-root > div:first-child {
                width: auto !important;
                padding-right: 4px !important;
            }
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(6),
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(7),
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(8),
            div:has(> div > #teleop-horizontal-anchor) > div:nth-child(9) {
                width: 100% !important;
            }
            div:has(> div > #teleop-horizontal-anchor) .mantine-Slider-root {
                flex-grow: 1 !important;
                min-width: 80px !important;
            }
            </style>
            """)

            self.cb_teleop = self.server.gui.add_checkbox("Teleop", initial_value=False)
            self.cb_gripper = self.server.gui.add_checkbox("Gripper", initial_value=False)

            self.server.gui.add_divider()

            self.slider_sensitivity = self.server.gui.add_slider(
                "Scale",
                min=0.25,
                max=3.0,
                step=0.05,
                initial_value=1.0
            )
            self.server.gui.add_divider()

            self.btn_home = self.server.gui.add_button("Go to Home")

        # Data Recorder Setup GUI folder
        with self.server.gui.add_folder("Data Recorder Setup", expand_by_default=True):
            self.server.gui.add_html("""
            <div id="recorder-grid-anchor" style="display:none;"></div>
            <style>
            div:has(> div > #recorder-grid-anchor) {
                display: flex !important;
                flex-wrap: wrap !important;
                row-gap: 4px !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:has(#recorder-grid-anchor) {
                display: none !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(2),
            div:has(> div > #recorder-grid-anchor) > div:nth-child(3) {
                width: 50% !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                padding-right: 4px !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(2) > div,
            div:has(> div > #recorder-grid-anchor) > div:nth-child(3) > div {
                width: 100% !important;
                padding-bottom: 0 !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(2) .mantine-Flex-root,
            div:has(> div > #recorder-grid-anchor) > div:nth-child(3) .mantine-Flex-root {
                justify-content: space-between !important;
                width: 100% !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(2) .mantine-Flex-root > div:first-child,
            div:has(> div > #recorder-grid-anchor) > div:nth-child(3) .mantine-Flex-root > div:first-child {
                width: auto !important;
                padding-right: 4px !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(n+5) {
                width: 50% !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                padding-right: 4px !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(n+5) > div {
                width: 100% !important;
                padding-bottom: 0 !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(n+5) .mantine-Flex-root {
                justify-content: space-between !important;
                width: 100% !important;
            }
            div:has(> div > #recorder-grid-anchor) > div:nth-child(n+5) .mantine-Flex-root > div:first-child {
                width: auto !important;
                padding-right: 4px !important;
            }
            </style>
            """)
            self.cb_record = self.server.gui.add_checkbox("Start Recording", initial_value=False)
            self.cb_discard = self.server.gui.add_checkbox("Discard", initial_value=False)
            self.server.gui.add_divider()
            self.cb_stream_kinematics = self.server.gui.add_checkbox("Kinematics", initial_value=True)
            self.cb_stream_dynamics = self.server.gui.add_checkbox("Dynamics", initial_value=True)
            self.cb_stream_wrench = self.server.gui.add_checkbox("Wrench", initial_value=True)
            self.cb_stream_temperatures = self.server.gui.add_checkbox("Temperatures", initial_value=False)
            self.cb_stream_payload = self.server.gui.add_checkbox("Payload", initial_value=True)
            self.cb_stream_safety = self.server.gui.add_checkbox("Safety Flags", initial_value=True)
            self.cb_stream_rgb = self.server.gui.add_checkbox("RGB Camera", initial_value=True)
            self.cb_stream_depth = self.server.gui.add_checkbox("Depth Camera", initial_value=False)

        # Visual Debug GUI folder
        with self.server.gui.add_folder("Visual Debug", expand_by_default=True):
            self.cb_show_collision = self.server.gui.add_checkbox(
                "Show Collision",
                initial_value=True
            )

        @self.cb_show_collision.on_update
        def _(event):
            self.set_collision_visibility(event.target.value)

    def set_collision_visibility(self, visible: bool):
        """Toggle visibility of robot +0.5cm inflated armor meshes and table armor."""
        for handle in self._robot_armor_handles:
            handle.visible = visible
        if hasattr(self, "_collision_frame"):
            self._collision_frame.visible = visible
        for handle in self._collision_handles.values():
            handle.visible = visible

    def get_recorder_config(self) -> Dict[str, bool]:
        """Extract recorder configuration flags from Viser UI."""
        return {
            "robot_kinematics": bool(self.cb_stream_kinematics.value),
            "robot_target": bool(self.cb_stream_kinematics.value),
            "robot_dynamics": bool(self.cb_stream_dynamics.value),
            "robot_contact_wrench": bool(self.cb_stream_wrench.value),
            "robot_temperatures": bool(self.cb_stream_temperatures.value),
            "robot_payload": bool(self.cb_stream_payload.value),
            "robot_safety_status": bool(self.cb_stream_safety.value),
            "camera_wrist_rgb": bool(self.cb_stream_rgb.value),
            "camera_wrist_depth": bool(self.cb_stream_depth.value),
            "actions": True,
            "timestamps": True,
        }

    def set_recording_state(self, is_recording: bool, ep_idx: int = 0):
        """Update recording checkbox label."""
        if hasattr(self, "cb_record") and self.cb_record is not None:
            self.cb_record.value = is_recording
            if is_recording:
                self.cb_record.label = f"Recording (Ep {ep_idx})"
            else:
                self.cb_record.label = "Start Recording"

    def get_camera_view_mode(self) -> str:
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
        """Update robot configuration in Viser 3D (CPU-only)."""
        if q is not None and len(q) > 0:
            self._current_q0 = float(q[0])
            self.viser_urdf.update_cfg(np.array(q, dtype=float))

    def update_cam_preview(self, frame_rgb: np.ndarray):
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
        """Calculate anchor position and orientation of 3D HUD in camera space."""
        try:
            r_cam = transforms3d.quaternions.quat2mat(cam_wxyz)
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
        """Anchor 3D HUD to bottom-right viewport of active camera."""
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
        Update 3D Controller HUD:
          1. Color variant according to active buttons.
          2. Rotation synchronization when trigger / grip is held.
          3. Micro-translation response (max 2cm).
          4. Elastic snap back to base pose upon trigger release.
        """
        if hasattr(self, "cb_show_controller") and not self.cb_show_controller.value:
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

        if self._hud_mode == "viewport":
            if self._active_camera is not None:
                self._calc_hud_pose_from_cam(self._active_camera.position, self._active_camera.wxyz)
            else:
                clients = self.server.get_clients()
                if clients:
                    c = next(iter(clients.values()))
                    self._active_camera = c.camera
                    self._calc_hud_pose_from_cam(c.camera.position, c.camera.wxyz)

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
            self._hud_frame.wxyz = self._hud_base_wxyz

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
                    disp = (d_lat * u_right - d_vert * u_down + d_fwd * u_fwd) * alpha_micro
                else:
                    disp = np.array([d_lat, -d_vert, d_fwd], dtype=float) * alpha_micro

                disp_norm = np.linalg.norm(disp)
                max_disp = 0.020  # Max 2.0cm micro-translation
                if disp_norm > max_disp:
                    disp = disp * (max_disp / disp_norm)
                self._hud_frame.position = tuple(self._hud_base_pos + disp)
            else:
                self._hud_frame.position = tuple(self._hud_base_pos)
