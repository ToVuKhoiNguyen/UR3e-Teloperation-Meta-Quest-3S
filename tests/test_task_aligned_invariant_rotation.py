#!/usr/bin/env python3
"""
tests/test_task_aligned_invariant_rotation.py

Bộ kiểm thử toán học & động học toàn diện cho cơ chế:
TASK-ALIGNED INVARIANT MAPPING (Grip Rotation)

Tiêu chí nghiệm thu (Acceptance Criteria):
1. Test 1 (Drinking Motion - Uống nước):
   Tại mọi góc phương vị robot trên bàn máy (0°, 30°, 45°, 60°, 90°, 120°, -45°, -90°, -135°):
   Thao tác gập/ngửa cổ tay người dùng luôn tạo chuyển động Pitch trong mặt phẳng
   thẳng đứng dọc theo thân tay máy (Body Pitch / Sagittal plane).
   -> Tip lệch thuần túy dọc theo u_radial, tuyệt đối không bị lắc ngang.

2. Test 2 (Key Turning Motion - Vặn chìa khóa):
   Tại mọi tư thế robot (chúc thẳng xuống, nghiêng 45°, nằm ngang, quay camera 90°/180°):
   Thao tác vặn cổ tay người dùng luôn xoay 100% quanh trục dọc đầu công cụ (Tool Roll / Approach Axis).
   -> Vector hướng đầu công cụ v_tool bất biến tuyệt đối (norm(v_new - v_anchor) < 1e-6).
   -> Không bị lai tạp sang trục khác.

3. Test 3 (Single-Axis Dominant Isolation):
   Khi người dùng vô tình xoay đa trục (ví dụ Vặn 25° + Gập 8° + Quét 5°),
   bộ khóa đơn trục chỉ mở khóa trục chủ đạo, triệt tiêu 100% hai trục còn lại.

4. Test 4 (Z-Axis Clamping):
   Khi quét góc Z (Yaw) lớn (> 90°), góc quay bị kẹp biên an toàn <= 75° (1.31 rad).
"""

import os
import sys
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.delta_pose_source import DeltaPoseSource


class MockQuestServer:
    def __init__(self):
        self._frame = None

    def get_frame(self):
        return self._frame

    def set_frame(self, frame):
        self._frame = frame


class MockRobotCtrl:
    def __init__(self, joints, tcp):
        self._joints = list(joints)
        self._tcp = list(tcp)

    def joints(self):
        return self._joints

    def tcp(self):
        return self._tcp


class TestTaskAlignedInvariantRotation(unittest.TestCase):
    def test_01_drinking_motion_body_pitch_invariant_across_postures(self):
        """
        Acceptance Criteria 1: Động tác 'Uống nước' (Pitch Bending)
        Gập/ngửa cổ tay luôn thực hiện động tác gập ngửa dọc theo thân tay máy ở mọi góc giờ.
        """
        server = MockQuestServer()
        test_angles_deg = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, -45.0, -90.0, -135.0]

        for deg in test_angles_deg:
            rad = math.radians(deg)
            r_dist = 0.35
            tcp_x = r_dist * math.cos(rad)
            tcp_y = r_dist * math.sin(rad)
            tcp_z = 0.25

            u_radial = np.array([math.cos(rad), math.sin(rad), 0.0], dtype=np.float64)
            u_pitch_axis_expected = np.array([-math.sin(rad), math.cos(rad), 0.0], dtype=np.float64)

            # Robot ở tư thế chúc xuống
            R_anchor = R.from_euler("xyz", [0.0, math.pi, 0.0])
            rotvec_anchor = R_anchor.as_rotvec()
            tcp_pose = [tcp_x, tcp_y, tcp_z, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
            joints = [rad, -1.57, 1.57, -1.57, -1.57, 0.0]

            ctrl = MockRobotCtrl(joints, tcp_pose)
            src = DeltaPoseSource(
                server, ctrl, scale=1.0,
                anchor_reference_func=lambda: list(ctrl.tcp())
            )
            src.rotation_mode = "task_aligned"
            src.nav_mode = "clock_face"

            # ── 1. GẬP CỔ TAY (Flexion: cụp ngàm về phía trước như rót/uống nước) ──
            # Clutch reset
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
            })
            src.get_pose()

            # Anchor Grip
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            src.get_pose()

            # Gập cổ tay +25° (Pitch quanh trục 2 của tay cầm)
            R_pitch_down = R.from_rotvec([0.0, 0.0, math.radians(25.0)])
            q_pitch = R_pitch_down.as_quat()
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [q_pitch[3], q_pitch[0], q_pitch[1], q_pitch[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            p_res = None
            for _ in range(15):
                p_res = src.get_pose()

            self.assertIsNotNone(p_res)
            q_res = p_res["quat"]
            R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])

            # Kiểm tra 1: Trục xoay Delta phải song song tuyệt đối với u_pitch_axis
            R_delta = R_target * R_anchor.inv()
            rotvec_delta = R_delta.as_rotvec()
            rot_mag = np.linalg.norm(rotvec_delta)
            self.assertGreater(rot_mag, 0.05, f"Tại góc {deg}°, góc xoay quá nhỏ!")
            rot_axis = rotvec_delta / rot_mag

            dot_pitch_axis = abs(float(np.dot(rot_axis, u_pitch_axis_expected)))
            self.assertGreaterEqual(
                dot_pitch_axis, 0.999,
                f"Tại góc {deg}°, trục xoay pitch phải khớp u_pitch_axis! Dot={dot_pitch_axis}"
            )

            # Kiểm tra 2: Độ lệch đầu công cụ (tip deflection) phải thuần túy hướng dọc theo u_radial
            v_old = R_anchor.apply([0.0, 0.0, 1.0])
            v_new = R_target.apply([0.0, 0.0, 1.0])
            delta_v = v_new - v_old
            delta_v_xy = np.array([delta_v[0], delta_v[1], 0.0])
            norm_xy = np.linalg.norm(delta_v_xy)
            self.assertGreater(norm_xy, 1e-4)
            dir_xy = delta_v_xy / norm_xy

            dot_radial = float(np.dot(dir_xy, u_radial))
            self.assertGreaterEqual(
                dot_radial, 0.999,
                f"Tại góc {deg}°, tip deflection phải hướng dọc theo u_radial! Dot={dot_radial}"
            )

            # Kiểm tra 3: Thành phần lệch ngang (lateral) phải bằng 0 tuyệt đối
            lateral_comp = abs(float(np.dot(dir_xy, u_pitch_axis_expected)))
            self.assertLess(
                lateral_comp, 0.01,
                f"Tại góc {deg}°, xuất hiện lệch ngang khi gập uống nước! Lateral={lateral_comp}"
            )

    def test_02_key_turning_motion_pure_tool_roll_in_any_posture(self):
        """
        Acceptance Criteria 2: Động tác 'Vặn chìa khóa' (Tool Roll)
        Vặn cổ tay luôn xoay thuần túy quanh trục dọc đầu công cụ (Approach Axis).
        Vector hướng của đầu công cụ không đổi (norm(v_new - v_old) < 1e-6).
        """
        server = MockQuestServer()

        # Thử nghiệm với các tư thế công cụ đa dạng:
        # 1. Chúc thẳng đứng xuống bàn: Euler [0, pi, 0]
        # 2. Nghiêng 45°: Euler [pi/4, pi/2, 0]
        # 3. Nằm ngang hướng ra ngoài: Euler [0, pi/2, 0]
        # 4. Quay camera/wrist 90° và 180°
        test_tool_orientations = [
            ("Chúc thẳng xuống bàn", R.from_euler("xyz", [0.0, math.pi, 0.0])),
            ("Nghiêng 45° ra trước", R.from_euler("xyz", [math.pi / 4, math.pi / 2, 0.0])),
            ("Nằm ngang hướng +X", R.from_euler("xyz", [0.0, math.pi / 2, 0.0])),
            ("Xoay camera 90°", R.from_euler("zyx", [math.pi / 2, 0.0, math.pi])),
            ("Xoay camera -60° nghiêng", R.from_euler("zyx", [-math.pi / 3, math.pi / 4, math.pi])),
        ]

        test_arm_angles = [0.0, 45.0, 90.0, -60.0]

        for arm_deg in test_arm_angles:
            arm_rad = math.radians(arm_deg)
            r_dist = 0.35
            tcp_x = r_dist * math.cos(arm_rad)
            tcp_y = r_dist * math.sin(arm_rad)
            tcp_z = 0.25

            for orient_name, R_anchor in test_tool_orientations:
                # Trục dọc đầu công cụ thực tế trong Base Frame:
                u_tool_axis_expected = R_anchor.apply([0.0, 0.0, 1.0])
                u_tool_axis_expected /= np.linalg.norm(u_tool_axis_expected)

                rotvec_anchor = R_anchor.as_rotvec()
                tcp_pose = [tcp_x, tcp_y, tcp_z, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
                joints = [arm_rad, -1.57, 1.57, -1.57, -1.57, 0.0]

                ctrl = MockRobotCtrl(joints, tcp_pose)
                src = DeltaPoseSource(
                    server, ctrl, scale=1.0,
                    anchor_reference_func=lambda: list(ctrl.tcp())
                )
                src.rotation_mode = "task_aligned"
                src.nav_mode = "clock_face"

                # ── Thao tác VẶN CHÌA KHÓA SANG PHẢI (Clockwise quanh u_fwd) ──
                # Clutch release
                server.set_frame({
                    "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                    "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
                })
                src.get_pose()

                # Re-anchor with Grip
                server.set_frame({
                    "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                    "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
                })
                src.get_pose()

                # Vặn cổ tay sang phải 30° quanh trục cẳng tay (trục 0 của tay cầm)
                R_wrist_twist = R.from_rotvec([-math.radians(30.0), 0.0, 0.0])
                q_twist = R_wrist_twist.as_quat()
                server.set_frame({
                    "pos": [0.0, 0.0, 0.0], "quat": [q_twist[3], q_twist[0], q_twist[1], q_twist[2]],
                    "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
                })
                p_res = None
                for _ in range(15):
                    p_res = src.get_pose()

                self.assertIsNotNone(p_res)
                q_res = p_res["quat"]
                R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])

                # Kiểm tra 1: Trục quay delta phải trùng 100% với trục dọc công cụ u_tool_axis
                R_delta = R_target * R_anchor.inv()
                rotvec_delta = R_delta.as_rotvec()
                rot_mag = np.linalg.norm(rotvec_delta)
                self.assertGreater(rot_mag, 0.05, f"Góc vặn quá nhỏ tại {orient_name}")
                rot_axis = rotvec_delta / rot_mag

                dot_tool = float(np.dot(rot_axis, u_tool_axis_expected))
                self.assertGreaterEqual(
                    dot_tool, 0.999,
                    f"Tại tư thế '{orient_name}' (arm={arm_deg}°), vặn chìa khóa phải xoay quanh u_tool_axis! Dot={dot_tool}"
                )

                # Kiểm tra 2: Vector hướng công cụ (Tool Approach Vector) BẤT BIẾN TUYỆT ĐỐI
                # Không bị vặn lệch hay lắc lư
                v_tool_anchor = R_anchor.apply([0.0, 0.0, 1.0])
                v_tool_new = R_target.apply([0.0, 0.0, 1.0])
                v_diff = np.linalg.norm(v_tool_new - v_tool_anchor)
                self.assertLess(
                    v_diff, 1e-6,
                    f"Tại tư thế '{orient_name}', đầu công cụ bị lệch phương khi vặn chìa khóa! diff={v_diff}"
                )

    def test_03_strict_single_axis_isolation_prevents_crosstalk(self):
        """
        Acceptance Criteria 3: Khóa đơn trục (Single-Axis Dominant Lock)
        Triệt tiêu hoàn toàn hiện tượng lẫn lộn giữa Uống nước (Pitch) và Vặn chìa khóa (Roll).
        """
        server = MockQuestServer()
        joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        R_anchor = R.from_euler("xyz", [0.0, math.pi, 0.0])
        rotvec_anchor = R_anchor.as_rotvec()
        tcp_pose = [0.35, 0.0, 0.25, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]

        ctrl = MockRobotCtrl(joints, tcp_pose)
        src = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )
        src.rotation_mode = "task_aligned"
        src.nav_mode = "clock_face"

        # Case A: Người dùng vặn chìa khóa là chính (25°), nhưng vô tình gập tay 7° và quét 5°
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        src.get_pose()

        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        src.get_pose()

        # Multi-axis gesture: Roll 25° (dominant trên trục 0), Pitch 7° (trục 2), Yaw 5° (trục 1)
        R_noisy_roll = R.from_rotvec([-math.radians(25.0), math.radians(5.0), math.radians(7.0)])
        q_noisy = R_noisy_roll.as_quat()
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_noisy[3], q_noisy[0], q_noisy[1], q_noisy[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })

        p_res = None
        for _ in range(15):
            p_res = src.get_pose()

        q_res = p_res["quat"]
        R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])
        R_delta = R_target * R_anchor.inv()
        rotvec_delta = R_delta.as_rotvec()
        rot_axis = rotvec_delta / np.linalg.norm(rotvec_delta)

        # Trục dọc công cụ
        u_tool = R_anchor.apply([0.0, 0.0, 1.0])
        u_tool /= np.linalg.norm(u_tool)

        # Phải khóa 100% vào Tool Roll, không rò rỉ sang Pitch hay Yaw
        dot_tool = float(np.dot(rot_axis, u_tool))
        self.assertGreaterEqual(dot_tool, 0.999, f"Single-axis lock thất bại với Roll dominant! Dot={dot_tool}")

        # Vector hướng công cụ bất biến
        v_diff = np.linalg.norm(R_target.apply([0, 0, 1]) - R_anchor.apply([0, 0, 1]))
        self.assertLess(v_diff, 1e-6)

        # Case B: Người dùng gập uống nước là chính (22°), vô tình vặn chìa khóa 6°
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        src.get_pose()

        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        src.get_pose()

        R_noisy_pitch = R.from_rotvec([-math.radians(6.0), 0.0, math.radians(22.0)])
        q_np = R_noisy_pitch.as_quat()
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_np[3], q_np[0], q_np[1], q_np[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })

        for _ in range(15):
            p_res = src.get_pose()

        q_res = p_res["quat"]
        R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])
        R_delta = R_target * R_anchor.inv()
        rotvec_delta = R_delta.as_rotvec()
        rot_axis = rotvec_delta / np.linalg.norm(rotvec_delta)

        # Trục pitch dự kiến tại arm=0° là [0, 1, 0]
        u_pitch = np.array([0.0, 1.0, 0.0])
        dot_pitch = abs(float(np.dot(rot_axis, u_pitch)))
        self.assertGreaterEqual(dot_pitch, 0.999, f"Single-axis lock thất bại với Pitch dominant! Dot={dot_pitch}")

    def test_04_z_axis_yaw_safety_clamping(self):
        """
        Acceptance Criteria 4: Khống chế an toàn góc quay Z (Yaw Clamping)
        Quét cổ tay cực đại vẫn kẹp biên an toàn <= 75° (1.31 rad).
        """
        server = MockQuestServer()
        joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        R_anchor = R.from_euler("xyz", [0.0, math.pi, 0.0])
        rotvec_anchor = R_anchor.as_rotvec()
        tcp_pose = [0.35, 0.0, 0.25, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]

        ctrl = MockRobotCtrl(joints, tcp_pose)
        src = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )
        src.rotation_mode = "task_aligned"
        src.nav_mode = "clock_face"

        # Clutch reset
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        src.get_pose()

        # Re-anchor with Grip
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        src.get_pose()

        # Quét cổ tay cực hạn 120°
        R_yaw_huge = R.from_rotvec([0.0, math.radians(120.0), 0.0])
        q_yh = R_yaw_huge.as_quat()
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_yh[3], q_yh[0], q_yh[1], q_yh[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })

        p_res = None
        for _ in range(15):
            p_res = src.get_pose()

        q_res = p_res["quat"]
        R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])
        R_delta = R_target * R_anchor.inv()
        rotvec_delta = R_delta.as_rotvec()

        # Góc quay Z phải <= 75° (1.31 rad)
        yaw_rad = abs(rotvec_delta[2])
        self.assertLessEqual(
            yaw_rad, math.radians(75.5),
            f"Góc Yaw vượt quá giới hạn an toàn 75°! Yaw={math.degrees(yaw_rad):.2f}°"
        )

    def test_05_tilted_controller_anatomical_wrist_isolation(self):
        """
        Acceptance Criteria 1 & 2: Phân tách tuyệt đối 3 động tác cổ tay khi tay cầm bị nghiêng tự nhiên
        (Bàn tay người dùng thường nghiêng [35°, 20°, -15°] so với mặt sàn khi cầm Quest).
        Chứng minh: Phép tính Local Frame triệt tiêu hoàn toàn lỗi chập vào trục Yaw thẳng đứng!
        """
        server = MockQuestServer()
        joints = [math.radians(30.0), -1.57, 1.57, -1.57, -1.57, 0.0]
        R_robot_anchor = R.from_euler("xyz", [0.0, math.pi, 0.0])
        rotvec_anchor = R_robot_anchor.as_rotvec()
        tcp_pose = [0.30, 0.173, 0.25, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]

        ctrl = MockRobotCtrl(joints, tcp_pose)
        src = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )
        src.rotation_mode = "task_aligned"
        src.nav_mode = "clock_face"

        # Tay cầm người dùng ở góc nghỉ tự nhiên (nghiêng 35° pitch, 20° yaw, -15° roll so với phòng)
        R_hand_resting = R.from_euler("xyz", [35.0, 20.0, -15.0], degrees=True)
        q_rest = R_hand_resting.as_quat()

        # ── KỊCH BẢN 1: Người dùng VẶN CHÌA KHÓA (Wrist Roll quanh trục dọc tay cầm) ──
        # Clutch reset
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        src.get_pose()

        # Anchor Grip tại góc nghỉ
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_rest[3], q_rest[0], q_rest[1], q_rest[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        src.get_pose()

        # Vặn cổ tay theo chiều kim đồng hồ quanh trục cẳng tay (local X: -25°)
        R_twist_local = R.from_rotvec([-math.radians(25.0), 0.0, 0.0])
        R_hand_roll = R_hand_resting * R_twist_local
        q_hr = R_hand_roll.as_quat()
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_hr[3], q_hr[0], q_hr[1], q_hr[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        p_res = None
        for _ in range(15):
            p_res = src.get_pose()

        # Xác nhận: Khóa 100% vào kênh 0 (Tool Roll), KHÔNG BỊ CHẬP VÀO YAW THẲNG ĐỨNG (kênh 2)!
        self.assertEqual(src._locked_axis_idx, 0, "Vặn chìa khóa khi tay nghiêng phải khóa vào Roll (kênh 0)!")
        q_res = p_res["quat"]
        R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])
        v_diff = np.linalg.norm(R_target.apply([0, 0, 1]) - R_robot_anchor.apply([0, 0, 1]))
        self.assertLess(v_diff, 1e-6, "Tool approach vector phải bất biến khi vặn chìa khóa!")

        # ── KỊCH BẢN 2: Người dùng GẬP CỔ TAY (Wrist Pitch quanh trục ngang tay cầm) ──
        # Clutch reset
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        src.get_pose()

        # Anchor Grip tại góc nghỉ
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_rest[3], q_rest[0], q_rest[1], q_rest[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        src.get_pose()

        # Gập cổ tay xuống quanh trục ngang tay cầm (local Z: +20°)
        R_flex_local = R.from_rotvec([0.0, 0.0, math.radians(20.0)])
        R_hand_pitch = R_hand_resting * R_flex_local
        q_hp = R_hand_pitch.as_quat()
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_hp[3], q_hp[0], q_hp[1], q_hp[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        for _ in range(15):
            p_res = src.get_pose()

        # Xác nhận: Khóa 100% vào kênh 1 (Pitch Uống nước), KHÔNG BỊ CHẬP VÀO YAW THẲNG ĐỨNG!
        self.assertEqual(src._locked_axis_idx, 1, "Gập cổ tay khi tay nghiêng phải khóa vào Pitch (kênh 1)!")

        # ── KỊCH BẢN 3: Người dùng LẮC CỔ TAY (Wrist Yaw quanh trục đứng tay cầm) ──
        # Clutch reset
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        src.get_pose()

        # Anchor Grip tại góc nghỉ
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_rest[3], q_rest[0], q_rest[1], q_rest[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        src.get_pose()

        # Lắc cổ tay sang trái quanh trục vuông góc mặt tay cầm (local Y: +20°)
        R_wag_local = R.from_rotvec([0.0, math.radians(20.0), 0.0])
        R_hand_yaw = R_hand_resting * R_wag_local
        q_hy = R_hand_yaw.as_quat()
        server.set_frame({
            "pos": [0.0, 0.0, 0.0], "quat": [q_hy[3], q_hy[0], q_hy[1], q_hy[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        for _ in range(15):
            p_res = src.get_pose()

        # Xác nhận: Khóa 100% vào kênh 2 (Base Yaw)!
        self.assertEqual(src._locked_axis_idx, 2, "Lắc cổ tay khi tay nghiêng phải khóa vào Yaw (kênh 2)!")


if __name__ == "__main__":
    unittest.main()

