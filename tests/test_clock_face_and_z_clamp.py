#!/usr/bin/env python3
"""
tests/test_clock_face_and_z_clamp.py

Xác minh 2 tiêu chí nghiệm thu (Acceptance Criteria):
- Test 1 (Clock-Face in New Posture):
  Thay đổi tư thế robot sang nhiều góc khác nhau (30°, 60°, 90°, -45°, 120°)
  -> Thao tác điều khiển (Tịnh tiến & Xoay) vẫn tuân thủ chuẩn xác hệ trục mặt đồng hồ:
     + Tịnh tiến: Tiến dọc theo phương vươn u_radial, Phải quay theo cung tiếp tuyến u_tangential.
     + Xoay: Vặn chìa khóa quay quanh u_radial, Gập cổ tay cụp theo u_radial, Quét quay quanh Z.
- Test 2 (Z-Axis Clamp):
  Xoay góc Z (Yaw) đến cực hạn (90°, 120°, 180°)
  -> Hệ thống tự động kẹp cứng ở ngưỡng an toàn max_z_rotation_delta (<= 75° ~ 1.31 rad),
     tuyệt đối không gây vặn xoắn gãy tay máy hay nổ nghiệm IK.
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
from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS


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


class TestClockFaceAndZClamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = PyrokiIKSolver()

    def test_01_clock_face_in_new_postures_translation_and_rotation(self):
        """Test 1: Hệ quy chiếu mặt đồng hồ luôn đồng bộ và chính xác ở mọi tư thế"""
        server = MockQuestServer()
        test_angles_deg = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, -45.0]

        for deg in test_angles_deg:
            rad = math.radians(deg)
            r_dist = 0.35
            tcp_x = r_dist * math.cos(rad)
            tcp_y = r_dist * math.sin(rad)
            tcp_z = 0.25

            u_radial = np.array([math.cos(rad), math.sin(rad), 0.0])
            u_tangential_right = np.array([math.sin(rad), -math.cos(rad), 0.0])

            # Robot đang ở tư thế này
            R_anchor = R.from_euler("xyz", [0.0, math.pi, 0.0])  # TCP chúc xuống
            rotvec_anchor = R_anchor.as_rotvec()
            tcp_pose = [tcp_x, tcp_y, tcp_z, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
            joints = [rad, -1.57, 1.57, -1.57, -1.57, 0.0]

            ctrl = MockRobotCtrl(joints, tcp_pose)
            src = DeltaPoseSource(
                server, ctrl, scale=1.0,
                anchor_reference_func=lambda: list(ctrl.tcp())
            )
            src.nav_mode = "clock_face"
            src.rotation_mode = "decoupled_3axis"

            # ── 1. Thao tác TỊNH TIẾN KHÔNG ẤN GRIP (chỉ bóp Trigger) ──
            # Anchor frame
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": True, "grip": False
            })
            p0 = src.get_pose()

            # Đẩy tay Tiến về phía trước 3cm (d_fwd = +0.03m theo trục nhìn tới u_fwd)
            server.set_frame({
                "pos": [0.0, 0.0, -0.03], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": True, "grip": False
            })
            p_fwd = None
            for _ in range(15):
                p_fwd = src.get_pose()

            self.assertIsNotNone(p_fwd)
            delta_pos = np.array(p_fwd["pos"][:2]) - np.array([tcp_x, tcp_y])
            delta_pos_dir = delta_pos / np.linalg.norm(delta_pos)
            dot_radial = float(np.dot(delta_pos_dir, u_radial[:2]))
            self.assertGreaterEqual(
                dot_radial, 0.99,
                f"Tại góc {deg}°, đẩy tiến phải vươn dọc theo u_radial! Dot={dot_radial}"
            )

            # ── 2. Thao tác XOAY CÓ ẤN GRIP (Vặn Roll chìa khóa) ──
            # Nhả phím trước khi bóp lại Grip
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
            })
            src.get_pose()

            # Re-anchor với Grip
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            src.get_pose()

            # Vặn chìa khóa sang phải 30° quanh u_fwd
            R_roll = R.from_rotvec([0.0, 0.0, -math.radians(30.0)])
            q_roll = R_roll.as_quat()
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [q_roll[3], q_roll[0], q_roll[1], q_roll[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            p_roll = None
            for _ in range(15):
                p_roll = src.get_pose()

            self.assertIsNotNone(p_roll)
            q_res = p_roll["quat"]
            R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])
            rotvec_delta = (R_target * R_anchor.inv()).as_rotvec()
            rot_axis = rotvec_delta / np.linalg.norm(rotvec_delta)
            dot_roll = float(np.dot(rot_axis, u_radial))
            self.assertGreaterEqual(
                dot_roll, 0.99,
                f"Tại góc {deg}°, Roll phải xoay quanh trục vươn u_radial! Dot={dot_roll}"
            )

            # ── 3. Thao tác XOAY PITCH (Gập cổ tay xuống) ──
            # Nhả phím trước khi đổi trục xoay
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
            })
            src.get_pose()

            # Re-anchor với Grip
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            src.get_pose()

            R_pitch = R.from_rotvec([math.radians(25.0), 0.0, 0.0])
            q_pitch = R_pitch.as_quat()
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [q_pitch[3], q_pitch[0], q_pitch[1], q_pitch[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            p_pitch = None
            for _ in range(15):
                p_pitch = src.get_pose()

            q_p = p_pitch["quat"]
            R_p = R.from_quat([q_p[1], q_p[2], q_p[3], q_p[0]])
            v_tip_new = R_p.apply([0, 0, 1])
            v_tip_old = R_anchor.apply([0, 0, 1])
            tip_deflect = v_tip_new - v_tip_old
            tip_deflect_xy = np.array([tip_deflect[0], tip_deflect[1], 0.0])
            tip_deflect_dir = tip_deflect_xy / np.linalg.norm(tip_deflect_xy)
            dot_pitch = float(np.dot(tip_deflect_dir, u_radial))
            self.assertGreaterEqual(
                dot_pitch, 0.99,
                f"Tại góc {deg}°, Gập cổ tay phải làm cụp ngàm về phía trước u_radial! Dot={dot_pitch}"
            )

    def test_02_strict_z_axis_yaw_rotation_clamping(self):
        """Test 2: Xoay góc Z (Yaw) đến cực hạn -> Hệ thống kẹp cứng ở ngưỡng an toàn <= 75°"""
        server = MockQuestServer()
        joints_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        R_anchor = R.from_euler("xyz", [0.0, math.pi, 0.0])
        rotvec_anchor = R_anchor.as_rotvec()
        tcp_home = [0.30, 0.0, 0.25, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
        ctrl = MockRobotCtrl(joints_home, tcp_home)

        src = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )
        src.nav_mode = "clock_face"
        src.max_z_rotation_delta = math.radians(75.0)  # 75 độ

        # Test các góc xoay quét cực hạn: 90°, 120°, 180°, -150°
        extreme_yaw_angles_deg = [90.0, 120.0, 180.0, -90.0, -150.0]

        for yaw_deg in extreme_yaw_angles_deg:
            # Re-anchor
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            src.get_pose()

            # Quét Yaw quanh u_up = [0, 1, 0] trong WebXR
            # Quét sang phải (góc âm) hoặc trái (góc dương)
            R_yaw = R.from_rotvec([0.0, math.radians(yaw_deg), 0.0])
            q_yaw = R_yaw.as_quat()
            server.set_frame({
                "pos": [0.0, 0.0, 0.0], "quat": [q_yaw[3], q_yaw[0], q_yaw[1], q_yaw[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })

            res = None
            for _ in range(25):
                res = src.get_pose()

            self.assertIsNotNone(res)
            q_out = res["quat"]
            R_out = R.from_quat([q_out[1], q_out[2], q_out[3], q_out[0]])
            delta_R = R_out * R_anchor.inv()
            delta_rotvec = delta_R.as_rotvec()

            yaw_output_rad = abs(delta_rotvec[2])
            yaw_output_deg = np.degrees(yaw_output_rad)

            # Ngưỡng tối đa cho phép là max_z_rotation_delta (75°)
            max_allowed_deg = 75.0 + 0.5  # sai số float nhỏ
            self.assertLessEqual(
                yaw_output_deg, max_allowed_deg,
                f"Góc Yaw ra ({yaw_output_deg:.2f}°) vượt ngưỡng kẹp tối đa 75° khi input là {yaw_deg}°!"
            )

            # Giải IK tư thế kẹp biên để xác nhận không vặn xoắn gãy tay máy
            pose_tuple = (res["pos"], res["quat"])
            sol = self.solver.solve(pose_tuple, q_seed=joints_home, is_pure_rot=True)
            self.assertIsNotNone(sol["q"], "IK Solver phải tìm được nghiệm hợp lệ khi góc Yaw bị kẹp an toàn!")

            q_sol = sol["q"]
            # Kiểm tra toàn bộ 6 khớp nằm nghiêm ngặt trong giới hạn UR3E_LIMITS
            for j_idx in range(6):
                lo, hi = UR3E_LIMITS[j_idx]
                self.assertGreaterEqual(q_sol[j_idx], lo - 1e-3, f"Khớp q{j_idx} vi phạm giới hạn dưới!")
                self.assertLessEqual(q_sol[j_idx], hi + 1e-3, f"Khớp q{j_idx} vi phạm giới hạn trên!")

            # Thân máy (Base, Shoulder, Elbow) không bị vặn xoay theo
            base_swing = abs(q_sol[0] - joints_home[0])
            self.assertLessEqual(
                base_swing, 0.02,
                f"Thân máy bị vặn xoay ({base_swing:.4f} rad) khi xoay Yaw kịch biên!"
            )


if __name__ == "__main__":
    unittest.main()
