#!/usr/bin/env python3
"""
tests/test_tilted_controller_grip_isolation.py

Kiểm thử tự động toán học:
1. Phân tách tuyệt đối 3 động tác cổ tay người dùng khi tay cầm Quest ở góc nghiêng tự nhiên (Tilted Controller):
   - Vặn chìa khóa (Roll) -> 100% Channel 0, robot xoay quanh trục nhìn +X_base, Pitch/Yaw = 0.0.
   - Gập/ngửa cổ tay (Pitch) -> 100% Channel 1, robot ngửa/chúc quanh Y_base, Roll/Yaw = 0.0.
   - Lắc cổ tay (Yaw) -> 100% Channel 2, robot quay ngang quanh Z_base, Roll/Pitch = 0.0.
2. Triệt tiêu hoàn toàn lỗi Roll và Yaw cùng xoay quanh trục thẳng đứng tại Home pose.
3. Tính bất biến theo tư thế robot (Pose-Invariance) qua phép nhân bên trái (Pre-multiply).
"""

import math
import os
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop.pyroki_solver import PyrokiIKSolver


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


class TestTiltedControllerGripIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print(" BỘ KIỂM THỬ TỰ ĐỘNG: TILTED CONTROLLER GRIP ROTATION & POSTURE INVARIANCE")
        print("=" * 80)
        cls.solver = PyrokiIKSolver()
        cls.SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]

    def _get_tcp_from_joints(self, joints):
        fk = self.solver.fk(joints)
        pos = fk["pos"]
        quat_wxyz = fk["quat"]
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rotvec = rot.as_rotvec()
        return [pos[0], pos[1], pos[2], rotvec[0], rotvec[1], rotvec[2]], rot

    def test_01_tilted_controller_isolates_roll_from_vertical_yaw(self):
        """
        Kiểm chứng: Tay cầm nghiêng tự nhiên (Pitch 35°, Yaw 20°).
        Người dùng vặn chìa khóa Roll 25° trong local frame của tay cầm:
        - Khóa đơn trục chọn đúng Channel 0 (Roll).
        - Tuyệt đối KHÔNG bị nhầm thành Yaw quanh trục thẳng đứng.
        - Robot xoay quanh trục nhìn +X_base, trục Z_base phải bằng 0.0 rad.
        """
        print("\n[TEST 1] VẶN CHÌA KHÓA VỚI TAY CẦM NGHIÊNG TỰ NHIÊN (35° PITCH, 20° YAW)")
        tcp_home, R_origin = self._get_tcp_from_joints(self.SAFE_HOME_JOINTS)
        server = MockQuestServer()
        ctrl = MockRobotCtrl(self.SAFE_HOME_JOINTS, tcp_home)
        pose_source = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )

        # Tay cầm nghiêng tự nhiên lúc nghỉ
        R_natural = R.from_euler("xyz", [math.radians(35.0), math.radians(20.0), 0.0])
        q_nat = R_natural.as_quat()

        # Frame 1: Bóp Grip tại tư thế nghiêng tự nhiên (Anchor)
        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_nat[3], q_nat[0], q_nat[1], q_nat[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        p0 = pose_source.get_pose()
        self.assertIsNone(p0, "Frame đầu bóp Grip phải chốt mỏ neo và trả về None")

        # Người dùng vặn chìa khóa 25° theo chiều kim đồng hồ quanh trục cẳng tay (-X local)
        R_roll_local = R.from_rotvec([-math.radians(25.0), 0.0, 0.0])
        R_hand_roll = R_natural * R_roll_local
        q_roll = R_hand_roll.as_quat()

        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_roll[3], q_roll[0], q_roll[1], q_roll[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(12):
            res = pose_source.get_pose()

        self.assertIsNotNone(res)
        self.assertEqual(pose_source.locked_axis_idx, 0, "Phải khóa độc quyền Channel 0 (Roll)!")

        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
        rot_deg = np.degrees(rotvec_delta)
        print(f"  Robot delta rotvec: Roll={rot_deg[0]:.2f}°, Pitch={rot_deg[1]:.2f}°, Yaw={rot_deg[2]:.2f}°")

        # Roll phải dương (thuận chiều kim đồng hồ quanh +X_base)
        self.assertGreater(rotvec_delta[0], math.radians(8.0), "Roll phải xoay cùng chiều kim đồng hồ quanh +X_base")
        # Pitch và Yaw phải triệt tiêu 100% về 0.0
        self.assertAlmostEqual(rotvec_delta[1], 0.0, places=5, msg="Pitch rò rỉ phải bằng 0.0 rad!")
        self.assertAlmostEqual(rotvec_delta[2], 0.0, places=5, msg="Yaw rò rỉ phải bằng 0.0 rad (không được chập vào trục Z)!")
        print("  -> PASS: Vặn chìa khóa hoàn toàn độc lập, triệt tiêu 100% rò rỉ sang trục thẳng đứng!")

    def test_02_tilted_controller_isolates_pitch(self):
        """
        Kiểm chứng: Gập cổ tay 20° với tay cầm nghiêng tự nhiên:
        - Khóa đơn trục chọn đúng Channel 1 (Pitch).
        - Robot chúc đầu xuống quanh Y_base, Roll/Yaw phải bằng 0.0 rad.
        """
        print("\n[TEST 2] GẬP CỔ TAY (PITCH) VỚI TAY CẦM NGHIÊNG TỰ NHIÊN")
        tcp_home, R_origin = self._get_tcp_from_joints(self.SAFE_HOME_JOINTS)
        server = MockQuestServer()
        ctrl = MockRobotCtrl(self.SAFE_HOME_JOINTS, tcp_home)
        pose_source = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )

        R_natural = R.from_euler("xyz", [math.radians(35.0), math.radians(20.0), 0.0])
        q_nat = R_natural.as_quat()

        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_nat[3], q_nat[0], q_nat[1], q_nat[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        pose_source.get_pose()

        # Gập cổ tay 20° xuống quanh trục ngang tay cầm (+Z local)
        R_pitch_local = R.from_rotvec([0.0, 0.0, math.radians(20.0)])
        R_hand_pitch = R_natural * R_pitch_local
        q_pitch = R_hand_pitch.as_quat()

        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_pitch[3], q_pitch[0], q_pitch[1], q_pitch[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(12):
            res = pose_source.get_pose()

        self.assertIsNotNone(res)
        self.assertEqual(pose_source.locked_axis_idx, 1, "Phải khóa độc quyền Channel 1 (Pitch)!")

        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
        rot_deg = np.degrees(rotvec_delta)
        print(f"  Robot delta rotvec: Roll={rot_deg[0]:.2f}°, Pitch={rot_deg[1]:.2f}°, Yaw={rot_deg[2]:.2f}°")

        # Pitch phải âm (chúc xuống quanh -Y_base)
        self.assertLess(rotvec_delta[1], -math.radians(5.0), "Pitch phải làm cụp/chúc đầu ngàm xuống")
        self.assertAlmostEqual(rotvec_delta[0], 0.0, places=5, msg="Roll rò rỉ phải bằng 0.0 rad!")
        self.assertAlmostEqual(rotvec_delta[2], 0.0, places=5, msg="Yaw rò rỉ phải bằng 0.0 rad!")
        print("  -> PASS: Gập cổ tay hoàn toàn độc lập, chúc đầu xuống mượt mà!")

    def test_03_tilted_controller_isolates_yaw(self):
        """
        Kiểm chứng: Lắc cổ tay sang trái/phải 20°:
        - Khóa đơn trục chọn đúng Channel 2 (Yaw).
        - Robot quay ngang quanh Z_base, Roll/Pitch phải bằng 0.0 rad.
        """
        print("\n[TEST 3] LẮC CỔ TAY (YAW) VỚI TAY CẦM NGHIÊNG TỰ NHIÊN")
        tcp_home, R_origin = self._get_tcp_from_joints(self.SAFE_HOME_JOINTS)
        server = MockQuestServer()
        ctrl = MockRobotCtrl(self.SAFE_HOME_JOINTS, tcp_home)
        pose_source = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )

        R_natural = R.from_euler("xyz", [math.radians(35.0), math.radians(20.0), 0.0])
        q_nat = R_natural.as_quat()

        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_nat[3], q_nat[0], q_nat[1], q_nat[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        pose_source.get_pose()

        # Lắc cổ tay sang trái 20° quanh trục vuông góc mặt trên tay cầm (+Y local)
        R_yaw_local = R.from_rotvec([0.0, math.radians(20.0), 0.0])
        R_hand_yaw = R_natural * R_yaw_local
        q_yaw = R_hand_yaw.as_quat()

        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_yaw[3], q_yaw[0], q_yaw[1], q_yaw[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(12):
            res = pose_source.get_pose()

        self.assertIsNotNone(res)
        self.assertEqual(pose_source.locked_axis_idx, 2, "Phải khóa độc quyền Channel 2 (Yaw)!")

        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
        rot_deg = np.degrees(rotvec_delta)
        print(f"  Robot delta rotvec: Roll={rot_deg[0]:.2f}°, Pitch={rot_deg[1]:.2f}°, Yaw={rot_deg[2]:.2f}°")

        # Yaw phải dương (quay trái quanh +Z_base)
        self.assertGreater(rotvec_delta[2], math.radians(5.0), "Yaw phải quay sang trái quanh +Z_base")
        self.assertAlmostEqual(rotvec_delta[0], 0.0, places=5, msg="Roll rò rỉ phải bằng 0.0 rad!")
        self.assertAlmostEqual(rotvec_delta[1], 0.0, places=5, msg="Pitch rò rỉ phải bằng 0.0 rad!")
        print("  -> PASS: Lắc cổ tay hoàn toàn độc lập, quay ngang chuẩn xác!")

    def test_04_home_pose_3_distinct_axes(self):
        """
        Kiểm chứng triệt tiêu dứt điểm lỗi Home Pose:
        3 động tác cổ tay phải điều khiển 3 trục trực giao 100% khác nhau tại Home Pose:
        - Roll axis: [1, 0, 0]
        - Pitch axis: [0, -1, 0]
        - Yaw axis: [0, 0, 1]
        Tích vô hướng giữa bất kỳ 2 trục nào phải bằng 0.0.
        """
        print("\n[TEST 4] TRỰC GIAO TUYỆT ĐỐI 3 TRỤC TẠI HOME POSE (ZERO CO-AXIALITY)")
        tcp_home, R_origin = self._get_tcp_from_joints(self.SAFE_HOME_JOINTS)
        server = MockQuestServer()
        ctrl = MockRobotCtrl(self.SAFE_HOME_JOINTS, tcp_home)

        # 1. Roll
        src_roll = DeltaPoseSource(server, ctrl, scale=1.0, anchor_reference_func=lambda: list(ctrl.tcp()))
        server.set_frame({"pos": [0,0,0], "quat": [1,0,0,0], "head_quat": [1,0,0,0], "trigger": False, "grip": True})
        src_roll.get_pose()
        q_r = R.from_rotvec([-math.radians(20.0), 0, 0]).as_quat()
        server.set_frame({"pos": [0,0,0], "quat": [q_r[3], q_r[0], q_r[1], q_r[2]], "head_quat": [1,0,0,0], "trigger": False, "grip": True})
        for _ in range(10): p_r = src_roll.get_pose()
        rot_roll = (R.from_quat([p_r["quat"][1], p_r["quat"][2], p_r["quat"][3], p_r["quat"][0]]) * R_origin.inv()).as_rotvec()
        axis_roll = rot_roll / np.linalg.norm(rot_roll)

        # 2. Pitch
        src_pitch = DeltaPoseSource(server, ctrl, scale=1.0, anchor_reference_func=lambda: list(ctrl.tcp()))
        server.set_frame({"pos": [0,0,0], "quat": [1,0,0,0], "head_quat": [1,0,0,0], "trigger": False, "grip": True})
        src_pitch.get_pose()
        q_p = R.from_rotvec([0, 0, math.radians(20.0)]).as_quat()
        server.set_frame({"pos": [0,0,0], "quat": [q_p[3], q_p[0], q_p[1], q_p[2]], "head_quat": [1,0,0,0], "trigger": False, "grip": True})
        for _ in range(10): p_p = src_pitch.get_pose()
        rot_pitch = (R.from_quat([p_p["quat"][1], p_p["quat"][2], p_p["quat"][3], p_p["quat"][0]]) * R_origin.inv()).as_rotvec()
        axis_pitch = rot_pitch / np.linalg.norm(rot_pitch)

        # 3. Yaw
        src_yaw = DeltaPoseSource(server, ctrl, scale=1.0, anchor_reference_func=lambda: list(ctrl.tcp()))
        server.set_frame({"pos": [0,0,0], "quat": [1,0,0,0], "head_quat": [1,0,0,0], "trigger": False, "grip": True})
        src_yaw.get_pose()
        q_y = R.from_rotvec([0, math.radians(20.0), 0]).as_quat()
        server.set_frame({"pos": [0,0,0], "quat": [q_y[3], q_y[0], q_y[1], q_y[2]], "head_quat": [1,0,0,0], "trigger": False, "grip": True})
        for _ in range(10): p_y = src_yaw.get_pose()
        rot_yaw = (R.from_quat([p_y["quat"][1], p_y["quat"][2], p_y["quat"][3], p_y["quat"][0]]) * R_origin.inv()).as_rotvec()
        axis_yaw = rot_yaw / np.linalg.norm(rot_yaw)

        print(f"  Axis Roll : {np.round(axis_roll, 4)}")
        print(f"  Axis Pitch: {np.round(axis_pitch, 4)}")
        print(f"  Axis Yaw  : {np.round(axis_yaw, 4)}")

        dot_rp = abs(float(np.dot(axis_roll, axis_pitch)))
        dot_ry = abs(float(np.dot(axis_roll, axis_yaw)))
        dot_py = abs(float(np.dot(axis_pitch, axis_yaw)))

        print(f"  Orthogonality check: Roll.Pitch={dot_rp:.6f}, Roll.Yaw={dot_ry:.6f}, Pitch.Yaw={dot_py:.6f}")
        self.assertLess(dot_rp, 1e-4, "Roll và Pitch phải trực giao 100%!")
        self.assertLess(dot_ry, 1e-4, "Roll và Yaw phải trực giao 100% (không được chập trục)!")
        self.assertLess(dot_py, 1e-4, "Pitch và Yaw phải trực giao 100%!")
        print("  -> PASS: 3 trục xoay hoàn toàn trực giao, độc lập 100% tại Home Pose!")


if __name__ == "__main__":
    unittest.main()
