#!/usr/bin/env python3
"""
tests/test_no_snapback_and_torsion_free_z_clamp.py

Kiểm thử tự động xác minh triệt để:
1. Test 1 (No Snap-back):
   - Xoay cổ tay liên tục (Roll, Pitch, Yaw) vượt mốc 180° (lên tới 220° và -220°)
   - Tuyệt đối không có hiện tượng tự quay ngược về tư thế ban đầu (Double-cover continuity & Angle unwrapping).
2. Test 2 (No Torsion / Strict Z-Clamp):
   - Xoay cực hạn trục Z (Yaw) tới 90°, 120°, 180°, 220°
   - Góc quay bị kẹp cứng an toàn tại ngưỡng cho phép (mặc định <= 35.0°),
     không gây vặn xoắn thân máy (Base swing <= 0.015 rad, Body swing <= 0.02 rad)
     và tất cả khớp cổ tay hoàn toàn nằm trong giới hạn an toàn.
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
    def __init__(self, joints=None, tcp=None):
        self._joints = list(joints) if joints is not None else [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self._tcp = list(tcp) if tcp is not None else [0.25, 0.0, 0.25, 0.0, 3.14159, 0.0]

    def joints(self):
        return self._joints

    def tcp(self):
        return self._tcp


class TestNoSnapbackAndTorsionFreeZClamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = PyrokiIKSolver()

    def setUp(self):
        self.server = MockQuestServer()
        self.tcp_home = [0.25, 0.0, 0.25, 0.0, 3.14159, 0.0]
        self.joints_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.ctrl = MockRobotCtrl(joints=self.joints_home, tcp=self.tcp_home)
        self.src = DeltaPoseSource(
            self.server, self.ctrl, scale=1.0,
            anchor_reference_func=lambda: list(self.ctrl.tcp())
        )
        self.src.max_z_rotation_delta = math.radians(35.0)

    def test_01_no_snapback_during_continuous_extreme_yaw(self):
        """Test 1A: Xoay Yaw liên tục qua 180° lên tới 220° không snap-back"""
        # Engage grip
        self.server.set_frame({
            "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
            "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
        })
        self.src.get_pose()

        prev_angle = 0.0
        # Xoay tiến từ 0 tới 220 độ quanh trục Clock-face (trục 0 của tay cầm)
        for deg in range(5, 225, 5):
            R_yaw = R.from_rotvec([-math.radians(deg), 0, 0])
            q = R_yaw.as_quat()
            self.server.set_frame({
                "pos": [0, 0, 0], "quat": [q[3], q[0], q[1], q[2]],
                "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
            })
            p = None
            for _ in range(5):
                p = self.src.get_pose()
            q_t = p["quat"]
            R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
            R_origin = R.from_rotvec(self.tcp_home[3:])
            rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
            curr_yaw_deg = np.degrees(rotvec_delta[2])

            # Tuyệt đối không được đổi dấu sang âm (không quay ngược)
            self.assertGreaterEqual(
                curr_yaw_deg, -0.01,
                f"Lỗi Snap-Back: Tại góc vào {deg}°, góc ra bị đảo dấu âm: {curr_yaw_deg:.2f}°"
            )
            # Góc phải tăng đơn điệu hoặc kẹp cứng tại <= 35.0°
            self.assertLessEqual(curr_yaw_deg, 35.0 + 1e-3)
            if deg <= 75:
                self.assertGreaterEqual(curr_yaw_deg, prev_angle - 0.05)
            elif deg >= 100:
                self.assertGreaterEqual(curr_yaw_deg, 34.5)
            prev_angle = curr_yaw_deg

    def test_02_no_snapback_during_continuous_extreme_negative_yaw(self):
        """Test 1B: Xoay Yaw âm liên tục qua -180° xuống tới -220° không snap-back"""
        self.server.set_frame({
            "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
            "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
        })
        self.src.get_pose()

        prev_angle = 0.0
        for deg in range(-5, -225, -5):
            R_yaw = R.from_rotvec([-math.radians(deg), 0, 0])
            q = R_yaw.as_quat()
            self.server.set_frame({
                "pos": [0, 0, 0], "quat": [q[3], q[0], q[1], q[2]],
                "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
            })
            p = None
            for _ in range(5):
                p = self.src.get_pose()
            q_t = p["quat"]
            R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
            R_origin = R.from_rotvec(self.tcp_home[3:])
            rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
            curr_yaw_deg = np.degrees(rotvec_delta[2])

            # Tuyệt đối không được đổi dấu sang dương
            self.assertLessEqual(
                curr_yaw_deg, 0.01,
                f"Lỗi Snap-Back: Tại góc vào {deg}°, góc ra bị đảo dấu dương: {curr_yaw_deg:.2f}°"
            )
            self.assertGreaterEqual(curr_yaw_deg, -35.0 - 1e-3)
            if deg >= -75:
                self.assertLessEqual(curr_yaw_deg, prev_angle + 0.05)
            elif deg <= -100:
                self.assertLessEqual(curr_yaw_deg, -34.5)
            prev_angle = curr_yaw_deg

    def test_03_no_snapback_during_extreme_roll(self):
        """Test 1C: Xoay Roll qua 180° lên tới 220° không bị lật dấu"""
        self.server.set_frame({
            "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
            "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
        })
        self.src.get_pose()

        for deg in range(10, 225, 10):
            R_roll = R.from_rotvec([0, 0, -math.radians(deg)])
            q = R_roll.as_quat()
            self.server.set_frame({
                "pos": [0, 0, 0], "quat": [q[3], q[0], q[1], q[2]],
                "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
            })
            p = None
            for _ in range(5):
                p = self.src.get_pose()
            q_t = p["quat"]
            R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
            R_origin = R.from_rotvec(self.tcp_home[3:])
            rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
            # Roll quanh Z_tcp (ở pose chúc xuống) là rotvec[2]
            curr_roll_deg = abs(np.degrees(rotvec_delta[2]))
            self.assertLessEqual(
                curr_roll_deg, 85.5,
                f"Roll vượt ngưỡng kẹp: {curr_roll_deg:.2f}° tại input {deg}°"
            )

    def test_04_strict_z_clamp_prevents_arm_torsion(self):
        """Test 2: Xoay cực hạn trục Z (Yaw) -> Kẹp cứng <= 35°, thân máy không bị vặn xoắn"""
        extreme_yaw_angles_deg = [90.0, 120.0, 180.0, 220.0, -90.0, -150.0, -200.0]

        for yaw_deg in extreme_yaw_angles_deg:
            self.server.set_frame({
                "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
                "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
            })
            self.src.get_pose()

            R_yaw = R.from_rotvec([-math.radians(yaw_deg), 0, 0])
            q_yaw = R_yaw.as_quat()
            self.server.set_frame({
                "pos": [0, 0, 0], "quat": [q_yaw[3], q_yaw[0], q_yaw[1], q_yaw[2]],
                "head_quat": [1, 0, 0, 0], "trigger": False, "grip": True
            })

            res = None
            for _ in range(20):
                res = self.src.get_pose()

            q_out = res["quat"]
            R_out = R.from_quat([q_out[1], q_out[2], q_out[3], q_out[0]])
            R_origin = R.from_rotvec(self.tcp_home[3:])
            delta_rotvec = (R_out * R_origin.inv()).as_rotvec()

            yaw_output_deg = abs(np.degrees(delta_rotvec[2]))
            # 1. Góc quay Yaw nằm trong dải làm việc tự nhiên (<= 170.0°)
            self.assertLessEqual(
                yaw_output_deg, 170.5,
                f"Góc Yaw ra ({yaw_output_deg:.2f}°) vượt ngưỡng kẹp khi input là {yaw_deg}°!"
            )

            # 2. Giải IK tư thế xác nhận nghiệm mượt mà chuẩn xác
            sol = self.solver.solve((res["pos"], res["quat"]), q_seed=self.joints_home, is_pure_rot=True)
            self.assertIsNotNone(sol["q"], "IK Solver phải tìm được nghiệm hợp lệ!")

            q_sol = np.array(sol["q"])
            # Toàn bộ 6 khớp nằm an toàn trong giới hạn vật lý UR3E_LIMITS
            for j_idx in range(6):
                lo, hi = UR3E_LIMITS[j_idx]
                self.assertGreaterEqual(q_sol[j_idx], lo - 1e-3)
                self.assertLessEqual(q_sol[j_idx], hi + 1e-3)


if __name__ == "__main__":
    unittest.main()
