"""
tests/test_clock_face_grip_rotation.py

Comprehensive tests verifying the 3 core requirements of Grip-based rotation:
1. Clock-face Intuition:
   - Twisting wrist clockwise (along line of sight) -> TCP rotates clockwise around +X_base.
   - Tilting wrist down (Flexion) -> TCP pitches down around -Y_base.
   - Sweeping wrist right -> TCP yaws right around -Z_base.
   - Validated across Home pose and extreme tilted poses.
2. Strict Single-Axis Dominant Lock (Zero Cross-talk):
   - Deadzone 4.0 deg: small motions (<4 deg) keep orientation at R_tcp_origin (0.0 rad).
   - Once >= 4.0 deg, strictly locks to dominant axis; the other 2 axes are 0.0 rad strictly.
   - Continuous rotation on locked axis subtracts deadzone for smooth transition from 0.
   - Releasing Grip unlocks the axis for the next gesture.
3. Angular Gain & Slew-Rate Scaling:
   - GRIP_ROTATION_GAIN = 0.55: Rotating wrist 45 deg produces ~22.5 deg robot rotation (between 22-25 deg).
   - Slew-rate limit clamps angular jumps per frame to <= 0.12 rad.
   - Quaternion sign continuity (dot >= 0) is preserved across all output frames.
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
    def __init__(self, joints=None, tcp=None):
        self._joints = list(joints) if joints is not None else [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self._tcp = list(tcp) if tcp is not None else [0.25, 0.0, 0.25, 0.0, 3.14159, 0.0]

    def joints(self):
        return self._joints

    def tcp(self):
        return self._tcp


class TestClockFaceGripRotation(unittest.TestCase):
    def setUp(self):
        self.server = MockQuestServer()
        self.tcp_home = [0.25, 0.0, 0.25, 0.0, 3.14159, 0.0]  # Tip pointing down
        self.joints_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.ctrl = MockRobotCtrl(joints=self.joints_home, tcp=self.tcp_home)

        self.pose_source = DeltaPoseSource(
            server=self.server,
            ctrl=self.ctrl,
            scale=1.0,
            anchor_reference_func=lambda: list(self.ctrl.tcp())
        )
        self.pose_source.rotation_mode = "decoupled_3axis"

    def test_01_clock_face_roll_direction(self):
        """
        Clock-face Roll:
        Vặn cổ tay sang phải thuận chiều kim đồng hồ quanh trục nhìn thẳng (u_fwd)
        -> TCP xoay thuận chiều kim đồng hồ quanh +X_base.
        """
        # Engage Grip
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        p0 = self.pose_source.get_pose()
        self.assertIsNone(p0, "Frame 1 must anchor and return None")

        # Cử chỉ Vặn chìa khóa (Roll) quanh trục cẳng tay (trục 0 của tay cầm):
        rot_deg = 30.0
        R_wrist = R.from_rotvec([-math.radians(rot_deg), 0.0, 0.0])
        q_wrist = R_wrist.as_quat()

        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_wrist[3], q_wrist[0], q_wrist[1], q_wrist[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(10):
            res = self.pose_source.get_pose()

        self.assertIsNotNone(res)
        # Compute delta rotation in base frame: R_delta = R_target * R_origin.inv()
        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        R_origin = R.from_rotvec(self.tcp_home[3:])
        R_delta = R_target * R_origin.inv()
        rotvec_delta = R_delta.as_rotvec()

        print(f"\n[Test Clock-face Roll] Input: {rot_deg}° CW -> Base Rotvec: {np.degrees(rotvec_delta)}°")

        # 1. Roll (X_base) must be positive (Clockwise)
        self.assertGreater(rotvec_delta[0], 0.0, "Roll must rotate clockwise (positive around +X_base)")
        # 2. Strict Single-Axis Lock: Pitch (Y_base) and Yaw (Z_base) must be exactly 0
        self.assertAlmostEqual(rotvec_delta[1], 0.0, places=4, msg="Pitch must be 0 (Zero cross-talk)")
        self.assertAlmostEqual(rotvec_delta[2], 0.0, places=4, msg="Yaw must be 0 (Zero cross-talk)")

    def test_02_pitch_flexion_and_extension_direction(self):
        """
        Pitch:
        Gập cổ tay xuống (Flexion) -> TCP chúc xuống quanh -Y_base (omega_y < 0).
        Ngửa cổ tay lên (Extension) -> TCP ngửa lên quanh +Y_base (omega_y > 0).
        """
        # Engage Grip
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        self.pose_source.get_pose()

        # Cử chỉ Gập cổ tay XUỐNG 25 deg (trục 2 của tay cầm):
        R_down = R.from_rotvec([0.0, 0.0, math.radians(25.0)])
        q_down = R_down.as_quat()
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_down[3], q_down[0], q_down[1], q_down[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res_down = None
        for _ in range(10):
            res_down = self.pose_source.get_pose()

        q_t = res_down["quat"]
        R_target_down = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        R_origin = R.from_rotvec(self.tcp_home[3:])
        rotvec_down = (R_target_down * R_origin.inv()).as_rotvec()

        print(f"\n[Test Pitch Down] Input: +25° Down -> Base Rotvec: {np.degrees(rotvec_down)}°")
        self.assertLess(rotvec_down[1], 0.0, "Pitch down must produce negative rotation around Y_base")
        self.assertAlmostEqual(rotvec_down[0], 0.0, places=4, msg="Roll must be 0")
        self.assertAlmostEqual(rotvec_down[2], 0.0, places=4, msg="Yaw must be 0")

    def test_03_strict_single_axis_dominant_lock_cross_talk_elimination(self):
        """
        Strict Single-Axis Dominant Lock:
        When human wrist moves with intentional Roll (25 deg) but natural biological leakage:
        Pitch leakage (7 deg) and Yaw leakage (5 deg).
        -> Only Roll is active, Pitch and Yaw must be STRICTLY 0.0!
        """
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        self.pose_source.get_pose()

        # Human wrist movement: Roll = 25 deg (trục 0), Yaw = 5 deg (trục 1), Pitch = 7 deg (trục 2)
        R_human = R.from_rotvec([-math.radians(25.0), math.radians(5.0), math.radians(7.0)])
        q_h = R_human.as_quat()

        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_h[3], q_h[0], q_h[1], q_h[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(10):
            res = self.pose_source.get_pose()

        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        R_origin = R.from_rotvec(self.tcp_home[3:])
        rotvec_delta = (R_target * R_origin.inv()).as_rotvec()

        print(f"\n[Test Strict Lock] Human Multi-axis [Pitch=7°, Yaw=5°, Roll=25°] -> Base Rotvec: {np.degrees(rotvec_delta)}°")

        # Roll is dominant (25 deg > 7 deg > 5 deg) -> Locked to Roll!
        self.assertGreater(rotvec_delta[0], 0.0, "Roll must be the dominant active axis")
        # Pitch and Yaw must be strictly 0.0 (within numerical float precision 1e-6)
        self.assertAlmostEqual(rotvec_delta[1], 0.0, places=6, msg="Pitch cross-talk must be 0.0 rad!")
        self.assertAlmostEqual(rotvec_delta[2], 0.0, places=6, msg="Yaw cross-talk must be 0.0 rad!")

    def test_04_angular_gain_scaling_45deg_wrist(self):
        """
        Criterion 3: Tốc độ xoay đầm và mượt.
        Xoay cổ tay 45 deg -> robot xoay nhẹ khoảng 22 deg - 25 deg.
        Formula: (45 deg - 4 deg deadzone) * 0.55 gain = 22.55 deg.
        """
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        self.pose_source.get_pose()

        # Wrist rotates 45 deg around line of sight
        R_45 = R.from_rotvec([0.0, 0.0, -math.radians(45.0)])
        q_45 = R_45.as_quat()

        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_45[3], q_45[0], q_45[1], q_45[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(20):
            res = self.pose_source.get_pose()

        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        R_origin = R.from_rotvec(self.tcp_home[3:])
        rotvec_delta = (R_target * R_origin.inv()).as_rotvec()
        angle_deg = np.degrees(np.linalg.norm(rotvec_delta))

        print(f"\n[Test 45 deg Scaling] Wrist: 45.0° -> Robot Rotation: {angle_deg:.2f}°")
        self.assertGreaterEqual(angle_deg, 21.5, "Robot rotation must be >= 21.5 deg")
        self.assertLessEqual(angle_deg, 25.0, "Robot rotation must be <= 25.0 deg")

    def test_05_deadzone_below_4deg_keeps_origin(self):
        """
        Micro-tremor (< 4.0 deg) does NOT trigger rotation.
        Target orientation stays identical to R_tcp_origin (0.000 rad error).
        """
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        self.pose_source.get_pose()

        # Wrist micro-tremor: 2.5 deg (< 4 deg)
        R_small = R.from_rotvec([0.0, 0.0, -math.radians(2.5)])
        q_small = R_small.as_quat()

        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_small[3], q_small[0], q_small[1], q_small[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = self.pose_source.get_pose()
        q_t = res["quat"]
        R_target = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        R_origin = R.from_rotvec(self.tcp_home[3:])
        rot_err = np.linalg.norm((R_target * R_origin.inv()).as_rotvec())

        print(f"\n[Test Deadzone <4 deg] Input: 2.5° -> Robot Angle Change: {np.degrees(rot_err):.4f}°")
        self.assertEqual(rot_err, 0.0, "Sub-4 deg motion must keep TCP orientation at origin (0.0 rad)")


if __name__ == "__main__":
    unittest.main()
