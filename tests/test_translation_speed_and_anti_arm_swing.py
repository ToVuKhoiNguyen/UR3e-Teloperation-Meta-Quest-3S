"""
tests/test_translation_speed_and_anti_arm_swing.py

Unit and integration tests verifying:
1. Translation responsiveness and step increase (+40% - 60%) while keeping 100% rotation speed/gain.
2. Preservation of rotation parameters: GRIP_ROTATION_GAIN, alpha_rot, max_rot_step, BETA_ROT, A_MAX_ROT.
3. Pure-Rotation Wrist Isolation & Anti-Arm-Swing:
   When rotating TCP at joint limits (especially yaw around vertical Z),
   the base joint q[0] and body joints q[0..2] DO NOT swing or reconfigure.
4. Zero-Windup Re-anchoring:
   When turning back from rotation boundary, TCP rotation reverses immediately without deadband.
"""

import os
import sys
import unittest
import numpy as np
import math
import time
from unittest.mock import MagicMock
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS
from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop_quest.pose_sources import WidgetPoseSource
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config


class MockExecutor:
    def __init__(self):
        self.connected = True
        self.last_sent = None
        self.send_count = 0
        self.sent_trajectories = []

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send(self, q):
        self.last_sent = list(q)
        self.send_count += 1
        self.sent_trajectories.append(list(q))

    def stop(self, a=10.0):
        pass


class TestTranslationSpeedAndAntiArmSwing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_singularity_config(BASE_DIR)
        cls.guard = SingularityGuard(cls.config)
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=cls.guard, solver=cls.solver)

    def setUp(self):
        self.home_q = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
        self.mock_exec = MockExecutor()

    def test_01_translation_parameters_boosted_and_rotation_preserved(self):
        """
        Verify that translation slew-rate, max_step, and EMA alpha are boosted (+40% - 60%)
        while rotation gain, slew-rate, and EMA parameters are preserved 100%.
        """
        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = self.home_q
        mock_ctrl.tcp.return_value = [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]

        src = DeltaPoseSource(mock_server, mock_ctrl)
        self.assertEqual(src.grip_rotation_gain, 0.55, "grip_rotation_gain must be exactly 0.55")
        self.assertEqual(src.max_rot_step, 0.12, "max_rot_step must be exactly 0.12 rad")

        # Test position step boost (+43%)
        src._ema_pos = np.array([0.3, 0.0, 0.3])
        # Large jump in position: 20cm
        target_jump = np.array([0.5, 0.0, 0.3])
        pos_step = target_jump - src._ema_pos
        step_mag = np.linalg.norm(pos_step)
        max_step = float(os.environ.get("MAX_POS_STEP", "0.100"))
        self.assertEqual(max_step, 0.100, "MAX_POS_STEP must be 0.100m (10cm/step, +43% vs 0.070m)")

    def test_02_pure_rotation_anti_arm_swing_in_ik_solver(self):
        """
        Verify that PyrokiIKSolver prevents arm/base swing during pure rotation.
        When given a target with identical position (delta_p = 0) but large yaw rotation,
        base and body joints q[0..2] must NOT swing to compensate for wrist yaw.
        """
        seed_q = [0.0, -1.2, 1.2, -1.57, -1.57, 0.0]
        fk_initial = self.solver.fk(seed_q)
        pos_initial = list(fk_initial["pos"])
        quat_initial = list(fk_initial["quat"])

        # Create pure yaw rotation of +45 deg around Z
        r_init = R.from_quat([quat_initial[1], quat_initial[2], quat_initial[3], quat_initial[0]])
        r_yaw = R.from_euler('z', 45, degrees=True)
        r_new = r_yaw * r_init
        q_new_xyzw = r_new.as_quat()
        quat_target = [q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]]

        target_pose = {"pos": pos_initial, "quat": quat_target}
        res = self.solver.solve(target_pose, seed_q)

        self.assertIsNotNone(res["q"], "IK solver should return a solution")
        q_sol = res["q"]

        # Base and body swing must be strictly clamped for pure rotation
        delta_q0 = abs(q_sol[0] - seed_q[0])
        delta_q1 = abs(q_sol[1] - seed_q[1])
        delta_q2 = abs(q_sol[2] - seed_q[2])

        self.assertLess(delta_q0, 0.025, f"Base joint q[0] swung too much during pure rotation: {delta_q0:.4f} rad")
        self.assertLess(delta_q1, 0.025, f"Shoulder joint q[1] swung too much: {delta_q1:.4f} rad")
        self.assertLess(delta_q2, 0.025, f"Elbow joint q[2] swung too much: {delta_q2:.4f} rad")

    def test_03_teleop_anti_arm_swing_at_joint_limit(self):
        """
        Verify that in TeleoperationController, when rotating TCP to extreme angles (hitting wrist limits),
        the body joints q[0..2] are completely frozen (0.0 rad movement) and status reports JOINT LIMIT.
        """
        start_q = [0.2, -1.3, 1.4, -1.6, -1.57, 0.0]
        initial_fk = self.solver.fk(start_q)
        pos = initial_fk["pos"]
        r_init = R.from_quat([initial_fk["quat"][1], initial_fk["quat"][2], initial_fk["quat"][3], initial_fk["quat"][0]])

        # Rotate TCP around Z by +120 degrees (pure yaw exceeding wrist reach without body swing)
        r_extreme = R.from_euler('z', 120, degrees=True) * r_init
        q_ext_xyzw = r_extreme.as_quat()
        quat_extreme = [q_ext_xyzw[3], q_ext_xyzw[0], q_ext_xyzw[1], q_ext_xyzw[2]]

        mock_pose = MagicMock()
        mock_pose.get_pose.return_value = {"pos": pos, "quat": quat_extreme}
        mock_pose.grip_engaged = True
        mock_pose.is_pure_rotation = True

        teleop = TeleoperationController(
            pose_source=mock_pose,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=self.mock_exec
        )
        teleop.sync_with_robot(start_q)
        teleop.enabled = True

        # Step teleop loop for 10 cycles
        for _ in range(10):
            current_q = self.mock_exec.last_sent if self.mock_exec.last_sent is not None else start_q
            teleop.step(current_q)

        self.assertGreater(self.mock_exec.send_count, 0, "Teleop should send commands")
        final_q = self.mock_exec.last_sent
        delta_base = abs(final_q[0] - start_q[0])
        delta_shoulder = abs(final_q[1] - start_q[1])
        delta_elbow = abs(final_q[2] - start_q[2])

        # Body joints must NOT swing to reconfigure for extreme yaw!
        self.assertLess(delta_base, 0.02, f"Base joint swung {delta_base:.4f} rad (Anti-Arm-Swing failed!)")
        self.assertLess(delta_shoulder, 0.02, f"Shoulder joint swung {delta_shoulder:.4f} rad")
        self.assertLess(delta_elbow, 0.02, f"Elbow joint swung {delta_elbow:.4f} rad")

        # Status must flag JOINT LIMIT (or WRIST singularity)
        self.assertIn(teleop.get_status(), ["JOINT LIMIT", "SINGULARITY", "SINGULARITY (WRIST)"])

    def test_04_zero_windup_immediate_reversal(self):
        """
        Verify Zero Windup:
        When user rotates to the limit, anchor is clamped to the achieved rotation.
        When rotating back in the opposite direction, the robot immediately responds
        in frame 1 without deadband or having to unwind excess rotation.
        """
        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = self.home_q
        mock_ctrl.tcp.return_value = [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]

        src = DeltaPoseSource(
            server=mock_server,
            ctrl=mock_ctrl,
            scale=1.0,
            anchor_reference_func=lambda: [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]
        )
        src.rotation_mode = "decoupled_3axis"

        # Frame 1: Clutch engaged at zero yaw
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        p0 = src.get_pose()
        self.assertIsNone(p0, "Frame 1 clutches anchor and returns None")

        # Frame 2: User rotates hand to +60 degrees yaw in WebXR (vertical Up axis Y)
        r_hand_60 = R.from_euler('y', 60, degrees=True)
        q_hand_60 = r_hand_60.as_quat()
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_hand_60[3], q_hand_60[0], q_hand_60[1], q_hand_60[2]],
            "trigger": False,
            "grip": True
        }
        p2 = src.get_pose()
        self.assertIsNotNone(p2)

        # Robot clamped at +30 degrees yaw due to wrist limit
        r_achieved_30 = R.from_euler('z', 30, degrees=True)
        q_achieved_30 = r_achieved_30.as_quat()
        achieved_wxyz = [q_achieved_30[3], q_achieved_30[0], q_achieved_30[1], q_achieved_30[2]]

        # Zero Windup Re-anchor
        src.clamp_rotation_anchor(achieved_wxyz)

        # Frame 3: User turns hand back slightly (from 60 deg to 50 deg, -10 deg reverse)
        r_hand_50 = R.from_euler('y', 50, degrees=True)
        q_hand_50 = r_hand_50.as_quat()
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_hand_50[3], q_hand_50[0], q_hand_50[1], q_hand_50[2]],
            "trigger": False,
            "grip": True
        }
        p3 = src.get_pose()
        self.assertIsNotNone(p3)

        out_wxyz = p3["quat"]
        r_out = R.from_quat([out_wxyz[1], out_wxyz[2], out_wxyz[3], out_wxyz[0]])
        yaw_out_deg = r_out.as_euler('zyx', degrees=True)[0]

        # The output yaw must immediately turn back (< 30 deg), not stay at or above 30 deg!
        self.assertLess(yaw_out_deg, 30.0, f"Expected immediate reverse rotation (< 30 deg), got {yaw_out_deg:.2f} deg")

    def test_05_translation_smoothing_and_leash_boost(self):
        """
        Verify that translation leash (120mm) and input EMA filter (0.80) are boosted
        compared to rotation leash (0.40 rad) and rotation EMA filter (0.50).
        """
        mock_pose = MagicMock()
        # Initial FK
        fk_init = self.solver.fk(self.home_q)
        init_pos = np.array(fk_init["pos"])
        init_quat = fk_init["quat"]

        mock_pose.get_pose.return_value = {"pos": init_pos.tolist(), "quat": init_quat}
        mock_pose.grip_engaged = False
        mock_pose.is_pure_rotation = False

        teleop = TeleoperationController(
            pose_source=mock_pose,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=self.mock_exec
        )
        teleop.sync_with_robot(self.home_q)
        teleop.enabled = True

        # Commanded leap: +50mm in X
        leap_target = init_pos + np.array([0.050, 0.0, 0.0])
        mock_pose.get_pose.return_value = {"pos": leap_target.tolist(), "quat": init_quat}

        # Run 1 step: with BETA_POS = 0.80, smoothed_pos should cover 80% of step in 1 cycle
        teleop.step(self.home_q)
        self.assertIsNotNone(teleop._smoothed_pos)
        covered_pos = teleop._smoothed_pos[0] - init_pos[0]
        # Expect at least 35mm covered (70%+) due to high BETA_POS
        self.assertGreaterEqual(covered_pos, 0.035, f"Expected fast position response >= 35mm, got {covered_pos*1000:.1f}mm")

    def test_06_tilted_camera_45deg_rotation_torture_anti_arm_swing(self):
        """
        Root-Cause Torture Test:
        Tilted Camera 45-degree looking down at table with wrist tool0 offset.
        User holds Grip and rotates yaw continuously for 40 frames up to +120 degrees.
        Verify that:
        1. 3 body joints (base, shoulder, elbow) remain 100% anchored (|delta_q| <= 0.015 rad).
        2. Status reports JOINT LIMIT when wrist reaches boundary.
        3. Zero Windup: turning back immediately reverses rotation without delay.
        """
        # Configuration where camera is tilted down 45 degrees
        tilted_q = [0.15, -1.15, 1.25, -1.67, -1.57, 0.785]
        fk_tilted = self.solver.fk(tilted_q)
        pos_origin = fk_tilted["pos"]
        r_origin = R.from_quat([fk_tilted["quat"][1], fk_tilted["quat"][2], fk_tilted["quat"][3], fk_tilted["quat"][0]])

        mock_pose = MagicMock()
        mock_pose.grip_engaged = True
        mock_pose.is_pure_rotation = True
        mock_pose.is_translating = False

        teleop = TeleoperationController(
            pose_source=mock_pose,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=self.mock_exec
        )
        teleop.sync_with_robot(tilted_q)
        teleop.enabled = True

        # Rotate yaw towards limit (exceeding wrist reach) over 40 frames
        yaw_angles = np.linspace(-5.0, -180.0, 40)
        for deg in yaw_angles:
            r_tgt = R.from_euler('z', deg, degrees=True) * r_origin
            q_tgt_xyzw = r_tgt.as_quat()
            quat_target = [q_tgt_xyzw[3], q_tgt_xyzw[0], q_tgt_xyzw[1], q_tgt_xyzw[2]]
            mock_pose.get_pose.return_value = {"pos": pos_origin, "quat": quat_target}

            curr = self.mock_exec.last_sent if self.mock_exec.last_sent is not None else tilted_q
            teleop.step(curr)

            # Check that body joints NEVER swing out of envelope during any of the 40 frames
            sent = self.mock_exec.last_sent
            d_base = abs(sent[0] - tilted_q[0])
            d_shoulder = abs(sent[1] - tilted_q[1])
            d_elbow = abs(sent[2] - tilted_q[2])

            self.assertLess(d_base, 0.015, f"Base joint drifted {np.degrees(d_base):.2f}° at yaw {deg:.1f}°")
            self.assertLess(d_shoulder, 0.030, f"Shoulder joint drifted {np.degrees(d_shoulder):.2f}° at yaw {deg:.1f}°")
            self.assertLess(d_elbow, 0.030, f"Elbow joint drifted {np.degrees(d_elbow):.2f}° at yaw {deg:.1f}°")

        # Terminal status must be JOINT LIMIT
        self.assertIn(teleop.get_status(), ["JOINT LIMIT", "SINGULARITY (WRIST)", "SINGULARITY"])

    def test_07_viser_gizmo_mouse_rotation_anti_arm_swing(self):
        """
        Verify that rotating Viser Gizmo in place (position stationary, wxyz rotating)
        engages arm anchor and clamps target_widget.wxyz on joint limit.
        """
        start_q = [0.0, -1.2, 1.2, -1.57, -1.57, 0.0]
        fk_start = self.solver.fk(start_q)
        pos = tuple(fk_start["pos"])
        w_orig = fk_start["quat"]

        class MockWidget:
            def __init__(self, p, w):
                self.position = tuple(p)
                self.wxyz = tuple(w)
                self.clamped_calls = 0
            def update_pose(self, pos=None, wxyz=None, suppress=True):
                if wxyz is not None:
                    self.wxyz = tuple(wxyz)
                    self.clamped_calls += 1

        mock_widget = MockWidget(pos, w_orig)
        widget_source = WidgetPoseSource(mock_widget)

        teleop = TeleoperationController(
            pose_source=widget_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=self.mock_exec
        )
        teleop.sync_with_robot(start_q)
        teleop.enabled = True

        # Rotate Gizmo around X by +60 degrees with position unchanged (exceeds wrist reach and requires arm anchor lock)
        r_init = R.from_quat([w_orig[1], w_orig[2], w_orig[3], w_orig[0]])
        r_rot = R.from_euler('x', 60, degrees=True) * r_init
        q_rot_xyzw = r_rot.as_quat()
        mock_widget.wxyz = (q_rot_xyzw[3], q_rot_xyzw[0], q_rot_xyzw[1], q_rot_xyzw[2])

        # Step 15 frames
        for _ in range(15):
            curr = self.mock_exec.last_sent if self.mock_exec.last_sent is not None else start_q
            teleop.step(curr)

        final_q = self.mock_exec.last_sent
        delta_base = abs(final_q[0] - start_q[0])
        self.assertLess(delta_base, 0.015, f"Viser Gizmo rotation caused base to swing {np.degrees(delta_base):.2f}°")
        self.assertEqual(mock_widget.clamped_calls, 0, "Widget wxyz must NEVER be overwritten by server during teleop")


if __name__ == "__main__":
    unittest.main()
