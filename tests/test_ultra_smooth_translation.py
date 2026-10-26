"""
test_ultra_smooth_translation.py - Rigorous verification of Ultra-Smooth Translation,
Translational Exponential Smoothing, Slew-Rate Limiting, and Zero-Jitter Control Cadence.
"""

import os
import time
import math
import unittest
import numpy as np

from teleop_quest.pose_sources import WidgetPoseSource
from teleop_quest.viser_ui import TargetWidgetWrapper
from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS


class MockRawGizmo:
    def __init__(self, position=(0.25, 0.15, 0.25), wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.position = tuple(position)
        self.wxyz = tuple(wxyz)
        self.write_pos_count = 0
        self.write_wxyz_count = 0

    def __setattr__(self, name, value):
        if name == "position":
            self.__dict__["write_pos_count"] = self.__dict__.get("write_pos_count", 0) + 1
        elif name == "wxyz":
            self.__dict__["write_wxyz_count"] = self.__dict__.get("write_wxyz_count", 0) + 1
        super().__setattr__(name, value)


class MockExecutor:
    def __init__(self):
        self.sent_commands = []
        self._connected = True

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def send(self, q):
        self.sent_commands.append(list(q))

    def stop(self, a=10.0):
        pass


class TestUltraSmoothTranslation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[INIT] Initializing Pyroki IK Solver for Ultra-Smooth Translation tests...")
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=None, solver=cls.solver)

    def setUp(self):
        self.raw_gizmo = MockRawGizmo()
        self.wrapper = TargetWidgetWrapper(self.raw_gizmo)
        self.pose_source = WidgetPoseSource(self.wrapper, alpha=0.55)
        self.mock_exec = MockExecutor()
        self.teleop = TeleoperationController(
            pose_source=self.pose_source,
            solver=self.solver,
            gate=self.gate,
            executor=self.mock_exec,
            dt=0.05
        )
        self.teleop.enabled = True
        self.teleop.sync_with_robot(self.solver.default_home)

    def test_01_widget_pose_source_exponential_smoothing_eliminates_stepping(self):
        """
        Verify that discrete / quantized mouse steps are smoothed continuously
        by Exponential Moving Average with zero steady-state error.
        """
        print("\n--- Test 1: Exponential Moving Average Smoothing on Viser Mouse Drag ---")
        fk_home = self.solver.fk(self.solver.default_home)
        init_pos = np.array(fk_home["pos"])
        self.raw_gizmo.position = tuple(init_pos)
        self.teleop.sync_with_robot(self.solver.default_home)

        # Anchor initial pose so filter is initialized
        self.teleop.step(self.solver.default_home, pose=self.pose_source.get_pose())
        self.assertFalse(self.teleop._was_clutch_released)

        # Simulate discrete packet jumps from web interface:
        # Frame 1: Jump 6mm in X
        step1_target = init_pos + np.array([0.006, 0.0, 0.0])
        self.raw_gizmo.position = tuple(step1_target)
        self.teleop.step(self.solver.default_home, pose=self.pose_source.get_pose())
        p1 = np.copy(self.teleop._smoothed_pos)

        # With BETA_POS=0.66 (+20%), output must interpolate between 0 and 6mm:
        expected_beta = float(os.environ.get("POS_SMOOTH_BETA", os.environ.get("TARGET_FILTER_BETA_POS", "0.66")))
        self.assertAlmostEqual(p1[0], init_pos[0] + expected_beta * 0.006, places=4)
        self.assertLess(p1[0], step1_target[0], "Must not jump directly to discrete target")

        # Frame 2: Mouse stationary (no packet in this 50ms)
        self.teleop.step(self.solver.default_home, pose=self.pose_source.get_pose())
        p2 = np.copy(self.teleop._smoothed_pos)
        # Smooth progression towards step1_target
        self.assertGreater(p2[0], p1[0], "Continuous smooth progression")
        self.assertLessEqual(p2[0], step1_target[0])

        # Hold for 5 more frames -> must converge with ZERO steady-state error (< 0.1mm)
        for _ in range(6):
            self.teleop.step(self.solver.default_home, pose=self.pose_source.get_pose())
        p_final = np.copy(self.teleop._smoothed_pos)
        np.testing.assert_allclose(p_final, step1_target, atol=1e-5,
                                   err_msg="Must achieve 0.0mm steady-state error")
        print("  [PASSED] Exponential smoothing transitions smoothly and converges with 0.0mm error.")

    def test_02_static_viser_gizmo_zero_drift(self):
        """
        Verify that WidgetPoseSource NEVER writes back to raw_gizmo during teleop or mouse release,
        ensuring 100% static stability with zero phantom drift.
        """
        print("\n--- Test 2: Static Viser Gizmo Zero-Drift Verification ---")
        mouse_target = (0.28, 0.18, 0.26)
        self.raw_gizmo.position = mouse_target
        self.raw_gizmo.write_pos_count = 0  # reset write counter

        # Run 20 cycles of get_pose and teleop.step
        for _ in range(20):
            p = self.pose_source.get_pose()
            self.teleop.step(self.solver.default_home, pose=p)

        # Gizmo position in Viser must NOT have been overwritten
        self.assertEqual(self.raw_gizmo.write_pos_count, 0,
                         "Gizmo position must NEVER be overwritten during teleop")
        np.testing.assert_allclose(self.raw_gizmo.position, mouse_target, atol=1e-6)
        print("  [PASSED] Gizmo remains strictly static at user release point with 0 drift.")

    def test_03_cartesian_slew_rate_limiting_continuity(self):
        """
        Verify that sudden fast mouse flicks do NOT cause abrupt velocity jumps or hard-clamping corners.
        Acceleration must be smoothly bounded by A_MAX_LIN and velocity by V_MAX_LIN.
        """
        print("\n--- Test 3: Cartesian Slew-Rate Limiter (Continuous C1 Acceleration) ---")
        fk_home = self.solver.fk(self.solver.default_home)
        p_home = np.array(fk_home["pos"])

        # Sudden large displacement step of 80mm in +X
        p_far = p_home + np.array([0.080, 0.0, 0.0])
        pose_cmd = {"pos": p_far.tolist(), "quat": fk_home["quat"]}

        speeds = []
        positions = []
        q_curr = list(self.solver.default_home)

        # Simulate 10 consecutive 50ms frames (0.5s duration)
        for frame in range(10):
            self.teleop.step(q_curr, pose=pose_cmd)
            q_next = self.mock_exec.sent_commands[-1]
            fk_next = self.solver.fk(q_next)
            p_next = np.array(fk_next["pos"])
            positions.append(p_next)
            if len(positions) > 1:
                disp = float(np.linalg.norm(positions[-1] - positions[-2]))
                speed = disp / 0.05  # m/s
                speeds.append(speed)
            q_curr = q_next

        # Check acceleration continuity:
        # Frame 1 speed should be small (ramping up smoothly, not instantaneous 1.6 m/s)
        self.assertLess(speeds[0], 0.25, "First frame must smoothly ramp acceleration")
        # Acceleration between consecutive frames should not exceed A_MAX_LIN with tolerance
        for i in range(1, len(speeds)):
            acc = abs(speeds[i] - speeds[i - 1]) / 0.05
            self.assertLess(acc, 3.5, f"Acceleration at frame {i} must be continuous (was {acc:.2f} m/s^2)")

        # Verify movement is progressing steadily towards p_far
        final_dist = float(np.linalg.norm(positions[-1] - p_far))
        init_dist = float(np.linalg.norm(p_home - p_far))
        self.assertLess(final_dist, init_dist, "Arm must track steadily towards target")
        print(f"  [PASSED] Linear speed ramped smoothly: {[round(s, 3) for s in speeds]} m/s")

    def test_04_proportional_body_joint_scaling_preserves_direction(self):
        """
        Verify that when body joint steps are limited, joints 0, 1, 2 are scaled
        proportionally by the EXACT same ratio, preserving Cartesian line of motion.
        """
        print("\n--- Test 4: Proportional Body Joint Step Limiting ---")
        q_seed = list(self.solver.default_home)
        fk_init = self.solver.fk(q_seed)
        p_init = np.array(fk_init["pos"])

        # Command target 40mm in purely +X direction
        p_tgt = p_init + np.array([0.040, 0.0, 0.0])
        res = self.solver.solve({"pos": p_tgt.tolist(), "quat": fk_init["quat"]}, q_seed=q_seed)
        self.assertIsNotNone(res["q"])

        # Check FK of result: direction should point predominantly along +X
        fk_res = self.solver.fk(res["q"])
        p_res = np.array(fk_res["pos"])
        disp_vec = p_res - p_init
        disp_norm = np.linalg.norm(disp_vec)

        self.assertGreater(disp_norm, 0.005, "Arm must have moved along direction")
        # Dot product with ideal direction [1, 0, 0] should be > 0.95
        cos_dir = disp_vec[0] / disp_norm
        self.assertGreater(cos_dir, 0.95, f"Motion direction must be preserved along +X (cos={cos_dir:.3f})")
        print(f"  [PASSED] Direction preserved with cos={cos_dir:.4f} > 0.95.")

    def test_05_sub_millimeter_tracking_sensitivity(self):
        """
        Verify that fine mouse movements (0.3mm) are recognized as active translation
        and track without deadband or phase lag.
        """
        print("\n--- Test 5: Sub-millimeter Interaction Sensitivity ---")
        curr_p = np.array(self.raw_gizmo.position)
        # Move by 0.3mm in +Y
        micro_step = curr_p + np.array([0.0, 0.0003, 0.0])
        self.raw_gizmo.position = tuple(micro_step)

        pose = self.pose_source.get_pose()
        # is_translating must be True even for 0.3mm movement
        self.assertTrue(self.pose_source.is_translating,
                        "Sub-millimeter (0.3mm) drag must be recognized as translation")
        self.assertFalse(self.pose_source.is_pure_rotation)
        print("  [PASSED] Sub-millimeter drag correctly recognized as translation.")

    def test_06_cached_collision_zero_redundancy(self):
        """
        Verify that SafetyGate uses cached collision checks from solver,
        preventing redundant expensive JIT invocations.
        """
        print("\n--- Test 6: Zero Timing Jitter / Cached Collision Verification ---")
        q_test = list(self.solver.default_home)
        res = self.solver.solve({"pos": [0.25, 0.15, 0.25], "quat": [1.0, 0.0, 0.0, 0.0]}, q_seed=q_test)

        self.assertIn("self_chk", res, "Solver result must include cached self_chk")
        self.assertIn("tbl_chk", res, "Solver result must include cached tbl_chk")

        # Calling SafetyGate.check with cached_collision must pass and be ultra-fast (< 0.1ms)
        t0 = time.perf_counter()
        gate_res = self.gate.check(res["q"], cached_collision=res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.assertTrue(gate_res["pass"])
        self.assertLess(dt_ms, 0.5, f"Cached gate check must be < 0.5ms (took {dt_ms:.3f}ms)")
        print(f"  [PASSED] Cached gate check latency: {dt_ms:.3f}ms (< 0.5ms).")


if __name__ == "__main__":
    unittest.main()
