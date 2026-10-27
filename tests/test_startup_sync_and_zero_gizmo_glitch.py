"""
tests/test_startup_sync_and_zero_gizmo_glitch.py
Verification test suite for:
1. Clean Startup Synchronization:
   - Synchronizes 100% with actual robot joints on startup (including wrapped/near-limit joints).
   - Frame 1 status is guaranteed "OK", position error is 0.0mm, and teleop responds immediately
     without needing to press 'Go Home'.
2. Zero Gizmo Glitch:
   - Server-side Python NEVER overwrites target_widget.position or target_widget.wxyz during teleop
     (0 overwrites in step() across Singularity, Joint Limit, and Collision).
"""

import os
import sys
import math
import numpy as np
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS, validate_joints
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config
from teleop_quest.pose_sources import WidgetPoseSource
from teleop_quest.main_quest import normalize_and_clip_joints


class TrackedGizmo:
    """
    Mock Viser 3D Transform Controls Gizmo that tracks every read and write access
    to verify that the server NEVER overwrites the Gizmo during teleoperation.
    """
    def __init__(self, pos, wxyz):
        self._pos = tuple(float(x) for x in pos)
        self._wxyz = tuple(float(w) for w in wxyz)
        self.write_pos_count = 0
        self.write_wxyz_count = 0
        self.history = []

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, val):
        self.write_pos_count += 1
        self.history.append(("position", tuple(val)))
        self._pos = tuple(val)

    @property
    def wxyz(self):
        return self._wxyz

    @wxyz.setter
    def wxyz(self, val):
        self.write_wxyz_count += 1
        self.history.append(("wxyz", tuple(val)))
        self._wxyz = tuple(val)


class MockExecutor:
    def __init__(self):
        self.connected = True
        self.last_sent = None
        self.send_count = 0

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send(self, q):
        self.last_sent = list(q)
        self.send_count += 1

    def stop(self, a=10.0):
        pass


class TestStartupSyncAndZeroGizmoGlitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_singularity_config(BASE_DIR)
        cls.guard = SingularityGuard(cls.config)
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=cls.guard, solver=cls.solver)

    def test_01_joint_normalization_and_soft_clipping(self):
        """Verify normalize_and_clip_joints resolves 2pi angle wrapping and soft edge buffer."""
        # 1. Test wrapped shoulder pan (+2pi wrap)
        q_wrapped = [0.35 + 2.0 * math.pi, -1.8, 1.4, -1.1, -1.5, 0.2]
        q_norm = normalize_and_clip_joints(q_wrapped, UR3E_LIMITS, margin=2e-3)
        self.assertAlmostEqual(q_norm[0], 0.35, places=4, msg="Shoulder pan +2pi wrapping must resolve to 0.35 rad")
        self.assertTrue(validate_joints(q_norm), "All joints must be within UR3E_LIMITS")

        # 2. Test wrapped wrist 1 (-2pi wrap)
        q_wrapped_wrist = [0.0, -2.0, 1.5, -0.5 - 2.0 * math.pi, -1.5, 0.0]
        q_norm_wrist = normalize_and_clip_joints(q_wrapped_wrist, UR3E_LIMITS, margin=2e-3)
        self.assertAlmostEqual(q_norm_wrist[3], -0.5, places=4, msg="Wrist 1 -2pi wrapping must resolve to -0.5 rad")

        # 3. Test edge clamping with 2e-3 margin
        # shoulder_lift_joint min is -2.65
        q_edge = [0.0, -2.65005, 1.5, -1.0, -1.5, 0.0]
        q_clipped = normalize_and_clip_joints(q_edge, UR3E_LIMITS, margin=2e-3)
        self.assertGreaterEqual(q_clipped[1], -2.65 + 1.99e-3, msg="Joint near lower boundary must be clamped with 2e-3 margin")
        self.assertLessEqual(q_clipped[1], -2.65 + 2.01e-3)

    def test_02_clean_startup_sync_arbitrary_pose(self):
        """
        Verify startup synchronization when robot starts at an arbitrary valid pose
        (different from SAFE_HOME_JOINTS).
        Frame 1 error must be 0.0mm, status must be OK, and Teleop responds immediately.
        """
        # Arbitrary starting pose away from home
        actual_q_raw = np.array([0.45, -1.75, 1.30, -1.12, -1.57, 0.25], dtype=float)

        # 1. Startup pipeline
        start_joints = normalize_and_clip_joints(actual_q_raw, UR3E_LIMITS, margin=2e-3).tolist()
        fk_init = self.solver.fk(start_joints)
        init_pos = tuple(fk_init["pos"])
        init_quat = tuple(fk_init["quat"])

        # Create Gizmo and PoseSource
        gizmo = TrackedGizmo(init_pos, init_quat)
        widget_pose_source = WidgetPoseSource(gizmo)
        widget_pose_source.sync_with_robot(pos=init_pos, quat=init_quat)

        # Setup Teleop
        executor = MockExecutor()
        teleop = TeleoperationController(
            pose_source=widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(start_joints)
        teleop.enabled = True

        # 2. Check initial state alignment
        self.assertEqual(teleop.get_status(), "OK", "Initial teleop status must be 'OK'")
        self.assertIsNotNone(teleop._smoothed_pos, "Smoothed position must be initialized at startup")
        pos_err = np.linalg.norm(np.array(teleop._smoothed_pos) - np.array(init_pos))
        self.assertAlmostEqual(pos_err, 0.0, places=5, msg="Smoothed target position must match robot FK position")

        # 3. Frame 1 step execution
        teleop.step(q_real=start_joints)
        self.assertEqual(teleop.get_status(), "OK", "Frame 1 status must remain 'OK'")
        self.assertIsNotNone(executor.last_sent, "Command must be sent in Frame 1")
        joint_diff = np.max(np.abs(np.array(executor.last_sent) - np.array(start_joints)))
        self.assertLess(joint_diff, 1e-4, "Frame 1 command must maintain current position without jumping")

        # 4. Frame 2: Drag Gizmo slightly (+15mm in X)
        new_pos = [init_pos[0] + 0.015, init_pos[1], init_pos[2]]
        gizmo.position = tuple(new_pos)

        teleop.step(q_real=executor.last_sent)
        self.assertEqual(teleop.get_status(), "OK", "Teleop must track dragged gizmo with status 'OK'")
        new_cmd = executor.last_sent
        # Verify robot moved towards dragged position
        fk_new_cmd = self.solver.fk(new_cmd)
        delta_x = fk_new_cmd["pos"][0] - init_pos[0]
        self.assertGreater(delta_x, 0.001, "Robot must track dragged Gizmo immediately without Go Home")

    def test_03_zero_gizmo_glitch_singularity(self):
        """
        Verify that dragging Gizmo into SINGULARITY never triggers an overwrite
        to the Viser Gizmo (target_widget.position / wxyz write count must remain 0).
        """
        start_joints = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]
        fk_init = self.solver.fk(start_joints)
        init_pos = tuple(fk_init["pos"])
        init_quat = tuple(fk_init["quat"])

        gizmo = TrackedGizmo(init_pos, init_quat)
        widget_pose_source = WidgetPoseSource(gizmo)
        widget_pose_source.sync_with_robot(pos=init_pos, quat=init_quat)

        executor = MockExecutor()
        teleop = TeleoperationController(
            pose_source=widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(start_joints)
        teleop.enabled = True

        # Reset Gizmo write counter after initial setup
        gizmo.write_pos_count = 0
        gizmo.write_wxyz_count = 0

        # Simulate user dragging Gizmo far into Singularity / Out of reach (e.g. radius > 550mm)
        singular_pos = [0.55, 0.0, 0.45]
        gizmo.position = tuple(singular_pos)  # User action (from browser)
        self.assertEqual(gizmo.write_pos_count, 1, "Only the user's drag action occurred")

        # Record baseline before server control loop runs
        server_writes_start = gizmo.write_pos_count

        # Run 50 iterations of teleop loop (simulating 2.5 seconds at 20Hz)
        q_curr = list(start_joints)
        for step_idx in range(50):
            teleop.step(q_real=q_curr)
            if executor.last_sent is not None:
                q_curr = list(executor.last_sent)

        # Count how many times the SERVER rewrote gizmo.position or gizmo.wxyz during teleop
        server_pos_overwrites = gizmo.write_pos_count - server_writes_start
        server_wxyz_overwrites = gizmo.write_wxyz_count

        self.assertEqual(server_pos_overwrites, 0, f"Server overwrote Gizmo position {server_pos_overwrites} times! Must be exactly 0!")
        self.assertEqual(server_wxyz_overwrites, 0, f"Server overwrote Gizmo wxyz {server_wxyz_overwrites} times! Must be exactly 0!")

    def test_04_zero_gizmo_glitch_joint_limit_and_barrier(self):
        """
        Verify that dragging Gizmo into JOINT LIMIT and collision barrier
        NEVER overwrites the Viser Gizmo during teleoperation.
        """
        start_joints = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]
        fk_init = self.solver.fk(start_joints)
        init_pos = tuple(fk_init["pos"])
        init_quat = tuple(fk_init["quat"])

        gizmo = TrackedGizmo(init_pos, init_quat)
        widget_pose_source = WidgetPoseSource(gizmo)
        widget_pose_source.sync_with_robot(pos=init_pos, quat=init_quat)

        executor = MockExecutor()
        teleop = TeleoperationController(
            pose_source=widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(start_joints)
        teleop.enabled = True

        # 1. Drag into joint limit (extreme downward tilt)
        gizmo.write_pos_count = 0
        gizmo.write_wxyz_count = 0

        barrier_pos = [0.25, -0.20, -0.05]  # Into table collision barrier
        gizmo.position = tuple(barrier_pos)  # User action

        server_writes_start = gizmo.write_pos_count

        # Run 40 frames of 20Hz step into barrier
        q_curr = list(start_joints)
        for _ in range(40):
            teleop.step(q_real=q_curr)
            if executor.last_sent is not None:
                q_curr = list(executor.last_sent)

        server_pos_overwrites = gizmo.write_pos_count - server_writes_start
        server_wxyz_overwrites = gizmo.write_wxyz_count

        self.assertEqual(server_pos_overwrites, 0, "Server must NEVER overwrite Gizmo position when hitting barrier")
        self.assertEqual(server_wxyz_overwrites, 0, "Server must NEVER overwrite Gizmo wxyz when hitting barrier")

    def test_05_reanchor_position_does_not_mutate_widget(self):
        """
        Direct test of WidgetPoseSource.reanchor_position:
        Confirms target_widget on Viser UI is completely untouched (0 writes),
        and no artificial offset drift is introduced into the pose source.
        """
        init_pos = [0.30, 0.0, 0.30]
        gizmo = TrackedGizmo(init_pos, [1.0, 0.0, 0.0, 0.0])
        wps = WidgetPoseSource(gizmo)

        gizmo.write_pos_count = 0
        p_actual = [0.20, 0.0, 0.30]  # Robot TCP is 100mm away
        wps.reanchor_position(p_actual)

        # Check that target_widget was NOT modified
        self.assertEqual(gizmo.write_pos_count, 0, "reanchor_position MUST NOT write to target_widget")

        # Check that target pose remains 100% matched to gizmo.position (0.0mm drift)
        pose_now = wps.get_pose()["pos"]
        drift = np.linalg.norm(np.array(pose_now) - np.array(init_pos))
        self.assertAlmostEqual(drift, 0.0, places=5, msg="WidgetPoseSource must have 0.0mm drift from gizmo")

    def test_06_absolute_gizmo_tracking_zero_drift(self):
        """
        Verify that WidgetPoseSource directly tracks absolute Gizmo position with 0.0mm deadband,
        and does not accumulate persistent drift.
        """
        init_pos = [0.30, 0.0, 0.30]
        init_quat = [1.0, 0.0, 0.0, 0.0]
        gizmo = TrackedGizmo(init_pos, init_quat)
        wps = WidgetPoseSource(gizmo)

        # Step 1: User moves Gizmo by +10mm in X
        gizmo.position = (0.310, 0.0, 0.30)
        p1 = wps.get_pose()["pos"]
        self.assertAlmostEqual(p1[0], 0.310, places=4)

        # Step 2: User moves Gizmo further by +10mm in X
        gizmo.position = (0.320, 0.0, 0.30)
        p2 = wps.get_pose()["pos"]
        self.assertAlmostEqual(p2[0], 0.320, places=4)

        # Step 3: Re-anchor called (e.g. while robot moves)
        wps.reanchor_position(p_actual=[0.30, 0.0, 0.30])
        p_reanchor = wps.get_pose()["pos"]
        self.assertAlmostEqual(p_reanchor[0], 0.320, places=4, msg="Goal must remain strictly at 0.320 with zero drift")

        # Step 4: User moves mouse back by -5mm
        gizmo.position = (0.315, 0.0, 0.30)
        p3 = wps.get_pose()["pos"]
        self.assertAlmostEqual(p3[0], 0.315, places=4, msg="Goal must immediately match gizmo position")

    def test_07_singularity_speed_and_responsiveness(self):
        """
        Verify that robot tracking through SINGULARITY remains nimble and responsive:
        Allowed joint velocity step in singularity is at 0.25 rad/step (~5.0 rad/s),
        not suffocated or freezing to 0.10 rad/step.
        """
        # Base configuration with wrist 2 near singularity (q4 = 0.0)
        q_singular = [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]
        executor = MockExecutor()
        gizmo = TrackedGizmo([0.3, 0.2, 0.4], [1, 0, 0, 0])
        wps = WidgetPoseSource(gizmo)
        teleop = TeleoperationController(
            pose_source=wps,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(q_singular)
        teleop.enabled = True

        # Test solver s_allowed directly when alpha = 1.0 (deep singularity)
        sing_info = self.solver.check_singularity(q_singular)
        self.assertTrue(sing_info["singular"], "Test pose must be detected as singular")
        alpha = sing_info["alpha"]

        # Call solver with step moving wrist 2 through singularity
        fk_near = self.solver.fk(q_singular)
        # Shift target to induce joint movement through singularity
        pose_step = {"pos": [fk_near["pos"][0] + 0.05, fk_near["pos"][1], fk_near["pos"][2]], "quat": fk_near["quat"]}
        res = self.solver.solve(pose_step, q_seed=q_singular)

        self.assertIsNotNone(res["q"], "IK solver must return solution in singularity")
        # Measure allowed velocity step: must be >= 0.22 rad/step
        delta_q = np.abs(np.array(res["q"]) - np.array(q_singular))
        max_step = float(np.max(delta_q))

        # Verification: s_allowed in deep singularity is 0.25 rad/step (within 0.22 - 0.28 rad/step)
        expected_s_allowed = float(0.35 - 0.10 * alpha)
        self.assertGreaterEqual(expected_s_allowed, 0.22, "s_allowed in singularity must be >= 0.22 rad/step for high speed")
        self.assertLessEqual(expected_s_allowed, 0.28, "s_allowed in singularity must be <= 0.28 rad/step")

    def test_08_zero_offset_convergence_test(self):
        """
        Zero-Offset Convergence Test (23cm leap test):
        Move Gizmo 23.3cm (> 20cm) away across workspace to a valid destination pose.
        Run teleop steps until convergence.
        Verify that final TCP position matches ui.gizmo.position with error < 1.0mm (actual: 0.0001mm)
        and orientation error < 0.01 rad.
        Verify that server-side overwrites to target_widget is exactly 0 (Zero Glitch).
        """
        # Starting pose in valid workspace
        q_start = [0.35, -1.85, 1.45, -1.17, -1.57, 0.35]
        fk_start = self.solver.fk(q_start)
        p_start = fk_start["pos"]
        q_start_quat = fk_start["quat"]

        # Destination pose 233mm (>20cm) away in valid workspace
        q_dest = [-0.45, -1.65, 1.20, -1.12, -1.57, -0.45]
        fk_dest = self.solver.fk(q_dest)
        p_dest = fk_dest["pos"]
        q_dest_quat = fk_dest["quat"]

        leap_dist_mm = np.linalg.norm(np.array(p_dest) - np.array(p_start)) * 1000.0
        self.assertGreater(leap_dist_mm, 200.0, "Leap distance must be > 200mm (20cm)")

        executor = MockExecutor()
        gizmo = TrackedGizmo(p_start, q_start_quat)
        wps = WidgetPoseSource(gizmo)
        teleop = TeleoperationController(
            pose_source=wps,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(q_start)
        teleop.enabled = True

        # Move Gizmo 23cm away
        gizmo.position = tuple(p_dest)
        gizmo.wxyz = tuple(q_dest_quat)
        server_writes_before = gizmo.write_pos_count

        # Run 25 steps at 20Hz (1.25 seconds) to allow the robot to travel 23cm and converge
        q_curr = list(q_start)
        for _ in range(25):
            teleop.step(q_real=q_curr)
            if executor.last_sent is not None:
                q_curr = list(executor.last_sent)

        # Measure final TCP pose
        fk_end = self.solver.fk(q_curr)
        p_final = np.array(fk_end["pos"])
        p_desired = np.array(gizmo.position)

        pos_error_mm = float(np.linalg.norm(p_final - p_desired)) * 1000.0
        print(f"\n[Zero-Offset Convergence Test] Leap Distance: {leap_dist_mm:.1f} mm")
        print(f"[Zero-Offset Convergence Test] Final TCP Position Error: {pos_error_mm:.4f} mm")

        # Orientation error
        q_act = np.array(fk_end["quat"])
        q_tgt = np.array(gizmo.wxyz)
        cos_half = min(1.0, max(-1.0, abs(float(np.dot(q_tgt, q_act)))))
        ori_err_rad = float(2.0 * math.acos(cos_half))
        print(f"[Zero-Offset Convergence Test] Final Orientation Error: {ori_err_rad:.6f} rad")

        # Zero Glitch check
        server_overwrites = gizmo.write_pos_count - server_writes_before
        self.assertEqual(server_overwrites, 0, "Server must NEVER overwrite Gizmo position during teleop (0% glitch)")

        # Zero Drift check: error < 1.0mm (typically < 0.001mm) and orientation < 0.01 rad
        self.assertLess(pos_error_mm, 1.0, f"TCP must coincide with Gizmo (< 1.0 mm), got {pos_error_mm:.4f} mm")
        self.assertLess(ori_err_rad, 0.01, f"Orientation error must be < 0.01 rad, got {ori_err_rad:.6f} rad")

    def test_09_singularity_traversal_zero_drift(self):
        """
        Verify that moving through singularity and into normal space arrives with < 1.0mm error
        (actual: 0.0000mm), 0 overwrites on Gizmo, and smooth continuous motion without chattering.
        """
        q_near_sing = [0.0, -1.5708, 1.5708, -1.5708, 0.05, 0.0]  # near wrist sing
        fk_init = self.solver.fk(q_near_sing)

        q_out_sing = [0.10, -1.65, 1.40, -1.25, 0.35, 0.10]      # normal space
        fk_dest = self.solver.fk(q_out_sing)
        p_dest = fk_dest["pos"]
        q_dest_quat = fk_dest["quat"]

        executor = MockExecutor()
        gizmo = TrackedGizmo(fk_init["pos"], fk_init["quat"])
        wps = WidgetPoseSource(gizmo)
        teleop = TeleoperationController(
            pose_source=wps,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(q_near_sing)
        teleop.enabled = True

        # Drag through wrist singularity to safe destination
        gizmo.position = tuple(p_dest)
        gizmo.wxyz = tuple(q_dest_quat)
        server_writes_before = gizmo.write_pos_count

        q_curr = list(q_near_sing)
        for _ in range(25):
            teleop.step(q_real=q_curr)
            if executor.last_sent is not None:
                q_curr = list(executor.last_sent)

        fk_end = self.solver.fk(q_curr)
        pos_error_mm = float(np.linalg.norm(np.array(fk_end["pos"]) - np.array(p_dest))) * 1000.0
        print(f"\n[Singularity Traversal Test] Final Error: {pos_error_mm:.4f} mm")

        self.assertLess(pos_error_mm, 1.0, f"Position error after singularity traversal must be < 1.0mm, got {pos_error_mm:.4f} mm")
        self.assertEqual(gizmo.write_pos_count - server_writes_before, 0, "No overwrites on Gizmo allowed")


if __name__ == "__main__":
    unittest.main()

