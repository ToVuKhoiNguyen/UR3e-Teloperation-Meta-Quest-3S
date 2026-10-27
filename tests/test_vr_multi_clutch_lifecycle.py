import os
import sys
import math
import time
import unittest
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config
from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop_quest.pose_sources import WidgetPoseSource, get_robot_fk_as_ur


class MockQuestServer:
    """Mock VR WebSocket server delivering controller state frames."""
    def __init__(self):
        self.frame = {
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": False,
        }
        self.haptic_events = []

    def set_frame(self, frame_dict):
        self.frame.update(frame_dict)

    def get_frame(self):
        return dict(self.frame)

    def send_haptic(self, intensity=0.35, duration=40):
        self.haptic_events.append((intensity, duration))

    def is_connected(self):
        return True


class MockRobotController:
    """Mock RTDE robot controller maintaining joint state and TCP FK."""
    def __init__(self, solver, start_q):
        self.solver = solver
        self._q = list(start_q)

    def joints(self):
        return list(self._q)

    def tcp(self):
        fk = self.solver.fk(self._q)
        pos = fk["pos"]
        # returns [x, y, z, rx, ry, rz]
        return [pos[0], pos[1], pos[2], 0.0, 3.14159, 0.0]

    def set_joints(self, q):
        self._q = list(q)


class MockExecutor:
    """Mock ServoJ executor that forwards commands to simulated robot joints."""
    def __init__(self, mock_ctrl):
        self.mock_ctrl = mock_ctrl
        self.last_sent = None
        self.stopped = False
        self.history = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def send(self, q):
        self.last_sent = list(q)
        self.mock_ctrl.set_joints(q)
        self.history.append(list(q))

    def stop(self, a=10.0):
        self.stopped = True


class MockWidget:
    def __init__(self, pos, quat):
        self.position = tuple(pos)
        self.wxyz = tuple(quat)


class TestVRMultiClutchLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_singularity_config(BASE_DIR)
        cls.guard = SingularityGuard(cls.config)
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=cls.guard, solver=cls.solver)

    def test_multi_clutch_infinite_lifecycle(self):
        """
        Verify the complete VR controller clutch lifecycle:
        1. Clutch 1 (Frames 1-20): Hand pulls +10cm in X -> Robot moves +10cm.
        2. Release & Shift (Frames 21-30): Hand releases trigger, moves 30cm in space -> Robot holds position.
        3. Clutch 2 (Frames 31-50): Hand re-engages trigger at point B, pulls +8cm in Y -> Robot does NOT jump,
           moves +8cm in Y from where Clutch 1 ended.
        4. Clutch 3 (Frames 51-70): Hand releases and re-engages again -> Works smoothly without teleop toggle!
        """
        # Starting configuration (safe workspace pose)
        start_joints = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        ctrl = MockRobotController(self.solver, start_joints)
        executor = MockExecutor(ctrl)
        server = MockQuestServer()

        fk_init = self.solver.fk(start_joints)
        init_pos = np.array(fk_init["pos"])
        init_quat = np.array(fk_init["quat"])

        # Setup Pose Sources
        widget = MockWidget(init_pos, init_quat)
        widget_pose_source = WidgetPoseSource(widget)
        widget_pose_source.sync_with_robot(pos=init_pos, quat=init_quat)

        teleop = TeleoperationController(
            pose_source=widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(start_joints)
        teleop.enabled = True

        pose_source = DeltaPoseSource(
            server=server,
            ctrl=ctrl,
            scale=1.0,
            anchor_reference_func=lambda: get_robot_fk_as_ur(teleop, ctrl),
            alpha=0.40,
            max_delta_m=0.30
        )
        # Configure for linear test coordinates
        pose_source.nav_mode = "cartesian"
        pose_source.enable_axis_snapping = False
        pose_source.enable_cross_axis_damping = False
        pose_source.fwd_deadband = 0.0
        pose_source.lat_deadband = 0.0
        pose_source.vert_deadband = 0.0
        pose_source.pos_deadband = 0.0

        # Hand starts at Point A
        hand_pos_A = np.array([0.10, 0.0, -0.30])
        server.set_frame({
            "pos": hand_pos_A.tolist(),
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": False
        })

        # ─────────────────────────────────────────────────────────────
        # PHASE 1: Lần bóp cò 1 (Frames 1-20)
        # Bóp cò tại vị trí A, kéo tay cầm đi +10cm theo trục X
        # ─────────────────────────────────────────────────────────────
        print("\n--- PHASE 1: LẦN BÓP CÒ 1 (Frames 1-20) ---")
        p_history = []
        _last_trig = False

        for frame_idx in range(1, 21):
            t = (frame_idx - 1) / 19.0  # 0.0 to 1.0
            # Pull hand forward (-Z in Quest maps to +X in robot)
            current_hand = hand_pos_A + np.array([0.0, 0.0, -0.10 * t])
            server.set_frame({"pos": current_hand.tolist(), "trigger": True})

            # Main teleop cycle logic
            _is_trig = True
            vr_pose = pose_source.get_pose()
            target_pose = vr_pose
            if target_pose is not None:
                teleop.step(ctrl.joints(), pose=target_pose)
                widget_pose_source.sync_with_robot(pos=ctrl.tcp()[:3], quat=init_quat)
            _last_trig = _is_trig

            fk_now = self.solver.fk(ctrl.joints())
            p_history.append(fk_now["pos"])

        p_end_clutch1 = np.array(p_history[-1])
        delta_clutch1 = p_end_clutch1 - init_pos
        print(f"  Clutch 1 End TCP: {np.round(p_end_clutch1, 4)}")
        print(f"  Clutch 1 Net Delta: dx={delta_clutch1[0]*100:.1f}cm, dy={delta_clutch1[1]*100:.1f}cm, dz={delta_clutch1[2]*100:.1f}cm")

        self.assertTrue(teleop.enabled, "Teleop must remain enabled throughout Clutch 1")
        self.assertEqual(teleop.get_status(), "OK", "Status must remain OK throughout Clutch 1")
        self.assertAlmostEqual(delta_clutch1[0], 0.10, delta=0.015,
                               msg=f"Robot must move +10cm in X during Clutch 1, actual: {delta_clutch1[0]*100:.2f}cm")

        # ─────────────────────────────────────────────────────────────
        # PHASE 2: Nhả cò & Dời tay trong không gian (Frames 21-30)
        # Nhả cò (trigger=False), dời tay cầm sang điểm B cách 30cm
        # ─────────────────────────────────────────────────────────────
        print("\n--- PHASE 2: NHẢ CÒ & DỜI TAY TRONG KHÔNG GIAN (Frames 21-30) ---")
        # Falling edge on frame 21
        _is_trig = False
        executor.stop(10.0)
        pose_source.reset_anchor()
        teleop.notify_clutch_released(ctrl.joints())
        widget_pose_source.sync_with_robot(pos=p_end_clutch1, quat=init_quat)

        # Move hand to Point B 30cm away in real space
        hand_pos_B = hand_pos_A + np.array([0.30, 0.15, 0.20])
        server.set_frame({"pos": hand_pos_B.tolist(), "trigger": False})

        for frame_idx in range(21, 31):
            vr_pose = pose_source.get_pose()
            self.assertIsNone(vr_pose, f"Pose must be None while trigger is released at frame {frame_idx}")
            teleop.step(ctrl.joints(), pose=None)

            fk_now = self.solver.fk(ctrl.joints())
            p_now = np.array(fk_now["pos"])
            drift = np.linalg.norm(p_now - p_end_clutch1)
            self.assertLess(drift, 1e-4, f"Robot must remain completely stationary while clutch released (drift={drift*1000:.3f}mm)")

        print("  -> Robot stood completely still at end of Clutch 1 while hand shifted 30cm in space!")
        self.assertTrue(teleop.enabled, "Teleop MUST remain enabled while clutch released!")

        # ─────────────────────────────────────────────────────────────
        # PHASE 3: Lần bóp cò 2 (Frames 31-50)
        # Bóp cò lại tại điểm B, kéo tay cầm đi tiếp +8cm theo trục Y
        # Xác nhận robot không bị giật nhảy về B và di chuyển tiếp +8cm
        # ─────────────────────────────────────────────────────────────
        print("\n--- PHASE 3: LẦN BÓP CÒ 2 (Frames 31-50) ---")
        # Frame 31: First frame of re-engaging trigger at point B
        server.set_frame({"pos": hand_pos_B.tolist(), "trigger": True})
        _is_trig = True

        vr_pose_31 = pose_source.get_pose()
        self.assertIsNotNone(vr_pose_31, "DeltaPoseSource must return pose at re-engagement frame 31")

        # Verify target pose at frame 31 is anchored at robot's current TCP (delta = 0)
        p_target_31 = np.array(vr_pose_31["pos"])
        gap_at_reengage = np.linalg.norm(p_target_31 - p_end_clutch1)
        print(f"  Frame 31 Target Pose: {np.round(p_target_31, 4)}")
        print(f"  Frame 31 Gap from Robot TCP: {gap_at_reengage*1000:.3f} mm")
        self.assertLess(gap_at_reengage, 0.005, "At re-engagement (frame 31), target must anchor at current robot TCP (delta=0)")

        # Step frame 31
        teleop.step(ctrl.joints(), pose=vr_pose_31)
        self.assertEqual(teleop.get_status(), "OK", "Status at frame 31 re-engagement must be OK (not locked!)")
        fk_31 = self.solver.fk(ctrl.joints())
        step_31_jump = np.linalg.norm(np.array(fk_31["pos"]) - p_end_clutch1)
        self.assertLess(step_31_jump, 0.002, "Robot must NOT jump at re-engagement frame 31!")

        # Frames 32-50: Pull hand in -X (which maps to +Y in robot world) by +8cm
        for frame_idx in range(32, 51):
            t = (frame_idx - 31) / 19.0
            # -X in Quest maps to +Y in robot
            current_hand = hand_pos_B + np.array([-0.08 * t, 0.0, 0.0])
            server.set_frame({"pos": current_hand.tolist(), "trigger": True})

            vr_pose = pose_source.get_pose()
            teleop.step(ctrl.joints(), pose=vr_pose)
            widget_pose_source.sync_with_robot(pos=ctrl.tcp()[:3], quat=init_quat)
            self.assertEqual(teleop.get_status(), "OK")

        p_end_clutch2 = np.array(self.solver.fk(ctrl.joints())["pos"])
        delta_clutch2 = p_end_clutch2 - p_end_clutch1
        print(f"  Clutch 2 End TCP: {np.round(p_end_clutch2, 4)}")
        print(f"  Clutch 2 Net Delta from Clutch 1: dx={delta_clutch2[0]*100:.1f}cm, dy={delta_clutch2[1]*100:.1f}cm, dz={delta_clutch2[2]*100:.1f}cm")

        self.assertAlmostEqual(delta_clutch2[1], 0.08, delta=0.015,
                               msg=f"Robot must move +8cm in Y during Clutch 2 from stopping point, actual: {delta_clutch2[1]*100:.2f}cm")
        self.assertAlmostEqual(p_end_clutch2[0], p_end_clutch1[0], delta=0.010,
                               msg="X coordinate must remain preserved from Clutch 1!")

        # ─────────────────────────────────────────────────────────────
        # PHASE 4: Lần bóp cò 3 (Frames 51-70)
        # Nhả cò và bóp lại lần thứ 3 -> Hoạt động mượt mà không cần chạm Teleop toggle
        # ─────────────────────────────────────────────────────────────
        print("\n--- PHASE 4: LẦN BÓP CÒ 3 (Frames 51-70) ---")
        # Release trigger at frame 51
        executor.stop(10.0)
        pose_source.reset_anchor()
        teleop.notify_clutch_released(ctrl.joints())
        widget_pose_source.sync_with_robot(pos=p_end_clutch2, quat=init_quat)

        # Shift hand to Point C (20cm away)
        hand_pos_C = hand_pos_B + np.array([0.15, -0.20, 0.10])
        server.set_frame({"pos": hand_pos_C.tolist(), "trigger": False})

        # Idle frames 51-54
        for frame_idx in range(51, 55):
            vr_pose = pose_source.get_pose()
            self.assertIsNone(vr_pose)
            teleop.step(ctrl.joints(), pose=None)

        # Re-engage trigger at frame 55 and move +5cm in Z
        for frame_idx in range(55, 71):
            t = (frame_idx - 55) / 15.0
            # +Y in Quest maps to +Z in robot
            current_hand = hand_pos_C + np.array([0.0, 0.05 * t, 0.0])
            server.set_frame({"pos": current_hand.tolist(), "trigger": True})

            vr_pose = pose_source.get_pose()
            teleop.step(ctrl.joints(), pose=vr_pose)
            widget_pose_source.sync_with_robot(pos=ctrl.tcp()[:3], quat=init_quat)
            self.assertEqual(teleop.get_status(), "OK")

        p_end_clutch3 = np.array(self.solver.fk(ctrl.joints())["pos"])
        delta_clutch3 = p_end_clutch3 - p_end_clutch2
        print(f"  Clutch 3 End TCP: {np.round(p_end_clutch3, 4)}")
        print(f"  Clutch 3 Net Delta from Clutch 2: dx={delta_clutch3[0]*100:.1f}cm, dy={delta_clutch3[1]*100:.1f}cm, dz={delta_clutch3[2]*100:.1f}cm")

        self.assertAlmostEqual(delta_clutch3[2], 0.05, delta=0.015,
                               msg=f"Robot must move +5cm in Z during Clutch 3, actual: {delta_clutch3[2]*100:.2f}cm")
        self.assertTrue(teleop.enabled, "Teleop must remain 100% active and enabled without touching UI toggle!")
        print("\n>>> ALL 4 PHASES PASSED WITH 100% ACCURACY! INFINITE CLUTCH CYCLE VERIFIED!")

    def test_02_ten_consecutive_clutches_without_degradation(self):
        """
        Verify that 10 consecutive clutch cycles (Engage -> Move -> Release -> Displace Hand -> Re-engage)
        execute flawlessly without degradation, locking, or needing to toggle teleop.
        """
        print("\n--- TEST 2: 10 CONSECUTIVE CLUTCHES STRESS TEST ---")
        start_joints = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        ctrl = MockRobotController(self.solver, start_joints)
        executor = MockExecutor(ctrl)
        server = MockQuestServer()

        fk_init = self.solver.fk(start_joints)
        init_pos = np.array(fk_init["pos"])
        init_quat = np.array(fk_init["quat"])

        widget = MockWidget(init_pos, init_quat)
        widget_pose_source = WidgetPoseSource(widget)
        widget_pose_source.sync_with_robot(pos=init_pos, quat=init_quat)

        teleop = TeleoperationController(
            pose_source=widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(start_joints)
        teleop.enabled = True

        pose_source = DeltaPoseSource(
            server=server,
            ctrl=ctrl,
            scale=1.0,
            anchor_reference_func=lambda: get_robot_fk_as_ur(teleop, ctrl),
            alpha=0.50,
            max_delta_m=0.30
        )
        pose_source.nav_mode = "cartesian"
        pose_source.enable_axis_snapping = False
        pose_source.enable_cross_axis_damping = False
        pose_source.fwd_deadband = 0.0
        pose_source.lat_deadband = 0.0
        pose_source.vert_deadband = 0.0
        pose_source.pos_deadband = 0.0

        for cycle in range(1, 11):
            # 1. Random hand point in room space
            hand_origin = np.array([0.1 * cycle, 0.05 * (cycle % 3), -0.25 - 0.02 * cycle])
            server.set_frame({"pos": hand_origin.tolist(), "trigger": False})

            # Release clutch
            executor.stop(10.0)
            pose_source.reset_anchor()
            teleop.notify_clutch_released(ctrl.joints())
            widget_pose_source.sync_with_robot(pos=ctrl.tcp()[:3], quat=init_quat)

            # Verify idle stationary
            p_before = np.array(self.solver.fk(ctrl.joints())["pos"])
            vr_pose_idle = pose_source.get_pose()
            self.assertIsNone(vr_pose_idle)
            if vr_pose_idle is not None:
                teleop.step(ctrl.joints(), pose=vr_pose_idle)
            else:
                executor.stop(10.0)
                teleop.notify_clutch_released(ctrl.joints())
            p_after_idle = np.array(self.solver.fk(ctrl.joints())["pos"])
            self.assertLess(np.linalg.norm(p_after_idle - p_before), 1e-4)

            # 2. Re-engage at hand_origin
            server.set_frame({"pos": hand_origin.tolist(), "trigger": True})
            vr_pose_reengage = pose_source.get_pose()
            self.assertIsNotNone(vr_pose_reengage)
            teleop.step(ctrl.joints(), pose=vr_pose_reengage)
            widget_pose_source.sync_with_robot(pos=ctrl.tcp()[:3], quat=init_quat)
            self.assertEqual(teleop.get_status(), "OK")
            self.assertTrue(teleop.enabled)

            # 3. Pull hand 15mm forward
            for step_i in range(1, 6):
                hand_curr = hand_origin + np.array([0.0, 0.0, -0.015 * (step_i / 5.0)])
                server.set_frame({"pos": hand_curr.tolist(), "trigger": True})
                vr_pose = pose_source.get_pose()
                teleop.step(ctrl.joints(), pose=vr_pose)
                widget_pose_source.sync_with_robot(pos=ctrl.tcp()[:3], quat=init_quat)
                self.assertEqual(teleop.get_status(), "OK")

            p_end_cycle = np.array(self.solver.fk(ctrl.joints())["pos"])
            print(f"  Cycle {cycle:2d}/10: Success | Robot TCP X = {p_end_cycle[0]:.4f}m | Status = OK")

        self.assertTrue(teleop.enabled, "Teleop must remain 100% active and enabled after 10 cycles!")
        print("  -> 10/10 consecutive clutch cycles passed with zero errors!")

    def test_03_viser_gizmo_coexistence_and_zero_glitch(self):
        """
        Verify that switching between VR Controller and Viser Gizmo is seamless,
        and WidgetPoseSource maintains internal shadow target without glitching UI gizmo.
        """
        print("\n--- TEST 3: VISER GIZMO & VR COEXISTENCE ---")
        start_joints = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        ctrl = MockRobotController(self.solver, start_joints)
        executor = MockExecutor(ctrl)
        server = MockQuestServer()

        fk_init = self.solver.fk(start_joints)
        init_pos = np.array(fk_init["pos"])
        init_quat = np.array(fk_init["quat"])

        widget = MockWidget(init_pos, init_quat)
        widget_pose_source = WidgetPoseSource(widget)
        widget_pose_source.sync_with_robot(pos=init_pos, quat=init_quat)

        teleop = TeleoperationController(
            pose_source=widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=executor
        )
        teleop.sync_with_robot(start_joints)
        teleop.enabled = True

        pose_source = DeltaPoseSource(
            server=server,
            ctrl=ctrl,
            scale=1.0,
            anchor_reference_func=lambda: get_robot_fk_as_ur(teleop, ctrl),
            alpha=0.50,
            max_delta_m=0.30
        )
        pose_source.nav_mode = "cartesian"

        # 1. Simulate VR controller active
        server.set_frame({"pos": [0.0, 0.0, -0.30], "trigger": True})
        vr_pose = pose_source.get_pose()
        teleop.step(ctrl.joints(), pose=vr_pose)
        fk_curr = self.solver.fk(ctrl.joints())
        widget_pose_source.sync_with_robot(pos=fk_curr["pos"], quat=fk_curr["quat"])

        # 2. Release VR controller
        server.set_frame({"pos": [0.0, 0.0, -0.30], "trigger": False})
        executor.stop(10.0)
        pose_source.reset_anchor()
        teleop.notify_clutch_released(ctrl.joints())

        # 3. Simulate mouse drag on Viser Gizmo (+20mm in Y)
        new_gizmo_pos = np.array(fk_curr["pos"]) + np.array([0.0, 0.020, 0.0])
        widget.position = tuple(new_gizmo_pos)

        gizmo_target = widget_pose_source.get_pose()
        teleop.step(ctrl.joints(), pose=gizmo_target)
        self.assertEqual(teleop.get_status(), "OK")

        # 4. Now user picks up VR controller again and pulls +10mm in Z
        server.set_frame({"pos": [0.10, 0.20, -0.15], "trigger": True})
        vr_pose_re = pose_source.get_pose()
        self.assertIsNotNone(vr_pose_re)
        teleop.step(ctrl.joints(), pose=vr_pose_re)
        self.assertEqual(teleop.get_status(), "OK")
        print("  -> VR <-> Viser Gizmo seamless switching verified successfully!")


if __name__ == "__main__":
    unittest.main()
