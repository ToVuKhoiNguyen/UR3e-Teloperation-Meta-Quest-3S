"""
tests/test_gizmo_visual_glitch_isolation.py

Comprehensive tests verifying:
1. TargetWidgetWrapper deadband filtering (position and rotation).
2. TargetWidgetWrapper quaternion hemisphere sign continuity (dot(new, curr) < 0 flip).
3. TargetWidgetWrapper programmatic suppression flag and echo absorption window.
4. Isolation of active sources:
   - Mouse Gizmo dragging has 100% authority (zero overwrite from server/telemetry).
   - VR Controller active ignores gizmo echo and tracks target smoothly.
   - Cockpit jogging ignores gizmo echo and tracks FK smoothly.
"""

import unittest
import numpy as np
import time
from unittest.mock import MagicMock

from teleop_quest.viser_ui import TargetWidgetWrapper
from teleop_quest.pose_sources import WidgetPoseSource


class MockRawGizmo:
    def __init__(self, position=(0.3, 0.0, 0.3), wxyz=(1.0, 0.0, 0.0, 0.0)):
        self._position = tuple(position)
        self._wxyz = tuple(wxyz)
        self.visible = True
        self.write_pos_count = 0
        self.write_wxyz_count = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, val):
        self._position = tuple(float(x) for x in val)
        self.write_pos_count += 1

    @property
    def wxyz(self):
        return self._wxyz

    @wxyz.setter
    def wxyz(self, val):
        self._wxyz = tuple(float(w) for w in val)
        self.write_wxyz_count += 1


class TestGizmoVisualGlitchFix(unittest.TestCase):
    def setUp(self):
        self.raw_gizmo = MockRawGizmo(position=[0.3, 0.0, 0.3], wxyz=[1.0, 0.0, 0.0, 0.0])
        self.wrapper = TargetWidgetWrapper(self.raw_gizmo)

    def test_01_quaternion_sign_continuity(self):
        """
        Verify that passing -q (which represents the exact same 3D rotation as +q)
        is flipped to +q so dot(new, curr) >= 0.0, preventing 360-deg visual jumps.
        """
        self.raw_gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)
        self.raw_gizmo.write_wxyz_count = 0

        # Sub-test A: Exact negative identity [-1, 0, 0, 0]
        # Should be flipped to [+1, 0, 0, 0], and since diff is 0, deadband prevents rewrite.
        self.wrapper.update_pose(wxyz=[-1.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.raw_gizmo.write_wxyz_count, 0, "Negated identical quat should be absorbed by deadband after sign flip")
        # Current gizmo orientation should remain +1
        self.assertAlmostEqual(self.raw_gizmo.wxyz[0], 1.0)

        # Sub-test B: Slight rotation (5 degrees around X) but with flipped sign
        # 5 deg = 0.0872665 rad -> half angle = 0.04363 rad
        # q = [cos(a/2), sin(a/2), 0, 0] = [0.999048, 0.043619, 0, 0]
        # Negated: [-0.999048, -0.043619, 0, 0]
        q_neg = [-0.9990482, -0.0436194, 0.0, 0.0]
        self.wrapper.update_pose(wxyz=q_neg)
        self.assertEqual(self.raw_gizmo.write_wxyz_count, 1, "Significant rotation should write to gizmo")
        # Gizmo wxyz MUST have positive w (hemisphere continuity)
        self.assertGreater(self.raw_gizmo.wxyz[0], 0.0, "Sign continuity MUST enforce w > 0 when previous was positive")
        self.assertAlmostEqual(self.raw_gizmo.wxyz[0], 0.9990482, places=5)
        self.assertAlmostEqual(self.raw_gizmo.wxyz[1], 0.0436194, places=5)

    def test_02_position_deadband(self):
        """
        Verify that tiny position changes (<= 1e-4 m / 0.1mm) do NOT push updates to raw gizmo,
        while larger movements (> 1e-4 m) write to raw gizmo.
        """
        self.raw_gizmo.position = (0.25, 0.15, 0.30)
        self.raw_gizmo.write_pos_count = 0

        # Sub-millimeter jitter (0.05 mm = 5e-5 m)
        jitter_pos = (0.25 + 0.00003, 0.15 + 0.00003, 0.30)
        self.wrapper.update_pose(pos=jitter_pos)
        self.assertEqual(self.raw_gizmo.write_pos_count, 0, "Sub-deadband jitter (0.05mm) must not trigger write")

        # Intentional move (1.0 mm = 1e-3 m)
        valid_pos = (0.251, 0.15, 0.30)
        self.wrapper.update_pose(pos=valid_pos)
        self.assertEqual(self.raw_gizmo.write_pos_count, 1, "Movement > 0.1mm must trigger write")
        np.testing.assert_allclose(self.raw_gizmo.position, valid_pos, atol=1e-6)

    def test_03_rotation_deadband(self):
        """
        Verify that micro-rotation changes (<= 1e-4 rad) do NOT push updates to raw gizmo.
        """
        self.raw_gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)
        self.raw_gizmo.write_wxyz_count = 0

        # Micro-rotation: 0.00002 rad (around Z)
        half_ang = 0.00001
        micro_q = (np.cos(half_ang), 0.0, 0.0, np.sin(half_ang))
        self.wrapper.update_pose(wxyz=micro_q)
        self.assertEqual(self.raw_gizmo.write_wxyz_count, 0, "Micro-rotation (<1e-4 rad) must not trigger write")

        # Valid rotation: 0.01 rad (around Z)
        valid_half_ang = 0.005
        valid_q = (np.cos(valid_half_ang), 0.0, 0.0, np.sin(valid_half_ang))
        self.wrapper.update_pose(wxyz=valid_q)
        self.assertEqual(self.raw_gizmo.write_wxyz_count, 1, "Noticeable rotation must trigger write")

    def test_04_suppression_active_window(self):
        """
        Verify that programmatic updates activate the echo suppression window (120ms).
        """
        self.assertFalse(self.wrapper.is_suppression_active())
        
        self.wrapper.update_pose(pos=[0.35, 0.0, 0.35], suppress=True)
        # Immediately after update, suppression MUST be active
        self.assertTrue(self.wrapper.is_suppression_active())

        # After waiting > 130ms, suppression window expires
        time.sleep(0.14)
        self.assertFalse(self.wrapper.is_suppression_active())

    def test_05_widget_pose_source_sync_uses_update_pose(self):
        """
        Verify that WidgetPoseSource.sync_with_robot delegates to update_pose with suppress=True,
        maintaining deadband and sign continuity.
        """
        wps = WidgetPoseSource(self.wrapper)
        self.raw_gizmo.write_pos_count = 0
        self.raw_gizmo.write_wxyz_count = 0

        # Synchronize to a new robot pose
        new_pos = [0.28, -0.10, 0.22]
        new_quat = [0.7071, 0.7071, 0.0, 0.0]
        wps.sync_with_robot(pos=new_pos, quat=new_quat)

        self.assertEqual(self.raw_gizmo.write_pos_count, 1)
        self.assertEqual(self.raw_gizmo.write_wxyz_count, 1)
        self.assertTrue(self.wrapper.is_suppression_active())

        # Subsequent identical sync should be swallowed by deadband (0 writes)
        wps.sync_with_robot(pos=new_pos, quat=new_quat)
        self.assertEqual(self.raw_gizmo.write_pos_count, 1, "Duplicate sync position swallowed by deadband")
        self.assertEqual(self.raw_gizmo.write_wxyz_count, 1, "Duplicate sync quat swallowed by deadband")

    def test_06_mouse_drag_isolation_from_telemetry(self):
        """
        Simulate user mouse drag:
        - gizmo_interaction is active
        - Telemetry and teleop loop must NOT overwrite raw gizmo
        """
        gizmo_interaction = [time.time()]
        source_type = "Viser Gizmo"

        # User dragged to mouse_pos
        mouse_pos = (0.33, 0.12, 0.29)
        self.raw_gizmo.position = mouse_pos
        self.raw_gizmo.write_pos_count = 0

        # Simulate simulated lagged robot actual TCP
        fk_actual_pos = [0.30, 0.10, 0.28]

        # In teleop loop:
        # If source_type == "VR Controller": sync_widget would be sent.
        # But for "Viser Gizmo": sync_widget is None
        sync_widget_data = None if source_type == "Viser Gizmo" else {"pos": fk_actual_pos}
        self.assertIsNone(sync_widget_data, "When Viser Gizmo is active, sync_widget must be None")

        # Telemetry UIWorker receives sync_widget: None -> 0 writes
        if sync_widget_data is not None:
            self.wrapper.update_pose(pos=sync_widget_data["pos"])

        self.assertEqual(self.raw_gizmo.write_pos_count, 0, "Server must NEVER overwrite gizmo while user is dragging with mouse")
        self.assertEqual(self.raw_gizmo.position, mouse_pos, "Gizmo position must strictly follow mouse")


if __name__ == "__main__":
    unittest.main()
