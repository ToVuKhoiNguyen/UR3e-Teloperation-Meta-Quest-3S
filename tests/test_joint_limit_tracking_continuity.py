#!/usr/bin/env python3
"""
tests/test_joint_limit_tracking_continuity.py
Automated Verification Suite for Joint Limit Tracking Continuity:
1. Shoulder Lift Limit Continuity: Drag Gizmo to push shoulder_lift (q1) to limit -> q1 clamped at limit,
   JOINT LIMIT logged, immediately drag Gizmo sideways -> other joints rotate smoothly, TCP follows Gizmo.
2. Elbow Limit Continuity: Drag Gizmo to push elbow (q2) to boundary -> q2 clamped, other joints move.
3. Shoulder Pan Limit Continuity: Rotate Gizmo to boundary ±pi -> q0 clamped, immediately move Gizmo in Z/R -> TCP tracks.
4. Instant Escape: Moving Gizmo away from limit transitions status to OK immediately in 1-2 cycles.
5. Zero Deadlock: Robot never freezes, no Go Home or reset required.
"""

import sys
import os
import math
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import (
    PyrokiIKSolver,
    UR3E_LIMITS,
    validate_joints,
    clamp_joints,
)
from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop_quest.pose_sources import WidgetPoseSource


class MockGizmoWidget:
    """Mock for Viser 3D TransformControls Gizmo with position and wxyz."""
    def __init__(self, position, wxyz):
        self.position = list(position)
        self.wxyz = list(wxyz)


class MockRobotExecutor:
    """Mock RTDE robot command executor."""
    def __init__(self):
        self.last_sent = None
        self.sent_history = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def send(self, q):
        self.last_sent = list(q)
        self.sent_history.append(list(q))

    def stop(self, a=10.0):
        pass


class TestJointLimitTrackingContinuity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("=" * 80)
        print(" TEST SUITE: JOINT LIMIT TRACKING CONTINUITY & GIZMO FREEDOM VERIFICATION")
        print("=" * 80)
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=None, solver=cls.solver)
        cls.safe_home = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]

    def test_01_shoulder_lift_limit_tracking_continuity(self):
        """
        Kéo Gizmo ép shoulder_lift (q1) chạm biên -> Khớp đó dừng tại biên,
        báo JOINT LIMIT, ngay lập tức kéo Gizmo dịch chuyển theo hướng khác (Y/sideways)
        -> Xác nhận tay máy tiếp tục bám theo Gizmo mượt mà không bị chết cứng.
        """
        print("\n[Test 1] Shoulder Lift Limit Tracking Continuity")
        fk_home = self.solver.fk(self.safe_home)
        p_home = np.array(fk_home["pos"])
        quat_home = list(fk_home["quat"])

        widget = MockGizmoWidget(p_home, quat_home)
        pose_source = WidgetPoseSource(widget)
        mock_exec = MockRobotExecutor()

        teleop = TeleoperationController(
            pose_source=pose_source,
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
            dt=0.05,
        )
        teleop.enabled = True
        teleop.sync_with_robot(self.safe_home)

        # Giai đoạn 1: Kéo Gizmo lùi sau (-X) để ép shoulder_lift (q1) về biên dưới (-2.65 rad)
        q_curr = list(self.safe_home)
        hit_limit = False
        for step in range(1, 20):
            widget.position = (p_home + np.array([-0.015 * step, 0.0, 0.0])).tolist()
            teleop.step(q_curr)
            q_curr = list(mock_exec.last_sent)
            st = teleop.get_status()
            if st == "JOINT LIMIT" or abs(q_curr[1] - UR3E_LIMITS[1][0]) <= 0.015:
                hit_limit = True
                print(f"  Reached JOINT LIMIT at step {step}: q1={q_curr[1]:.4f} rad, status={st}")
                break

        self.assertTrue(hit_limit, "Phải đạt trạng thái JOINT LIMIT khi kéo Gizmo tới biên!")
        q_at_limit = list(q_curr)

        # Kiểm tra khớp q1 nằm đúng trong dải an toàn của UR3E_LIMITS
        self.assertGreaterEqual(q_at_limit[1], UR3E_LIMITS[1][0] - 1e-4)
        self.assertLessEqual(q_at_limit[1], UR3E_LIMITS[1][1] + 1e-4)

        # Giai đoạn 2: NGAY LẬP TỨC kéo Gizmo sang ngang theo trục Y (sideways)
        print("  Ngay lập tức kéo Gizmo sang ngang theo trục Y (sideways)...")
        p_limit = np.array(widget.position)
        for side_step in range(1, 6):
            widget.position = (p_limit + np.array([0.0, 0.015 * side_step, 0.0])).tolist()
            teleop.step(q_curr)
            q_next = list(mock_exec.last_sent)
            delta = np.array(q_next) - np.array(q_curr)
            max_dq = float(np.max(np.abs(delta)))

            print(f"    Sideways step {side_step}: status={teleop.get_status()}, max_dq={max_dq:.4f} rad, q0={q_next[0]:.4f}, q1={q_next[1]:.4f}")
            self.assertGreater(max_dq, 0.001, f"Robot bị đơ tại step {side_step}! delta_q = {max_dq}")
            q_curr = q_next

        # Xác nhận base joint q0 đã xoay để bám theo chuyển động Y của Gizmo
        self.assertNotAlmostEqual(q_curr[0], q_at_limit[0], places=2, msg="Khớp base q0 phải xoay để bám theo Gizmo!")
        print("  -> PASS: Khớp chạm biên kẹp mềm, các khớp còn lại bám Gizmo mượt mà 100%!")

    def test_02_elbow_limit_tracking_continuity(self):
        """
        Kéo Gizmo ép elbow (q2) chạm biên -> q2 dừng tại biên,
        ngay lập tức kéo Gizmo theo hướng khác -> tay máy tiếp tục bám theo Gizmo.
        """
        print("\n[Test 2] Elbow Limit Tracking Continuity")
        fk_home = self.solver.fk(self.safe_home)
        p_home = np.array(fk_home["pos"])
        quat_home = list(fk_home["quat"])

        widget = MockGizmoWidget(p_home, quat_home)
        pose_source = WidgetPoseSource(widget)
        mock_exec = MockRobotExecutor()

        teleop = TeleoperationController(
            pose_source=pose_source,
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
            dt=0.05,
        )
        teleop.enabled = True
        teleop.sync_with_robot(self.safe_home)

        # Kéo Gizmo theo đường cong vươn cao và duỗi thẳng cánh tay để ép elbow về biên dưới (0.05 rad)
        q_curr = list(self.safe_home)
        hit_limit = False
        for step in range(1, 20):
            widget.position = (p_home + np.array([0.012 * step, 0.0, 0.015 * step])).tolist()
            teleop.step(q_curr)
            q_curr = list(mock_exec.last_sent)
            st = teleop.get_status()
            if st == "JOINT LIMIT" or abs(q_curr[2] - UR3E_LIMITS[2][0]) <= 0.020:
                hit_limit = True
                print(f"  Elbow at limit at step {step}: q2={q_curr[2]:.4f} rad (limit={UR3E_LIMITS[2][0]}), status={st}")
                break

        self.assertTrue(hit_limit, "Elbow phải đạt sát giới hạn!")
        q_elbow_limit = list(q_curr)
        self.assertGreaterEqual(q_elbow_limit[2], UR3E_LIMITS[2][0] - 1e-4)

        # Ngay lập tức kéo Gizmo xoay ngang (trục Y)
        print("  Ngay lập tức kéo Gizmo dịch chuyển theo trục Y...")
        p_elbow_limit = np.array(widget.position)
        for side_step in range(1, 6):
            widget.position = (p_elbow_limit + np.array([0.0, 0.015 * side_step, 0.0])).tolist()
            teleop.step(q_curr)
            q_next = list(mock_exec.last_sent)
            delta = np.array(q_next) - np.array(q_curr)
            max_dq = float(np.max(np.abs(delta)))

            print(f"    Elbow sideways step {side_step}: status={teleop.get_status()}, max_dq={max_dq:.4f} rad, q0={q_next[0]:.4f}, q2={q_next[2]:.4f}")
            self.assertGreater(max_dq, 0.001, f"Robot bị đơ khi elbow chạm limit! max_dq={max_dq}")
            q_curr = q_next

        self.assertNotAlmostEqual(q_curr[0], q_elbow_limit[0], places=2, msg="Base joint phải xoay theo Gizmo!")
        print("  -> PASS: Elbow chạm biên nhưng robot vẫn linh hoạt bám Gizmo theo các bậc tự do khác!")

    def test_03_shoulder_pan_limit_tracking_continuity(self):
        """
        Xoay Gizmo quanh trục Z ép shoulder_pan (q0) chạm kịch biên ±pi -> q0 dừng tại biên,
        ngay lập tức kéo Gizmo theo trục thẳng đứng Z -> TCP nâng hạ bình thường, không kẹt.
        """
        print("\n[Test 3] Shoulder Pan Limit Tracking Continuity")
        fk_home = self.solver.fk(self.safe_home)
        p_home = np.array(fk_home["pos"])
        r = float(np.linalg.norm(p_home[:2]))
        z = float(p_home[2])

        widget = MockGizmoWidget(p_home, fk_home["quat"])
        pose_source = WidgetPoseSource(widget)
        mock_exec = MockRobotExecutor()

        teleop = TeleoperationController(
            pose_source=pose_source,
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
            dt=0.05,
        )
        teleop.enabled = True
        teleop.sync_with_robot(self.safe_home)

        # Xoay Gizmo qua theta để ép q0 tới sát biên pi
        q_curr = list(self.safe_home)
        for step in range(1, 42):
            theta = 0.1 * step
            widget.position = [r * math.cos(theta), r * math.sin(theta), z]
            teleop.step(q_curr)
            q_curr = list(mock_exec.last_sent)

        st = teleop.get_status()
        print(f"  At pan boundary: status={st}, q0={q_curr[0]:.4f} rad (pi={math.pi:.4f})")
        self.assertAlmostEqual(abs(q_curr[0]), math.pi - 1e-3, delta=0.05, msg="q0 phải nằm sát biên pi!")

        # NGAY LẬP TỨC nâng Gizmo theo phương thẳng đứng Z
        print("  Ngay lập tức nâng Gizmo theo phương thẳng đứng Z (+50mm)...")
        p_at_pan = list(widget.position)
        for z_step in range(1, 6):
            widget.position = [p_at_pan[0], p_at_pan[1], p_at_pan[2] + 0.010 * z_step]
            teleop.step(q_curr)
            q_next = list(mock_exec.last_sent)
            delta = np.array(q_next) - np.array(q_curr)
            max_dq = float(np.max(np.abs(delta)))

            fk_curr = self.solver.fk(q_next)
            print(f"    Z step {z_step}: status={teleop.get_status()}, max_dq={max_dq:.4f} rad, actual_Z={fk_curr['pos'][2]:.4f}m, q0={q_next[0]:.4f}")
            self.assertGreater(max_dq, 0.001, f"Robot bị khóa khi q0 chạm biên pi! max_dq={max_dq}")
            q_curr = q_next

        # Xác nhận Z của TCP đã dịch chuyển lên trên
        fk_final = self.solver.fk(q_curr)
        self.assertGreater(fk_final["pos"][2], z + 0.015, "TCP phải bám theo chuyển động nâng Z của Gizmo!")
        print("  -> PASS: Khớp pan chạm biên pi nhưng các khớp còn lại nâng hạ TCP bám Gizmo hoàn hảo!")

    def test_04_instant_escape_and_zero_deadlock(self):
        """
        Xác nhận từ trạng thái JOINT LIMIT, chỉ cần kéo Gizmo ngược lại hướng hợp lệ,
        trạng thái lập tức chuyển về OK ngay trong 1-2 chu kỳ điều khiển.
        """
        print("\n[Test 4] Instant Escape from JOINT LIMIT without Go Home")
        q_start = [0.0, -2.60, 1.2, -1.2, -1.57, 0.0]
        fk_start = self.solver.fk(q_start)
        p_start = np.array(fk_start["pos"])

        widget = MockGizmoWidget(p_start, fk_start["quat"])
        pose_source = WidgetPoseSource(widget)
        mock_exec = MockRobotExecutor()

        teleop = TeleoperationController(
            pose_source=pose_source,
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
            dt=0.05,
        )
        teleop.enabled = True
        teleop.sync_with_robot(q_start)

        # Kéo vào vùng vi phạm biên dưới shoulder_lift (-2.65 rad)
        q_limit = None
        for step in range(1, 4):
            widget.position = (p_start + np.array([-0.05 * step, 0.0, 0.0])).tolist()
            teleop.step(q_start if step == 1 else q_limit)
            q_limit = list(mock_exec.last_sent)
            st_limit = teleop.get_status()
            if st_limit == "JOINT LIMIT":
                break

        print(f"  Chạm biên: status = '{st_limit}', q1 = {q_limit[1]:.4f} rad")
        self.assertEqual(st_limit, "JOINT LIMIT")

        # Đảo chiều kéo Gizmo về hướng an toàn
        for escape_step in range(1, 3):
            widget.position = (p_start + np.array([0.06 * escape_step, 0.0, 0.0])).tolist()
            teleop.step(q_limit)
            q_escaped = list(mock_exec.last_sent)
            st_escape = teleop.get_status()
            print(f"  Escape step {escape_step}: status = '{st_escape}', q1 = {q_escaped[1]:.4f} rad")
            q_limit = q_escaped
            if st_escape == "OK":
                break

        self.assertEqual(st_escape, "OK", "Trạng thái phải chuyển về 'OK' ngay lập tức!")
        self.assertGreater(q_escaped[1], UR3E_LIMITS[1][0] + 0.02, "Khớp q1 phải dịch chuyển ra xa biên giới hạn!")
        print("  -> PASS: Thoát biên tức thì, 0 deadlock!")


if __name__ == "__main__":
    unittest.main()
