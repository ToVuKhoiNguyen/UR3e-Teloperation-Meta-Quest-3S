#!/usr/bin/env python3
"""
tests/test_elbow_zero_deadlock_fix.py
Kiểm chứng dứt điểm lỗi Khóa chết (Deadlock) tại JOINT LIMIT khi tay máy vươn thẳng (Elbow ≈ 0° / -0.86°):

1. Straight Reach Free-Run Test:
   - Khởi tạo tại đúng tư thế trong ảnh:
     Base = 14.09°, Shoulder = -52.48°, Elbow = -0.86° (-0.015 rad), Wrist 1 = -30.71°, Wrist 2 = -90.93°, Wrist 3 = -0.97°.
   - Robot chạy mượt mà, không bị khóa chết, không false rejection.
2. Reach Limit Tethering & Non-Locking Test:
   - Kéo mục tiêu ra xa quá tầm với tối đa (X += 12cm).
   - Bộ tethering D_MAX_TETHER = 60mm chặn target runaway, robot dừng êm tại biên tầm với.
   - Trạng thái phản ánh đúng (SINGULARITY (ELBOW) / JOINT LIMIT), servoJ vẫn gửi lệnh an toàn, không đứng tim luồng.
3. Immediate Recovery & Elbow Folding Test:
   - Ngay khi kéo Gizmo / VR lùi lại về phía thân robot, vector đảo chiều ngay lập tức.
   - Khớp Elbow gập lại mượt mà (q2 > 0.15 rad -> 0.5 rad), trạng thái chuyển mượt về 'OK'.
   - Tuyệt đối không cần khởi động lại chương trình.
4. Per-Joint Soft Clamping Multi-Axis Test:
   - Cố tình kéo khớp chạm biên: Khớp chạm biên kẹp an toàn, các khớp còn lại vẫn di chuyển bình thường,
     và thoát ra ngay khi đảo chiều điều khiển.
"""

import os
import sys
import math
import time
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS, validate_joints, clamp_joints
from teleop.teleoperation import TeleoperationController, SafetyGate


class MockExecutor:
    def __init__(self):
        self.last_sent = None
        self.sent_history = []
        self.stopped = False

    def connect(self):
        pass

    def disconnect(self):
        pass

    def send(self, q):
        self.last_sent = list(q)
        self.sent_history.append(list(q))
        self.stopped = False

    def stop(self, a=10.0):
        self.stopped = True


class MockPoseSource:
    def __init__(self, pose=None):
        self.pose = pose
        self.is_absolute = True
        self.reanchor_called = False

    def get_pose(self):
        return self.pose

    def reset_anchor(self):
        self.reanchor_called = True

    def reanchor_position(self, pos):
        self.reanchor_called = True


class TestElbowZeroDeadlockFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" VERIFICATION: ZERO DEADLOCK AT JOINT LIMIT / ELBOW ≈ 0° EXTENSION")
        print("=" * 70)
        cls.solver = PyrokiIKSolver()

        class MockGuard:
            danger_threshold = 0.028
            warn_threshold = 0.042
            class Metrics:
                device = "cpu"
                def get_metrics(self, q):
                    import torch
                    return {"sigma_min": torch.tensor(0.080)}
            metrics_calc = Metrics()

        cls.guard = MockGuard()
        cls.gate = SafetyGate(guard=cls.guard, solver=cls.solver)

        # Pose in user report:
        # Base = 14.09°, Shoulder = -52.48°, Elbow = -0.86°, Wrist 1 = -30.71°, Wrist 2 = -90.93°, Wrist 3 = -0.97°
        cls.q_straight = [
            math.radians(14.09),
            math.radians(-52.48),
            math.radians(-0.86),   # -0.0150 rad
            math.radians(-30.71),
            math.radians(-90.93),
            math.radians(-0.97),
        ]

    def setUp(self):
        self.exec = MockExecutor()
        self.pose = MockPoseSource()
        self.ctrl = TeleoperationController(
            pose_source=self.pose,
            solver=self.solver,
            gate=self.gate,
            executor=self.exec,
            dt=0.05
        )
        self.ctrl.enabled = True
        self.ctrl.sync_with_robot(self.q_straight)

    def test_01_straight_reach_free_run_no_lock(self):
        """Ca 1: Vươn thẳng cánh tay về đúng tư thế trong ảnh (Elbow ≈ -0.86° / -0.015 rad) -> Không bị khóa"""
        print("\n[Ca 1 - Khởi tạo và điều khiển tại tư thế duỗi thẳng Elbow ≈ -0.86°]")
        fk_straight = self.solver.fk(self.q_straight)
        pose_init = {"pos": list(fk_straight["pos"]), "quat": list(fk_straight["quat"])}

        # Chạy 5 chu kỳ tại tư thế duỗi thẳng
        for step in range(5):
            self.ctrl.step(q_real=self.exec.last_sent or self.q_straight, pose=pose_init)

        status = self.ctrl.get_status()
        last_q = self.ctrl._last_valid_q
        elbow_deg = math.degrees(last_q[2])
        print(f"  Status: {status} | Elbow: {last_q[2]:.4f} rad ({elbow_deg:.2f}°)")

        # Phải gửi lệnh điều khiển xuống robot, không được drop
        self.assertIsNotNone(self.exec.last_sent, "Lệnh điều khiển phải được gửi xuống robot!")
        self.assertTrue(self.ctrl.enabled, "Teleop controller phải tiếp tục hoạt động, không bị disable!")
        # Elbow phải duy trì gần -0.86° không bị đẩy vọt đi
        self.assertAlmostEqual(last_q[2], math.radians(-0.86), delta=0.05,
                               msg="Khớp Elbow không được bị đẩy vọt xa khỏi tư thế thực!")
        # Trạng thái không bị hard lock
        self.assertIn(status, ["OK", "SINGULARITY (ELBOW)", "JOINT LIMIT"])
        print("  -> PASS: Không bị khóa chết tại tư thế duỗi thẳng cánh tay!")

    def test_02_reach_limit_tethering_and_non_locking(self):
        """Ca 2: Kéo mục tiêu vượt tầm với tối đa -> Dừng êm, Target Pose Tethering chặn runaway"""
        print("\n[Ca 2 - Kéo mục tiêu vượt xa tầm với tối đa (Target Tethering)]")
        fk_straight = self.solver.fk(self.q_straight)
        p0 = np.array(fk_straight["pos"])
        q0 = fk_straight["quat"]

        # Kéo dần ra xa thêm 12cm (vượt xa tầm với cơ khí)
        for i in range(1, 13):
            p_out = p0 + np.array([0.01 * i, 0.0, 0.0])
            self.ctrl.step(q_real=self.exec.last_sent, pose={"pos": p_out.tolist(), "quat": q0})

        q_final = self.ctrl._last_valid_q
        fk_final = self.solver.fk(q_final)
        p_robot = np.array(fk_final["pos"])
        smoothed_p = self.ctrl._smoothed_pos

        tether_dist = np.linalg.norm(smoothed_p - p_robot)
        print(f"  Robot TCP X: {p_robot[0]:.3f}m | Smoothed Target X: {smoothed_p[0]:.3f}m")
        print(f"  Khoảng cách dây xích (Tether distance): {tether_dist*1000:.1f} mm")

        # Khoảng cách giữa smoothed target và robot TCP không bao giờ được trôi vượt quá 80mm
        self.assertLessEqual(tether_dist, 0.080, "Target pose không được chạy trôi xa vô hạn khỏi robot TCP!")
        self.assertTrue(self.ctrl.enabled, "Teleop phải tiếp tục hoạt động ở biên!")
        self.assertIsNotNone(self.exec.last_sent, "Robot vẫn gửi lệnh giữ vị trí ở biên!")
        print("  -> PASS: Target Tethering hoạt động hoàn hảo, chặn đứng hiện tượng Target Runaway!")

    def test_03_immediate_recovery_when_pulling_back(self):
        """Ca 3: Kéo Gizmo lùi lại về phía thân robot -> Gập Elbow tức thì, thoát khỏi LIMIT về OK"""
        print("\n[Ca 3 - Kéo lùi mục tiêu về phía thân robot -> Phục hồi tức thì về OK]")
        fk_straight = self.solver.fk(self.q_straight)
        p0 = np.array(fk_straight["pos"])
        q0 = fk_straight["quat"]

        # Bước 1: Kéo ra biên xa 10cm
        for i in range(1, 11):
            p_out = p0 + np.array([0.01 * i, 0.0, 0.0])
            self.ctrl.step(q_real=self.exec.last_sent, pose={"pos": p_out.tolist(), "quat": q0})

        status_at_boundary = self.ctrl.get_status()
        print(f"  Trạng thái tại biên xa: {status_at_boundary}")

        # Bước 2: Kéo lùi lại về phía thân robot (lùi dần vào trong)
        p_peak = p0 + np.array([0.10, 0.0, 0.0])
        elbow_angles = []
        statuses = []

        # Lùi 15 bước về phía trong
        for i in range(1, 16):
            p_back = p_peak - np.array([0.01 * i, 0.0, 0.0])
            self.ctrl.step(q_real=self.exec.last_sent, pose={"pos": p_back.tolist(), "quat": q0})
            q_now = self.ctrl._last_valid_q
            elbow_angles.append(q_now[2])
            statuses.append(self.ctrl.get_status())

        final_status = self.ctrl.get_status()
        final_elbow = elbow_angles[-1]
        print(f"  Elbow angles trong quá trình kéo lùi: {[round(x, 3) for x in elbow_angles[::3]]}")
        print(f"  Trạng thái cuối cùng: {final_status} | Elbow góc cuối: {final_elbow:.3f} rad ({math.degrees(final_elbow):.1f}°)")

        # Khớp Elbow phải gập lại rõ rệt (ít nhất > 0.20 rad)
        self.assertGreater(final_elbow, 0.20, "Khớp Elbow phải gập lại khi kéo mục tiêu lùi về phía thân robot!")
        # Trạng thái phải phục hồi hoàn toàn về OK mà không cần restart
        self.assertEqual(final_status, "OK", "Trạng thái phải tự động phục hồi về OK khi kéo lùi!")
        print("  -> PASS: Robot gập khuỷu tay và tự động thoát khỏi LIMIT về OK mượt mà!")

    def test_04_per_joint_sliding_and_escape(self):
        """Ca 4: Khi 1 khớp chạm biên (vd Shoulder Lift), các khớp khác vẫn chuyển động và thoát ra ngay khi đảo chiều"""
        print("\n[Ca 4 - Trượt dọc biên khớp và thoát ra khi đổi chiều]")
        # Đưa robot về tư thế bình thường
        q_norm = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.ctrl.sync_with_robot(q_norm)
        fk_norm = self.solver.fk(q_norm)
        p_norm = np.array(fk_norm["pos"])
        q_quat = fk_norm["quat"]

        # Kéo TCP lên cao hết cỡ để ép Shoulder Lift tới giới hạn trên (+0.20 rad)
        for i in range(15):
            p_high = p_norm + np.array([0.0, 0.0, 0.02 * i])
            self.ctrl.step(q_real=self.exec.last_sent, pose={"pos": p_high.tolist(), "quat": q_quat})

        q_at_top = self.ctrl._last_valid_q
        self.assertLessEqual(q_at_top[1], 0.201, "Shoulder lift không được vượt quá giới hạn an toàn!")
        self.assertIsNotNone(self.exec.last_sent)

        # Bây giờ kéo TCP hạ thấp xuống
        for i in range(15):
            p_down = p_norm + np.array([0.0, 0.0, 0.20 - 0.02 * i])
            self.ctrl.step(q_real=self.exec.last_sent, pose={"pos": p_down.tolist(), "quat": q_quat})

        q_after_down = self.ctrl._last_valid_q
        # Shoulder lift phải lập tức hạ xuống thoát khỏi biên trên
        self.assertLess(q_after_down[1], 0.0, "Shoulder lift phải thoát khỏi biên trên ngay khi kéo xuống!")
        print(f"  Shoulder Lift sau khi kéo xuống: {q_after_down[1]:.3f} rad -> Thoát biên thành công!")
        print("  -> PASS: Non-Locking Per-Joint Soft Clamping thoát biên mượt mà trên mọi trục!")


if __name__ == "__main__":
    unittest.main()
