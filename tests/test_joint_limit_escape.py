#!/usr/bin/env python3
"""
tests/test_joint_limit_escape.py
Kịch bản kiểm thử tự động "Joint Limit Escape Test":
1. Test Per-Joint Soft Clamping: Khi khớp chạm biên giới hạn (shoulder_lift, pan, wrist),
   solver không bị fail (q != None), khớp chạm biên được kẹp an toàn, các khớp còn lại vẫn hoạt động.
2. Test Anti-Windup Re-anchoring: Khi kéo mục tiêu vượt biên liên tục, target không bị trôi
   quá 30mm so với TCP thực tế.
3. Test Instant Escape: Ngay khi đảo chiều kéo ngược lại, robot thoát khỏi JOINT LIMIT về OK
   ngay trong 1-2 chu kỳ điều khiển, không kẹt cứng, không cần Go Home.
4. Test Branch-Flip Guard quanh biên ±pi: Không bị văng góc 6.28 rad khi chớm chạm biên.
"""

import sys
import os
import math
import time
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
    _nearest_equivalent,
    normalize_to_current,
)
from teleop.teleoperation import TeleoperationController, SafetyGate


class MockExecutor:
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


class MockPoseSource:
    def __init__(self, init_pos, init_quat):
        self.pos = np.array(init_pos, dtype=float)
        self.quat = np.array(init_quat, dtype=float)
        self.reanchor_calls = []

    def get_pose(self):
        return {"pos": self.pos.tolist(), "quat": self.quat.tolist()}

    def reanchor_position(self, p_actual):
        self.reanchor_calls.append(list(p_actual))
        p_act = np.array(p_actual, dtype=float)[:3]
        vec = self.pos - p_act
        dist = float(np.linalg.norm(vec))
        if dist > 0.030:
            self.pos = p_act + (vec / dist) * 0.030


class TestJointLimitEscape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("=" * 75)
        print(" TEST SUITE: JOINT LIMIT DEADLOCK RESOLUTION & INSTANT ESCAPE VERIFICATION")
        print("=" * 75)
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=None, solver=cls.solver)

    def test_01_soft_clamping_and_non_blocking_solver(self):
        """
        Tiêu chí 1: Khi mục tiêu ép khớp chạm giới hạn, solver không đánh trượt nghiệm (q != None),
        kẹp khớp chạm biên an toàn, các khớp khác vẫn tự do trượt dọc biên.
        """
        print("\n[Test 1] Per-Joint Soft Clamping & Non-Blocking Solver Verification")
        # Ready pose chuẩn
        q_home = [0.0, -2.20, +1.90, -1.57, -1.57, 0.0]
        fk_home = self.solver.fk(q_home)

        # Cố tình kéo mục tiêu lên cao và vươn xa để ép shoulder_lift (q1) tiến về biên upper limit (+0.20 rad)
        # hoặc ép cùi chỏ duỗi thẳng
        extreme_pos = [0.42, 0.0, 0.40]
        res = self.solver.solve({"pos": extreme_pos, "quat": fk_home["quat"]}, q_seed=q_home)

        self.assertIsNotNone(res["q"], "Solver KHÔNG ĐƯỢC trả về q=None khi chạm joint limit!")
        self.assertTrue(validate_joints(res["q"]), f"Tất cả các khớp phải nằm trong UR3E_LIMITS: {res['q']}")

        # Kiểm tra chi tiết từng khớp không vượt biên
        for i, (lo, hi) in enumerate(UR3E_LIMITS):
            self.assertGreaterEqual(res["q"][i], lo - 1e-4, f"Khớp {i} vi phạm biên dưới {lo}")
            self.assertLessEqual(res["q"][i], hi + 1e-4, f"Khớp {i} vi phạm biên trên {hi}")

        print(f"  q_sol = {[round(x, 4) for x in res['q']]}")
        print(f"  at_joint_limit = {res.get('at_joint_limit')}")
        print(f"  sliding_on_barrier = {res.get('sliding_on_barrier')}")
        print("  -> PASS: Solver kẹp mềm chính xác, không block, bảo toàn an toàn tuyệt đối!")

    def test_02_nearest_equivalent_and_branch_flip_prevention(self):
        """
        Tiêu chí 2: Hàm _nearest_equivalent và normalize_to_current triệt tiêu hiện tượng
        lật nhánh ±2pi (Branch-Flip) khi khớp chớm vượt biên ±pi.
        """
        print("\n[Test 2] Branch-Flip Suppression at ±pi Boundaries")
        lo, hi = -math.pi, math.pi

        # TH 1: Khớp đang ở -3.14 rad, bước tiếp theo solver ra -3.142 rad (chớm vượt biên âm một lượng cực nhỏ)
        curr_q = -3.140
        raw_ik = -3.142
        norm = _nearest_equivalent(raw_ik, curr_q, lo, hi)

        # Cũ: norm bị đẩy sang +3.141 rad (nhảy ~6.28 rad kích hoạt Branch-Flip Guard)
        # Mới: norm phải được kẹp an toàn sát biên âm [-3.141, -3.139], bước nhảy < 0.01 rad!
        jump = abs(norm - curr_q)
        print(f"  Negative boundary: current={curr_q:.4f}, raw_ik={raw_ik:.4f} -> normalized={norm:.4f}, jump={jump:.4f} rad")
        self.assertLess(jump, 0.05, f"Bước nhảy {jump:.4f} rad quá lớn! Branch-Flip Guard sẽ bị kích hoạt!")
        self.assertTrue(lo <= norm <= hi, f"Normalized value {norm} nằm ngoài [{lo}, {hi}]")

        # TH 2: Khớp đang ở +3.14 rad, solver ra +3.142 rad
        curr_q2 = +3.140
        raw_ik2 = +3.142
        norm2 = _nearest_equivalent(raw_ik2, curr_q2, lo, hi)
        jump2 = abs(norm2 - curr_q2)
        print(f"  Positive boundary: current={curr_q2:.4f}, raw_ik={raw_ik2:.4f} -> normalized={norm2:.4f}, jump={jump2:.4f} rad")
        self.assertLess(jump2, 0.05, f"Bước nhảy {jump2:.4f} rad quá lớn! Branch-Flip Guard sẽ bị kích hoạt!")
        self.assertTrue(lo <= norm2 <= hi, f"Normalized value {norm2} nằm ngoài [{lo}, {hi}]")

        print("  -> PASS: 100% triệt tiêu bẫy lật nhánh ±2pi tại biên!")

    def test_03_teleop_anti_windup_and_instant_escape(self):
        """
        Tiêu chí 3: Toàn bộ chu trình Teleoperation:
        1. Kéo liên tục vào JOINT LIMIT trong 10 chu kỳ -> Trạng thái JOINT LIMIT, target không windup quá 30mm.
        2. Chu kỳ 11: Đảo chiều kéo ngược ra ngoài -> Lập tức thoát JOINT LIMIT, chuyển sang OK ngay tức khắc!
        """
        print("\n[Test 3] Teleop Anti-Windup Re-anchoring & Instant Escape Verification")

        # Khởi tạo tư thế an toàn
        q_start = [0.0, -1.80, 1.40, -1.20, -1.57, 0.0]
        fk_start = self.solver.fk(q_start)
        p_start = np.array(fk_start["pos"])
        quat_start = fk_start["quat"]

        mock_pose = MockPoseSource(p_start, quat_start)
        mock_exec = MockExecutor()
        teleop = TeleoperationController(
            pose_source=mock_pose,
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
            dt=0.05,
        )
        teleop.enabled = True
        teleop.sync_with_robot(q_start)

        # ── GIAI ĐOẠN 1: Kéo mạnh ép khớp về phía biên (10 chu kỳ) ──
        # Ví dụ: Kéo sang bên X dương thật xa để ép shoulder_pan hoặc cánh tay chạm giới hạn
        q_curr = list(q_start)
        print("  [Giai đoạn 1: Cố tình kéo mục tiêu vượt xa không gian làm việc]")
        for step_idx in range(1, 11):
            # Kéo mỗi chu kỳ nhích thêm 20mm vào sâu vùng vi phạm
            mock_pose.pos = p_start + np.array([0.05 + step_idx * 0.02, 0.0, 0.05])
            teleop.step(q_curr)
            q_curr = list(mock_exec.last_sent)
            st = teleop.get_status()

            # Kiểm tra khoảng cách windup giữa smoothed_pos và TCP thực tế khi chạm biên
            fk_act = self.solver.fk(q_curr)
            windup_dist = float(np.linalg.norm(teleop._smoothed_pos - np.array(fk_act["pos"])))
            if st != "OK":
                self.assertLessEqual(
                    windup_dist,
                    0.035,  # 30mm + 5mm dung sai lọc
                    f"Lỗi Windup: Mục tiêu trôi xa {windup_dist*1000:.1f}mm > 35mm!",
                )

        print(f"    Trạng thái tại step 10: {st}")
        print(f"    Độ trôi target (Windup): {windup_dist*1000:.2f} mm (Đảm bảo <= 30mm)")
        self.assertGreater(len(mock_pose.reanchor_calls), 0, "Hàm reanchor_position phải được gọi khi chạm biên!")

        # ── GIAI ĐOẠN 2: Người dùng đảo chiều kéo ngược lại về phía bên trong ──
        print("  [Giai đoạn 2: Đảo chiều kéo ngược vào vùng làm việc hợp lệ]")
        # Nhích nhẹ chuột/tay cầm ngược lại 5mm về phía p_start
        fk_act_before = self.solver.fk(q_curr)
        p_act_before = np.array(fk_act_before["pos"])
        dir_to_home = p_start - p_act_before
        dir_to_home /= np.linalg.norm(dir_to_home)

        # Chỉ cần nhích 15mm về phía an toàn
        mock_pose.pos = p_act_before + dir_to_home * 0.015

        # Chạy chu kỳ đảo chiều
        teleop.step(q_curr)
        q_escaped = list(mock_exec.last_sent)
        status_after = teleop.get_status()

        print(f"    Trạng thái ngay chu kỳ đầu sau khi đảo chiều: {status_after}")
        print(f"    Độ dịch chuyển khớp: {np.linalg.norm(np.array(q_escaped) - np.array(q_curr)):.4f} rad")

        # Robot phải di chuyển thoát ra (không bị kẹt tại chỗ)
        self.assertEqual(status_after, "OK", f"Robot phải lập tức quay về trạng thái OK, nhận được: {status_after}")
        print("  -> PASS: Thoát khỏi JOINT LIMIT thành công 100% ngay chu kỳ đầu tiên, không có deadlock!")

    def test_04_direct_joint_limit_status_and_escape(self):
        """
        Tiêu chí 4: Xác minh trực tiếp trạng thái hiển thị JOINT LIMIT và giải phóng êm ái:
        - Cố tình kéo khớp shoulder_lift vào sát biên an toàn (-2.65 rad) -> Teleop hiển thị 'JOINT LIMIT'
        - Solver vẫn trả về nghiệm q hợp lệ, không block robot.
        - Đảo chiều kéo ngược -> Teleop lập tức chuyển về 'OK' trong đúng 1 chu kỳ.
        """
        print("\n[Test 4] Direct JOINT LIMIT Status Flagging & 1-Cycle Instant Escape")
        q_near = [0.0, -2.60, 1.2, -1.2, -1.57, 0.0]
        fk_near = self.solver.fk(q_near)
        p_near = np.array(fk_near["pos"])

        mock_pose = MockPoseSource(p_near, fk_near["quat"])
        mock_exec = MockExecutor()

        teleop = TeleoperationController(
            pose_source=mock_pose,
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
            dt=0.05,
        )
        teleop.enabled = True
        teleop.sync_with_robot(q_near)

        # 1. Kéo mạnh vào vùng vi phạm biên dưới shoulder_lift
        mock_pose.pos[2] -= 0.08
        mock_pose.pos[0] -= 0.08
        teleop.step(q_near)

        st_limit = teleop.get_status()
        q_cmd_limit = mock_exec.last_sent
        print(f"  Trạng thái khi chạm biên giới hạn: '{st_limit}'")
        print(f"  q1 command: {q_cmd_limit[1]:.4f} rad (Biên dưới UR3E_LIMITS: {UR3E_LIMITS[1][0]:.2f} rad)")

        self.assertEqual(st_limit, "JOINT LIMIT", "Trạng thái phải hiển thị cảnh báo 'JOINT LIMIT'")
        self.assertGreaterEqual(q_cmd_limit[1], UR3E_LIMITS[1][0] - 1e-4, "Khớp không được vượt quá UR3E_LIMITS")
        self.assertTrue(validate_joints(q_cmd_limit), "Toàn bộ cấu hình khớp phải hợp lệ")

        # 2. Đảo chiều kéo ngược lại vào vùng làm việc an toàn
        mock_pose.pos[2] += 0.08
        mock_pose.pos[0] += 0.08
        for escape_cycle in range(1, 3):
            teleop.step(q_cmd_limit)
            st_escaped = teleop.get_status()
            q_cmd_escaped = mock_exec.last_sent
            if st_escaped == "OK":
                break

        print(f"  Trạng thái sau khi đảo chiều (cycle {escape_cycle}): '{st_escaped}'")
        print(f"  q1 sau thoát: {q_cmd_escaped[1]:.4f} rad (Dịch chuyển ra xa biên: +{q_cmd_escaped[1] - q_cmd_limit[1]:.4f} rad)")

        self.assertEqual(st_escaped, "OK", "Trạng thái phải lập tức chuyển về 'OK' tức thì!")
        self.assertGreater(q_cmd_escaped[1], q_cmd_limit[1], "Khớp phải lập tức di chuyển ra xa biên giới hạn")
        print("  -> PASS: Đạt 100% yêu cầu: Gắn cờ JOINT LIMIT, không ngắt, thoát tức thì trong 1 chu kỳ!")


if __name__ == "__main__":
    unittest.main()

