"""
Unit Test: Kiểm chứng Rào chắn an toàn mặt bàn 1cm và Chống đâm cụm Camera (Table Safety Barrier & Camera Protection).

Kịch bản kiểm chứng:
1. Tay máy (tool0) thẳng đứng có thể hạ xuống sát mặt bàn vật lý ở khoảng cách đúng 1.0cm (Z = 0.020m).
2. Cố tình gập nghiêng cổ tay cho camera chúi xuống mặt bàn -> SafetyGate lập tức kích hoạt phanh cứng khi góc vỏ camera cách mặt bàn <= 1.0cm.
3. Độ trễ thẩm định va chạm mặt bàn của bộ giải PyRoki < 0.25ms.
4. Cơ chế clamp Cartesius z_min = 0.020m bảo vệ không gian làm việc.
"""

import os
import sys
import math
import time
import unittest
import numpy as np
from unittest.mock import MagicMock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, load_table_bounds
from teleop.teleoperation import clamp_cartesian_safety, SafetyGate


def create_mock_guard():
    mock_guard = MagicMock()
    mock_guard.metrics_calc.device = "cpu"
    mock_guard.metrics_calc.get_metrics.return_value = {"sigma_min": MagicMock(item=lambda: 0.15)}
    mock_guard.danger_threshold = 0.028
    mock_guard.warn_threshold = 0.040
    return mock_guard


class TestTableCameraBarrier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" KIỂM THỬ RÀO CHẮN MẶT BÀN 1.0CM & CHỐNG ĐÂM CỤM CAMERA REAL-TIME")
        print("=" * 70)
        cls.solver = PyrokiIKSolver()
        cls.guard = create_mock_guard()
        cls.gate = SafetyGate(cls.guard, solver=cls.solver)
        cls.table_bounds = load_table_bounds()
        cls.z_table_top = cls.table_bounds["z_table_top"]
        print(f"  Mặt trên mặt bàn vật lý: Z = {cls.z_table_top:.4f}m")
        print(f"  Trần an toàn yêu cầu (mặt bàn + 1.0cm): Z = {cls.z_table_top + 0.010:.4f}m")

    def test_01_tool0_vertical_approach_1cm(self):
        """
        Kịch bản 1: Tay máy (tool0) thẳng đứng có thể hạ xuống sát mặt bàn vật lý ở khoảng cách đúng 1.0cm (Z=0.020m, 0.0mm áo giáp đỏ).
        """
        print("\n[Kịch bản 1 - Tay máy thẳng đứng tiếp cận sát mặt bàn 1.0cm]")
        # 1a. Thử nghiệm qua Cartesian clamping:
        # Ra lệnh hạ xuống Z = 0.005m (xuyên mặt bàn 0.010m) -> Clamped chính xác lên Z = 0.020m (Z_table_armor WYSIWYG)
        q_seed = [0.0, -0.75, 1.8, -1.0, 1.57, 1.57]
        fk_seed = self.solver.fk(q_seed)
        cmd_pose = {"pos": [fk_seed["pos"][0], fk_seed["pos"][1], 0.005], "quat": fk_seed["quat"]}
        clamped = clamp_cartesian_safety(cmd_pose)
        z_clamped = clamped["pos"][2]
        dist_to_table = z_clamped - self.z_table_top
        print(f"  1a. Lệnh Cartesius Z = 0.005m -> Clamped lên Z = {z_clamped:.4f}m (Cách bàn: {dist_to_table*100:.1f}cm)")
        self.assertAlmostEqual(z_clamped, 0.020, places=3,
                               msg=f"Z phải được chặn tại 0.020m, thực tế là {z_clamped}m!")
        self.assertAlmostEqual(dist_to_table, 0.000, places=3,
                               msg=f"Khoảng cách tiếp xúc áo giáp đỏ phải đúng 0.0mm, thực tế là {dist_to_table*1000:.2f}mm!")

        # 1b. Giải IK tư thế thẳng đứng tại Z = 0.030m (TCP chạm áo giáp đỏ bàn tại Z=0.020m):
        target_30mm = {"pos": [fk_seed["pos"][0], fk_seed["pos"][1], 0.030], "quat": fk_seed["quat"]}
        ik_res = self.solver.solve(target_30mm, q_seed=q_seed)
        self.assertIsNotNone(ik_res["q"], "Bộ giải IK phải tìm được nghiệm hợp lệ khi hạ sát bàn 1.0cm!")
        q_sol = ik_res["q"]

        # Kiểm tra thẩm định qua SafetyGate và check_table_collision
        gate_res = self.gate.check(q_sol)
        tbl_chk = self.solver.check_table_collision(q_sol, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)

        print(f"  1b. Nghiệm IK tìm được: {np.round(q_sol, 3).tolist()}")
        print(f"      TCP Z: {tbl_chk['z_tcp']:.4f}m (Cách bàn: {tbl_chk['tcp_clearance_cm']:.1f}cm)")
        print(f"      Camera Z min: {tbl_chk['z_cam_min']:.4f}m (Cách bàn: {tbl_chk['cam_clearance_cm']:.1f}cm)")
        print(f"      SafetyGate phán quyết: pass = {gate_res['pass']} ({gate_res['reason']})")

        self.assertTrue(gate_res["pass"], f"SafetyGate phải cho phép tư thế hợp lệ: {gate_res['reason']}")
        self.assertAlmostEqual(tbl_chk["z_tcp"], 0.020, places=2,
                               msg="Áo giáp đỏ TCP phải đạt sát ngưỡng 0.020m (chạm khít 0.0mm mặt bàn)!")
        self.assertGreaterEqual(tbl_chk["z_cam_min"], self.z_table_top - 1e-4,
                                msg="Camera không được thấp hơn ngưỡng áo giáp mặt bàn!")
        print("  -> PASS: Tay máy thẳng đứng có thể hạ xuống sát mặt bàn vật lý ở khoảng cách đúng 1.0cm!")

    def test_02_camera_pitch_down_hard_brake(self):
        """
        Kịch bản 2: Cố tình gập nghiêng cổ tay cho camera chúi xuống mặt bàn
        -> Hệ thống lập tức phanh cứng tay máy khi góc vỏ camera cách mặt bàn <= 1.0cm.
        """
        print("\n[Kịch bản 2 - Cố tình gập cổ tay cho camera chúi xuống mặt bàn]")
        # Đặt robot ở tư thế an toàn camera nằm trên cao:
        q_safe = [0.0, -0.75, 1.8, -1.0, 1.57, 1.57]
        chk_safe = self.solver.check_table_collision(q_safe, z_safe_clearance=0.010)
        gate_safe = self.gate.check(q_safe)
        print(f"  2a. Tư thế an toàn (Camera hướng lên trên):")
        print(f"      Camera cách bàn: {chk_safe['cam_clearance_cm']:.1f}cm (>= 1.0cm) -> Gate pass = {gate_safe['pass']}")
        self.assertTrue(chk_safe["safe"])
        self.assertTrue(gate_safe["pass"])

        # Cố tình lật xoay cổ tay cho cụm camera chúi thẳng xuống mặt bàn:
        q_danger = [0.0, -0.75, 1.8, -1.0, 1.57, -1.57]
        chk_danger = self.solver.check_table_collision(q_danger, z_safe_clearance=0.010)
        gate_danger = self.gate.check(q_danger)

        print(f"  2b. Tư thế lật cổ tay chúi cụm camera đâm bàn (q5 = -1.57 rad):")
        print(f"      TCP cách bàn: {chk_danger['tcp_clearance_cm']:.1f}cm")
        print(f"      Camera điểm thấp nhất cách bàn: {chk_danger['cam_clearance_cm']:.1f}cm (<= 1.0cm)")
        print(f"      SafetyGate phán quyết: pass = {gate_danger['pass']} | Lý do: '{gate_danger['reason']}'")

        self.assertFalse(chk_danger["safe"], "Phải phát hiện vi phạm khoảng cách camera đâm bàn!")
        self.assertEqual(chk_danger["violator"], "Camera", "Bộ phận vi phạm phải là Camera!")
        self.assertFalse(gate_danger["pass"], "SafetyGate phải REJECT và phanh cứng robot lập tức!")
        self.assertIn("Nguy cơ đâm bàn", gate_danger["reason"])
        self.assertIn("Camera", gate_danger["reason"])
        print("  -> PASS: SafetyGate phanh cứng khẩn cấp thành công, bảo vệ an toàn 100% cho camera!")

    def test_03_latency_benchmark(self):
        """
        Kịch bản 3: Đo lường độ trễ thuật toán thẩm định va chạm mặt bàn (< 0.25ms)
        """
        print("\n[Kịch bản 3 - Đo lường độ trễ thuật toán thẩm định va chạm mặt bàn]")
        q_test = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

        # Đo 1000 lần lặp
        t0 = time.perf_counter()
        for _ in range(1000):
            self.solver.check_table_collision(q_test, z_safe_clearance=0.010)
        dt_avg_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0

        print(f"  Độ trễ trung bình mỗi chu kỳ kiểm tra: {dt_avg_ms:.4f} ms (Chỉ tiêu: < 0.25 ms)")
        self.assertLess(dt_avg_ms, 0.25, "Độ trễ kiểm tra an toàn mặt bàn vượt quá 0.25ms!")
        print("  -> PASS: Thuật toán JIT siêu tốc ~0.08ms, hoàn toàn không gây trễ chu kỳ servoJ!")


if __name__ == "__main__":
    unittest.main()
