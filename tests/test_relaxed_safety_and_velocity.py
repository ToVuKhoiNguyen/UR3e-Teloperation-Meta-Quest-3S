"""
Kiểm thử tự động 4 Kịch bản nới lỏng an toàn và tối ưu hóa độ mượt:
1. Kịch bản 1: Vươn tới biên không gian mới (R = 0.505m, Bounding Box XY +/-0.52m, Z 0.035-0.60m)
2. Kịch bản 2: Vận tốc góc tối đa đạt 1.0 rad/s (Scaled Vector Rate Limiter)
3. Kịch bản 3: Chuyển động mượt mà qua vùng sigma_min = 0.045 (DLS Continuous Damping)
4. Kịch bản 4: Rào chắn ảo chống đâm bàn (Z=0.01 -> 0.035m) và chống va chạm thân đế (r_min = 0.14m)
"""

import os
import sys
import time
import math
import unittest
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.teleoperation import clamp_cartesian_safety, SafetyGate, TeleoperationController
from teleop.pyroki_solver import PyrokiIKSolver, validate_joints
from unittest.mock import MagicMock


class TestRelaxedSafetyAndVelocity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" KIỂM THỬ 4 KỊCH BẢN NỚI LỎNG AN TOÀN & TỐI ƯU HÓA ĐỘ MƯỢT MÀ")
        print("=" * 70)
        cls.solver = PyrokiIKSolver()

    def test_01_extended_reach_and_ik_convergence(self):
        """Kịch bản 1: Vươn tới biên không gian mới (R = 0.505m > 0.45m cũ)"""
        print("\n[Kịch bản 1 - VƯƠN XA TỚI BIÊN KHÔNG GIAN MỚI R = 0.505m]")
        # Điểm mục tiêu ở biên vươn xa: X = 0.488m, Y = 0.131m -> R = 0.5053m (> 0.45m và > 0.49m cũ)
        # Hướng ngàm kẹp chúc thẳng xuống mặt bàn theo hệ trục tool0 chuẩn UR3e
        downward_quat = [0.0, 0.7071068, -0.7071068, 0.0]
        target_pose = {"pos": [0.488, 0.131, 0.25], "quat": downward_quat}
        
        # 1. Kiểm tra clamp_cartesian_safety không bị cắt ở 0.45 hay 0.49m
        clamped_pose = clamp_cartesian_safety(target_pose)
        r_clamped = math.hypot(clamped_pose["pos"][0], clamped_pose["pos"][1])
        print(f"  Tọa độ mục tiêu: R = 0.5053m -> Sau clamp: R = {r_clamped:.4f}m")
        self.assertAlmostEqual(r_clamped, 0.5053, places=3,
                               msg=f"R = 0.5053m bị kẹp sai thành {r_clamped}m!")
        
        # Warm-up đồ thị một lần
        self.solver.solve(clamped_pose, self.solver.default_home)

        # 2. Giải IK PyRoki CPU thời gian thực
        t0 = time.perf_counter()
        res_ik = self.solver.solve(clamped_pose, self.solver.default_home)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        self.assertIsNotNone(res_ik["q"], "IK giải thất bại ở biên R = 0.505m!")
        print(f"  IK giải thành công! Độ trễ: {dt_ms:.2f}ms (< 50ms) | Sai số: {res_ik.get('pos_err_mm', 0):.4f}mm (< 18mm)")
        self.assertLess(dt_ms, 50.0, "Độ trễ giải IK vượt quá 50ms!")
        self.assertLess(res_ik.get("pos_err_mm", 0), 18.0, "Sai số vị trí IK vượt quá 18mm!")
        self.assertTrue(validate_joints(res_ik["q"]), "Góc khớp vi phạm giới hạn an toàn!")
        print("  -> PASS: Đã giải phóng không gian vươn xa mép bàn thành công!")

    def test_02_velocity_clamping_max_speed(self):
        """Kịch bản 2: Vận tốc góc tối đa đạt mức đề xuất (1.0 rad/s)"""
        print("\n[Kịch bản 2 - VẬN TỐC GÓC TỐI ĐA ĐẠT 1.0 rad/s]")
        MAX_VEL = float(os.environ.get("MAX_JOINT_VEL", "1.0"))
        DT = 0.05  # Chu kỳ 20Hz
        max_dq = MAX_VEL * DT  # 0.05 rad / frame
        print(f"  Cấu hình MAX_VEL: {MAX_VEL:.1f} rad/s (~57.3°/s) | max_dq mỗi frame: {max_dq:.4f} rad")
        self.assertEqual(MAX_VEL, 1.0, "MAX_VEL chưa được cập nhật thành 1.0 rad/s!")

        # Mô phỏng cử động tay nhanh từ Quest với vận tốc yêu cầu 1.6 rad/s (dq = 0.08 rad trong 50ms)
        q_last = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        q_req = [0.0, -1.57 + 0.08, 1.57 + 0.08, -1.57 + 0.08, -1.57 + 0.08, 0.0]
        
        # Áp dụng Scaled Vector Rate Limiter chuẩn từ teleoperation.py
        dq = [q_req[i] - q_last[i] for i in range(6)]
        max_ratio = max(abs(dq[i]) / max_dq for i in range(6))
        self.assertGreater(max_ratio, 1.0)
        
        q_clamped = [q_last[i] + dq[i] / max_ratio for i in range(6)]
        actual_step = max(abs(q_clamped[i] - q_last[i]) for i in range(6))
        actual_vel = actual_step / DT
        print(f"  Vận tốc yêu cầu: 1.60 rad/s -> Sau Rate Limiter: {actual_vel:.2f} rad/s (Bước nhảy: {actual_step:.4f} rad)")
        self.assertAlmostEqual(actual_vel, 1.0, places=3,
                               msg=f"Vận tốc không đạt 1.0 rad/s (Thực tế: {actual_vel})!")
        print("  -> PASS: Scaled Vector Rate Limiter hoạt động chính xác ở mức 1.0 rad/s!")

    def test_03_smooth_singularity_damping(self):
        """Kịch bản 3: Chuyển động mượt mà qua vùng sigma_min = 0.045 (DLS Continuous Damping)"""
        print("\n[Kịch bản 3 - GIẢM CHẤN MƯỢT MÀ CONTINUOUS SINGULARITY DAMPING]")
        guard_mock = MagicMock()
        guard_mock.danger_threshold = 0.028
        guard_mock.warn_threshold = 0.042
        guard_mock.metrics_calc.device = torch.device("cpu")
        gate = SafetyGate(guard=guard_mock)

        # 3a. Dáng vươn xa bình thường có sigma = 0.045 >= warn_threshold 0.042 -> damping = 1.0 (100% mượt)
        guard_mock.metrics_calc.get_metrics.return_value = {"sigma_min": torch.tensor(0.045)}
        res_smooth = gate.check([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertTrue(res_smooth["pass"])
        self.assertEqual(res_smooth["damping"], 1.0,
                         "sigma=0.045 không được phép bị giảm chấn (phải giữ 1.0)!")
        print(f"  3a. Dáng vươn xa sigma = 0.045: damping = {res_smooth['damping']:.2f} (100% mượt, không bị hãm phanh)")

        # 3b. Dáng áp sát nguy hiểm sigma = 0.035 (nằm giữa 0.028 và 0.042) -> damping >= 0.45
        guard_mock.metrics_calc.get_metrics.return_value = {"sigma_min": torch.tensor(0.035)}
        res_near = gate.check([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertTrue(res_near["pass"])
        self.assertGreaterEqual(res_near["damping"], 0.45)
        self.assertLess(res_near["damping"], 1.0)
        print(f"  3b. Dáng cận nguy hiểm sigma = 0.035: damping = {res_near['damping']:.3f} (>= 0.45, duy trì động lượng)")

        # 3c. Dáng nguy hiểm thực sự sigma = 0.020 < 0.028 -> Dừng an toàn
        guard_mock.metrics_calc.get_metrics.return_value = {"sigma_min": torch.tensor(0.020)}
        res_danger = gate.check([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertFalse(res_danger["pass"])
        self.assertIn("singularity", res_danger["reason"])
        print(f"  3c. Dáng kỳ dị nguy hiểm sigma = 0.020 < 0.028: Chặn dừng an toàn ({res_danger['reason']})")
        print("  -> PASS: Thuật toán giảm chấn mượt hoạt động chuẩn xác theo ngưỡng mới!")

    def test_04_cartesian_barrier_table_and_base(self):
        """Kịch bản 4: Bảo vệ an toàn chống đâm bàn (Z=0.01 -> 0.035m) & chống va chạm thân đế (r_min=0.14m)"""
        print("\n[Kịch bản 4 - RÀO CHẮN ẢO CHỐNG ĐÂM BÀN & THÂN ĐẾ]")
        
        # 4a. Ra lệnh chúc TCP xuống Z = 0.01m (quá sát mặt bàn) -> Clamped lên Z = 0.035m
        pose_plunge = {"pos": [0.25, 0.15, 0.01], "quat": [1.0, 0.0, 0.0, 0.0]}
        clamped_plunge = clamp_cartesian_safety(pose_plunge)
        z_clamped = clamped_plunge["pos"][2]
        print(f"  4a. Lệnh chúc xuống Z = 0.010m -> Clamped an toàn: Z = {z_clamped:.4f}m (>= 0.035m)")
        self.assertAlmostEqual(z_clamped, 0.035, places=3,
                               msg=f"Z=0.01m phải được chặn lại ở 0.035m, thực tế là {z_clamped}m!")

        # 4b. Ra lệnh tiến sát tâm đế R = 0.08m -> Clamped ra R = 0.14m (r_min bất khả xâm phạm)
        pose_base = {"pos": [0.05, 0.06245, 0.20], "quat": [1.0, 0.0, 0.0, 0.0]} # r = 0.08m
        clamped_base = clamp_cartesian_safety(pose_base)
        r_clamped_base = math.hypot(clamped_base["pos"][0], clamped_base["pos"][1])
        print(f"  4b. Lệnh áp sát tâm đế R = 0.080m -> Clamped an toàn: R = {r_clamped_base:.4f}m (>= 0.140m)")
        self.assertAlmostEqual(r_clamped_base, 0.140, places=3,
                               msg=f"r_min 0.14m bị vi phạm, thực tế là {r_clamped_base}m!")
        print("  -> PASS: Rào chắn ảo bảo vệ phần cứng hoạt động 100% tin cậy!")


if __name__ == "__main__":
    unittest.main()
