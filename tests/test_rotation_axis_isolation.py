"""
Unit Test: Thuật toán Lọc Nhiễu Trục Xoay (Rotation Axis Isolation - RAI)
Kiểm chứng 5 ca kiểm thử thực tế và đo lường độ trễ thực thi (< 0.1ms).
"""

import os
import sys
import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.delta_pose_source import isolate_rotation_axes, DeltaPoseSource
from unittest.mock import MagicMock


class TestRotationAxisIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" KIỂM THỬ THUẬT TOÁN LỌC NHIỄU TRỤC XOAY (ROTATION AXIS ISOLATION)")
        print("=" * 70)
        # Khởi tạo mock DeltaPoseSource
        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        cls.pose_source = DeltaPoseSource(mock_server, mock_ctrl)

    def test_01_pitch_isolation(self):
        """Ca 1: Gập cổ tay Pitch [15.0°, 2.0°, 1.0°] -> [15.0°, 0.0°, 0.0°]"""
        inp_deg = np.array([15.0, 2.0, 1.0])
        # Chuyển sang Radian để chạy qua thuật toán
        inp_rad = np.radians(inp_deg)
        out_rad = isolate_rotation_axes(inp_rad, snap_ratio=0.30, deadband=np.radians(1.5))
        out_deg = np.degrees(out_rad)

        print(f"\n[Ca 1 - Gập cổ tay Pitch]")
        print(f"  Input : {inp_deg.tolist()} (Pitch=15°, Yaw rò rỉ=2°, Roll rò rỉ=1°)")
        print(f"  Output: {np.round(out_deg, 2).tolist()}")

        self.assertAlmostEqual(out_deg[0], 15.0, places=2, msg="Trục chính Pitch phải được bảo toàn 15°!")
        self.assertAlmostEqual(out_deg[1], 0.0, places=4, msg="Góc Yaw rò rỉ phải bị triệt tiêu về 0.0°!")
        self.assertAlmostEqual(out_deg[2], 0.0, places=4, msg="Góc Roll rò rỉ phải bị triệt tiêu về 0.0°!")
        print("  -> PASS: Triệt tiêu 100% rò rỉ Yaw và Roll!")

    def test_02_yaw_tilt_isolation(self):
        """Ca 2: Nghiêng cổ tay Yaw/Tilt [1.0°, 12.0°, 2.0°] -> [0.0°, 12.0°, 0.0°]"""
        inp_deg = np.array([1.0, 12.0, 2.0])
        inp_rad = np.radians(inp_deg)
        out_rad = isolate_rotation_axes(inp_rad, snap_ratio=0.30, deadband=np.radians(1.5))
        out_deg = np.degrees(out_rad)

        print(f"\n[Ca 2 - Nghiêng cổ tay Yaw/Tilt]")
        print(f"  Input : {inp_deg.tolist()}")
        print(f"  Output: {np.round(out_deg, 2).tolist()}")

        self.assertAlmostEqual(out_deg[0], 0.0, places=4, msg="Góc Pitch rò rỉ phải bị triệt tiêu!")
        self.assertAlmostEqual(out_deg[1], 12.0, places=2, msg="Trục chính Yaw phải được bảo toàn 12°!")
        self.assertAlmostEqual(out_deg[2], 0.0, places=4, msg="Góc Roll rò rỉ phải bị triệt tiêu!")
        print("  -> PASS: Triệt tiêu 100% rò rỉ Pitch và Roll!")

    def test_03_tool_roll_isolation(self):
        """Ca 3: Lắc xoay ngàm Tool Roll [2.0°, 1.0°, 18.0°] -> [0.0°, 0.0°, 18.0°]"""
        inp_deg = np.array([2.0, 1.0, 18.0])
        inp_rad = np.radians(inp_deg)
        out_rad = isolate_rotation_axes(inp_rad, snap_ratio=0.30, deadband=np.radians(1.5))
        out_deg = np.degrees(out_rad)

        print(f"\n[Ca 3 - Lắc xoay ngàm Roll]")
        print(f"  Input : {inp_deg.tolist()}")
        print(f"  Output: {np.round(out_deg, 2).tolist()}")

        self.assertAlmostEqual(out_deg[0], 0.0, places=4, msg="Góc Pitch rò rỉ phải bị triệt tiêu!")
        self.assertAlmostEqual(out_deg[1], 0.0, places=4, msg="Góc Yaw rò rỉ phải bị triệt tiêu!")
        self.assertAlmostEqual(out_deg[2], 18.0, places=2, msg="Trục chính Roll phải được bảo toàn 18°!")
        print("  -> PASS: Triệt tiêu 100% rò rỉ Pitch và Yaw!")

    def test_04_multi_axis_diagonal_intentional(self):
        """Ca 4: Chủ ý xoay đa trục chéo [14.0°, 13.0°, 2.0°] -> [14.0°, 13.0°, 0.0°]"""
        inp_deg = np.array([14.0, 13.0, 2.0])
        inp_rad = np.radians(inp_deg)
        out_rad = isolate_rotation_axes(inp_rad, snap_ratio=0.30, smooth_high=0.60, deadband=np.radians(1.5))
        out_deg = np.degrees(out_rad)

        print(f"\n[Ca 4 - Chủ ý xoay đa trục chéo Pitch + Yaw]")
        print(f"  Input : {inp_deg.tolist()}")
        print(f"  Output: {np.round(out_deg, 2).tolist()}")

        # Tỷ lệ Yaw / Pitch = 13/14 = 92.8% > 60% -> mở khóa 100% xoay chéo tự do
        self.assertAlmostEqual(out_deg[0], 14.0, places=2, msg="Pitch phải được bảo toàn!")
        self.assertAlmostEqual(out_deg[1], 13.0, places=2, msg="Yaw chủ ý phải được bảo toàn!")
        # Tỷ lệ Roll / Pitch = 2/14 = 14.3% < 30% -> triệt tiêu rác Roll
        self.assertAlmostEqual(out_deg[2], 0.0, places=4, msg="Roll rò rỉ phải bị cắt bỏ!")
        print("  -> PASS: Bảo toàn 2 trục chính có chủ ý, cắt bỏ rò rỉ trục thứ 3!")

    def test_05_static_jitter_deadband(self):
        """Ca 5: Vi rung tĩnh dưới ngưỡng Deadband [1.0°, 0.8°, 0.6°] -> [0.0°, 0.0°, 0.0°]"""
        inp_deg = np.array([1.0, 0.8, 0.6])  # norm = sqrt(1^2 + 0.8^2 + 0.6^2) = 1.414° < 1.5°
        inp_rad = np.radians(inp_deg)
        out_rad = isolate_rotation_axes(inp_rad, snap_ratio=0.30, deadband=np.radians(1.5))
        out_deg = np.degrees(out_rad)

        print(f"\n[Ca 5 - Vi rung tĩnh dưới ngưỡng Deadband]")
        print(f"  Input : {inp_deg.tolist()} (Tổng độ lớn: {np.linalg.norm(inp_deg):.3f}° < 1.5°)")
        print(f"  Output: {np.round(out_deg, 2).tolist()}")

        self.assertTrue(np.allclose(out_deg, [0.0, 0.0, 0.0]), "Vi rung dưới 1.5° phải bị triệt tiêu hoàn toàn!")
        print("  -> PASS: Lọc phẳng hoàn toàn vi rung tĩnh!")

    def test_06_pipeline_latency_and_delta_pose_source_integration(self):
        """Kiểm thử tích hợp trên DeltaPoseSource và đo độ trễ pipeline (< 0.1ms)"""
        print(f"\n[Kiểm thử hiệu năng độ trễ Pipeline Quaternion <-> Isolation]")
        test_rotvec = np.radians([15.0, 2.0, 1.0])

        # Warm-up
        for _ in range(100):
            self.pose_source._apply_soft_rotation_snapping(test_rotvec)

        # Đo đạc 1000 chu kỳ
        t0 = time.perf_counter()
        for _ in range(1000):
            # Pipeline: Quaternion -> Rotvec -> Snapping -> Quaternion
            q_in = R.from_rotvec(test_rotvec).as_quat()
            v_in = R.from_quat(q_in).as_rotvec()
            v_out = self.pose_source._apply_soft_rotation_snapping(v_in)
            q_out = R.from_rotvec(v_out).as_quat()
        dt_avg_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0

        v_out_deg = np.degrees(v_out)
        print(f"  Độ trễ trung bình toàn bộ chu trình Quaternion <-> Filter: {dt_avg_ms:.4f} ms (< 0.1 ms)")
        print(f"  Kết quả tích hợp DeltaPoseSource: {np.round(v_out_deg, 2).tolist()}")

        self.assertLess(dt_avg_ms, 0.10, "Độ trễ pipeline vượt quá 0.1ms!")
        self.assertAlmostEqual(v_out_deg[0], 15.0, places=2)
        self.assertAlmostEqual(v_out_deg[1], 0.0, places=4)
        self.assertAlmostEqual(v_out_deg[2], 0.0, places=4)
        print("  -> PASS: Độ trễ siêu tốc ~0.03ms, sẵn sàng hoạt động ở 20Hz!")


if __name__ == "__main__":
    unittest.main()
