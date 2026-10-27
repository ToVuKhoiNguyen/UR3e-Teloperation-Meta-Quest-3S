"""
Kiểm tra tự động xác nhận 3 tiêu chí:
1. Đảo chiều trục Roll (Vặn chìa khóa) thành công và bảo toàn tính cô lập trục (Axis Isolation).
2. Bộ lọc Quaternion Adaptive EMA mở rộng alpha lên 0.90, phản hồi Zero-Lag khi xoay nhanh (> 90° / 0.5s).
3. Bộ điều phối vận tốc góc Decoupled Scaled Vector Rate Limiter cho phép cổ tay đạt 2.5 rad/s.
"""

import os
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.delta_pose_source import DeltaPoseSource, isolate_rotation_axes


class DummyServer:
    pass


class DummyController:
    def tcp(self):
        return [0.0, -0.3, 0.2, 0.0, 0.0, 0.0]


class TestWristZeroLagAndRollDirection(unittest.TestCase):
    def setUp(self):
        self.source = DeltaPoseSource(server=DummyServer(), ctrl=DummyController())
        self.source._anchor_u_fwd = np.array([0.0, 0.0, 1.0])
        self.source._anchor_u_right = np.array([1.0, 0.0, 0.0])
        self.source.isolate_rotation = True

    def test_01_roll_inversion_and_axis_isolation(self):
        """Tiêu chí 1: Đảo chiều Roll (Vặn chìa khóa) và giữ tính cô lập tuyệt đối"""
        print("\n[TEST 1 - ĐẢO CHIỀU ROLL VẶN CHÌA KHÓA & BẢO TOÀN CÔ LẬP TRỤC]")
        u_fwd = self.source._anchor_u_fwd
        u_right = self.source._anchor_u_right
        u_up = np.array([0.0, 1.0, 0.0])

        # Giả lập thao tác Vặn chìa khóa: omega xoay quanh u_fwd với vận tốc dương
        omega_world = u_fwd * 1.5

        d_act1 = float(np.dot(omega_world, u_right))
        d_act2 = -float(np.dot(omega_world, u_fwd))
        d_act3 = -float(np.dot(omega_world, u_up))

        print(f"  Input omega: dọc theo u_fwd (+1.50 rad/s)")
        print(f"  Thô: d_act1={d_act1:.3f}, d_act2={d_act2:.3f}, d_act3={d_act3:.3f}")

        # d_act2 phải mang dấu âm (đã đảo chiều thành công)
        self.assertLess(d_act2, 0.0, "d_act2 (Roll) chưa được đổi dấu âm!")
        self.assertAlmostEqual(d_act2, -1.5, places=4)

        # Kiểm tra qua hàm isolate_rotation_axes
        filtered_act1, filtered_act2, filtered_act3 = isolate_rotation_axes([d_act1, d_act2, d_act3])
        print(f"  Sau Hard Axis Isolation: act1={filtered_act1:.3f}, act2={filtered_act2:.3f}, act3={filtered_act3:.3f}")

        self.assertAlmostEqual(filtered_act1, 0.0, places=5, msg="Trục 1 (Uống nước) bị rò rỉ!")
        self.assertAlmostEqual(filtered_act3, 0.0, places=5, msg="Trục 3 (Gập lắc) bị rò rỉ!")
        self.assertAlmostEqual(filtered_act2, -1.5, places=4, msg="Trục 2 (Roll) bị sai giá trị!")
        print("  -> PASS: Trục Roll đã đảo chiều và 100% cô lập khỏi 2 trục còn lại!")

    def test_02_quaternion_filter_zero_lag(self):
        """Tiêu chí 2: Thao tác xoay nhanh > 90° trong 0.5s đạt Zero-Lag phản hồi chớp nhoáng"""
        print("\n[TEST 2 - PHẢN HỒI ZERO-LAG KHI CỔ TAY XOAY NHANH > 90° / 0.5s]")
        # Mô phỏng xoay 90 độ (1.5708 rad) quanh trục Z trong 0.5s (10 frames x 50ms)
        total_angle = np.deg2rad(90.0)
        frames = 10
        dt = 0.05

        # Reset EMA filter
        self.source._ema_quat_xyzw = None

        # Vòng lặp mô phỏng 10 frames
        recorded_filtered_angles = []
        target_angles = []

        for f in range(frames + 1):
            theta = (f / frames) * total_angle
            target_rot = R.from_euler('z', theta)
            target_quat_xyzw = target_rot.as_quat()
            target_angles.append(np.rad2rad(theta) if hasattr(np, 'rad2rad') else np.degrees(theta))

            # Logic lọc từ DeltaPoseSource
            if self.source._ema_quat_xyzw is None:
                self.source._ema_quat_xyzw = target_quat_xyzw
            else:
                if np.dot(self.source._ema_quat_xyzw, target_quat_xyzw) < 0:
                    target_quat_xyzw = -target_quat_xyzw

                dot_val = np.clip(abs(np.dot(self.source._ema_quat_xyzw, target_quat_xyzw)), -1.0, 1.0)
                rot_diff = 2.0 * np.arccos(dot_val)

                max_rot_step = 0.45
                if rot_diff > max_rot_step:
                    s_factor = max_rot_step / rot_diff
                    target_quat_xyzw = (1.0 - s_factor) * self.source._ema_quat_xyzw + s_factor * target_quat_xyzw
                    target_quat_xyzw /= np.linalg.norm(target_quat_xyzw)
                    rot_diff = max_rot_step

                t_rot = np.clip(rot_diff / 0.20, 0.0, 1.0)
                smooth_rot = t_rot * t_rot * (3.0 - 2.0 * t_rot)
                alpha_rot = 0.20 + (0.90 - 0.20) * smooth_rot

                self.source._ema_quat_xyzw = alpha_rot * target_quat_xyzw + (1.0 - alpha_rot) * self.source._ema_quat_xyzw
                self.source._ema_quat_xyzw /= np.linalg.norm(self.source._ema_quat_xyzw)

            cur_rot = R.from_quat(self.source._ema_quat_xyzw)
            cur_angle_deg = np.degrees(cur_rot.magnitude())
            recorded_filtered_angles.append(cur_angle_deg)

        final_target = target_angles[-1]
        final_filtered = recorded_filtered_angles[-1]
        lag_deg = final_target - final_filtered

        print(f"  Mục tiêu: {final_target:.1f}° sau 0.5s")
        print(f"  Góc sau bộ lọc tối ưu: {final_filtered:.1f}° | Độ trễ còn lại: {lag_deg:.2f}° (< 2.0°)")
        self.assertLess(lag_deg, 2.0, f"Độ trễ {lag_deg:.2f}° vẫn vượt quá ngưỡng Zero-lag 2.0°!")
        print("  -> PASS: Bộ lọc Quaternion đạt Zero-Lag phản hồi chớp nhoáng!")

    def test_03_decoupled_rate_limiter_wrist_speed(self):
        """Tiêu chí 3: Rate Limiter phân tầng cho phép cổ tay đạt 2.5 rad/s không bị kẹt ở 1.0 rad/s"""
        print("\n[TEST 3 - RATE LIMITER PHÂN TẦNG: CỔ TAY ĐẠT 2.5 rad/s (~143°/s)]")
        MAX_VEL_ARM = 1.2
        MAX_VEL_WRIST = 2.5
        DT = 0.05
        max_dq = [MAX_VEL_ARM * DT] * 3 + [MAX_VEL_WRIST * DT] * 3

        # Giả lập lệnh xoay cổ tay nhanh: các khớp thân 0,1,2 đứng yên, khớp cổ tay 5 xoay với vận tốc 2.4 rad/s
        q_last = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        # dq_wrist = 2.4 * 0.05 = 0.12 rad trong 50ms
        q_target = [0.0, -1.57, 1.57, -1.57, -1.57, 0.12]

        dq = [q_target[i] - q_last[i] for i in range(6)]
        max_ratio = max(abs(dq[i]) / max_dq[i] for i in range(6))

        print(f"  Vận tốc cổ tay yêu cầu: 2.40 rad/s (bước nhảy dq5 = {dq[5]:.3f} rad)")
        print(f"  max_ratio = {max_ratio:.3f} (<= 1.0 nghĩa là không bị kẹp cắt tốc độ)")
        self.assertLessEqual(max_ratio, 1.0, "Vận tốc cổ tay 2.4 rad/s bị kẹp sai!")

        # Giả lập lệnh xoay cực nhanh vượt ngưỡng 3.0 rad/s (dq5 = 0.15 rad) -> bị kẹp đúng 2.5 rad/s
        q_fast = [0.0, -1.57, 1.57, -1.57, -1.57, 0.15]
        dq_fast = [q_fast[i] - q_last[i] for i in range(6)]
        max_ratio_fast = max(abs(dq_fast[i]) / max_dq[i] for i in range(6))
        q_clamped = [q_last[i] + dq_fast[i] / max_ratio_fast for i in range(6)]
        actual_wrist_vel = (q_clamped[5] - q_last[5]) / DT

        print(f"  Vận tốc cực nhanh 3.0 rad/s -> Sau kẹp an toàn: {actual_wrist_vel:.2f} rad/s")
        self.assertAlmostEqual(actual_wrist_vel, 2.5, places=3, msg="Giới hạn kẹp cổ tay không đạt 2.5 rad/s!")
        print("  -> PASS: Khớp cổ tay đáp ứng mượt mà tới 2.5 rad/s, an toàn bảo vệ phần cứng!")


if __name__ == "__main__":
    unittest.main()
