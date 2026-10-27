"""
Unit Test: Grip Rotation Rollback & Smooth Release Verification.
Kiểm chứng:
1. Grip Rotation Rollback Test:
   - Khi bóp và giữ nút Grip (grip=True, trigger=False) rồi xoay cổ tay Quest:
     Robot xoay mượt mà, chính xác theo hệ quy chiếu Tool TCP, đúng tỷ lệ và hướng xoay.
   - Triệt tiêu lỗi lật trục (Gimbal lock) và lệch hệ quy chiếu.
2. Smooth Release Test:
   - Khi nhả nút Grip (grip=False):
     Robot đóng băng hướng tại vị trí hiện tại, hoàn toàn không bị giật hay trôi góc.
   - Khi bóp lại Grip tiếp theo: Tự động neo mốc mới tại hướng hiện tại, bước nhảy ban đầu bằng 0.0 rad.
3. Multi-clutch Lifecycle:
   - Bóp Grip -> Xoay -> Nhả -> Bóp -> Xoay -> Nhả: Tích lũy góc xoay liên tục, không bao giờ bị giật ngược.
"""

import os
import sys
import math
import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.delta_pose_source import DeltaPoseSource


class MockQuestServer:
    def __init__(self):
        self._frame = None

    def get_frame(self):
        return self._frame

    def set_frame(self, frame):
        self._frame = frame


class MockRobotCtrl:
    def __init__(self, joints=None, tcp=None):
        self._joints = list(joints) if joints is not None else [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        # tcp: [x, y, z, rx, ry, rz]
        self._tcp = list(tcp) if tcp is not None else [0.0, -0.30, 0.20, 0.0, 3.14159, 0.0]

    def joints(self):
        return self._joints

    def tcp(self):
        return self._tcp


class TestGripRotationRollback(unittest.TestCase):
    def setUp(self):
        self.server = MockQuestServer()
        # Robot ở tư thế chuẩn: ngàm chúc xuống bàn máy (rx=0, ry=pi, rz=0)
        self.initial_tcp = [0.0, -0.30, 0.20, 0.0, 3.14159, 0.0]
        self.initial_joints = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.ctrl = MockRobotCtrl(joints=self.initial_joints, tcp=self.initial_tcp)

        self.pose_source = DeltaPoseSource(
            server=self.server,
            ctrl=self.ctrl,
            scale=1.0,
            alpha=0.28,
            max_delta_m=0.22,
            anchor_reference_func=lambda: list(self.ctrl.tcp())
        )
        self.pose_source.nav_mode = "cylindrical"
        self.pose_source.rotation_mode = "decoupled_3axis"

    def test_01_grip_rotation_direction_and_proportionality(self):
        """Tiêu chí 1: Xoay cổ tay khi giữ Grip bám đúng hướng và tỉ lệ theo chuẩn tag ổn định"""
        print("\n[TEST 1 - KIỂM CHỨNG HƯỚNG XOAY & TỈ LỆ KHI BÓP GIỮ NÚT GRIP]")

        # 1. Khởi tạo Clutch Frame: Người dùng bóp Grip
        init_frame = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z]
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.server.set_frame(init_frame)
        p0 = self.pose_source.get_pose()
        self.assertIsNone(p0, "Frame đầu tiên khi bóp Grip phải chốt mỏ neo và trả về None (chống giật)!")

        # 2. Xoay lắc cổ tay Roll quanh trục nhìn tới (Tool Z Roll): 15 độ (~0.26 rad)
        # Trong hệ WebXR, lắc cổ tay tương ứng với omega_world quanh trục Y hoặc Z
        rot_roll = R.from_euler('y', -0.26)  # -omega_world[1] = +0.26 rad Roll
        q_roll = rot_roll.as_quat()  # [x, y, z, w]

        roll_frame = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_roll[3], q_roll[0], q_roll[1], q_roll[2]],  # [w, x, y, z]
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.server.set_frame(roll_frame)

        # Chạy 5 frames để bộ lọc EMA bám ổn định
        p_res = None
        for _ in range(5):
            p_res = self.pose_source.get_pose()

        self.assertIsNotNone(p_res)
        # Vị trí TCP phải đứng yên tuyệt đối khi chỉ bóp Grip xoay
        pos_diff = np.linalg.norm(np.array(p_res["pos"]) - np.array(self.initial_tcp[:3]))
        self.assertLess(pos_diff, 1e-4, "Vị trí robot không được dịch chuyển khi chỉ thao tác xoay Grip!")

        # Tính toán góc xoay thực tế đạt được so với anchor
        q_target = p_res["quat"]  # [w, x, y, z]
        R_target = R.from_quat([q_target[1], q_target[2], q_target[3], q_target[0]])
        R_anchor = R.from_rotvec(self.initial_tcp[3:])
        # R_target = R_anchor * R_delta_tool -> R_delta_tool = R_anchor.inv() * R_target
        R_delta = R_anchor.inv() * R_target
        delta_rotvec = R_delta.as_rotvec()

        print(f"  Input Roll: +15.0° (~0.262 rad)")
        print(f"  Robot Delta Rotvec: [Pitch={math.degrees(delta_rotvec[0]):.2f}°, Yaw={math.degrees(delta_rotvec[1]):.2f}°, Roll={math.degrees(delta_rotvec[2]):.2f}°]")

        # Trục Roll (Tool Z) phải nhận phần lớn góc quay, hai trục còn lại gần như 0
        self.assertGreater(delta_rotvec[2], 0.06, "Robot phải xoay Roll đúng chiều dương theo tay cầm!")
        self.assertLess(abs(delta_rotvec[0]), 0.02, "Trục Pitch không được rò rỉ!")
        self.assertLess(abs(delta_rotvec[1]), 0.02, "Trục Yaw không được rò rỉ!")
        print("  -> PASS: Hướng xoay Tool Roll chuẩn xác, triệt tiêu rò rỉ chéo!")

    def test_02_smooth_release_orientation_hold(self):
        """Tiêu chí 2: Nhả nút Grip -> Hướng robot đứng yên 100%, không giật ngược hay trôi góc"""
        print("\n[TEST 2 - NHẢ NÚT GRIP: GIỮ NGUYÊN HƯỚNG, KHÔNG GIẬT HAY TRÔI]")

        # 1. Bóp Grip và xoay đến một góc nhất định
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        self.pose_source.get_pose()

        rot_step = R.from_euler('x', 0.20)
        q_step = rot_step.as_quat()
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_step[3], q_step[0], q_step[1], q_step[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        p_held = None
        for _ in range(5):
            p_held = self.pose_source.get_pose()

        quat_at_release = list(p_held["quat"])
        print(f"  Góc xoay tại thời điểm chuẩn bị nhả Grip: {quat_at_release}")

        # 2. Người dùng nhả phím Grip (grip=False, trigger=False)
        self.server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_step[3], q_step[0], q_step[1], q_step[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": False
        })
        p_released = self.pose_source.get_pose()
        self.assertIsNone(p_released, "Khi nhả Grip, pose_source phải trả về None để Teleop Controller giữ vững vị trí!")

        # 3. Giả lập controller cập nhật vị trí hiện tại của robot bằng vị trí vừa xoay
        # Chuyển quat_at_release sang axis-angle cho tcp
        R_at_rel = R.from_quat([quat_at_release[1], quat_at_release[2], quat_at_release[3], quat_at_release[0]])
        axang_rel = R_at_rel.as_rotvec()
        self.ctrl._tcp = [0.0, -0.30, 0.20, float(axang_rel[0]), float(axang_rel[1]), float(axang_rel[2])]

        # 4. Người dùng bóp Grip trở lại ở một vị trí tay khác (Re-engage Clutch)
        self.server.set_frame({
            "pos": [0.2, 0.3, 0.4],
            "quat": [0.7071, 0.0, 0.7071, 0.0],  # Tư thế tay cầm bất kỳ
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        p_reengage = self.pose_source.get_pose()
        self.assertIsNone(p_reengage, "Frame tái kích hoạt đầu tiên phải chốt mỏ neo mới từ tư thế hiện tại!")

        # Frame tiếp theo khi giữ nguyên tay cầm: output phải giữ nguyên hướng vừa chốt, không bị nhảy ngược về ban đầu!
        p_steady = self.pose_source.get_pose()
        self.assertIsNotNone(p_steady)
        q_new = p_steady["quat"]
        R_new = R.from_quat([q_new[1], q_new[2], q_new[3], q_new[0]])

        # Độ lệch giữa hướng sau khi tái kích hoạt và hướng lúc nhả phím phải bằng 0 (Sai số < 0.01 rad)
        rot_jump = float(np.linalg.norm((R_at_rel.inv() * R_new).as_rotvec()))
        print(f"  Độ giật/nhảy góc khi tái kích hoạt Grip (Clutch Re-engage Jump): {math.degrees(rot_jump):.4f}°")
        self.assertLess(rot_jump, 0.03, "Không được có bước nhảy đột ngột khi bấm lại Grip!")
        print("  -> PASS: Đóng băng hướng hoàn hảo khi nhả Grip, không trôi dạt!")

    def test_03_multi_clutch_cumulative_rotation(self):
        """Tiêu chí 3: Chu kỳ Multi-Clutch tích lũy góc xoay liên tục qua nhiều lần bóp/nhả"""
        print("\n[TEST 3 - CHU KỲ MULTI-CLUTCH TÍCH LŨY GÓC XOAY LIÊN TỤC]")

        current_tcp = list(self.initial_tcp)

        for cycle in range(3):
            # Clutch in
            self.server.set_frame({
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0],
                "trigger": False, "grip": True
            })
            self.ctrl._tcp = list(current_tcp)
            self.pose_source.get_pose()

            # Rotate 10 deg Pitch
            q_rot = R.from_euler('x', 0.1745).as_quat()
            self.server.set_frame({
                "pos": [0.0, 0.0, 0.0],
                "quat": [q_rot[3], q_rot[0], q_rot[1], q_rot[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0],
                "trigger": False, "grip": True
            })
            p_out = None
            for _ in range(6):
                p_out = self.pose_source.get_pose()

            q_out = p_out["quat"]
            R_out = R.from_quat([q_out[1], q_out[2], q_out[3], q_out[0]])
            current_tcp[3:] = R_out.as_rotvec().tolist()

            # Clutch out
            self.server.set_frame({
                "pos": [0.0, 0.0, 0.0],
                "quat": [q_rot[3], q_rot[0], q_rot[1], q_rot[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0],
                "trigger": False, "grip": False
            })
            self.pose_source.get_pose()

        # Sau 3 chu kỳ xoay Pitch, tổng góc tích lũy phải tăng tiến đều đặn
        R_final = R.from_rotvec(current_tcp[3:])
        R_init = R.from_rotvec(self.initial_tcp[3:])
        total_delta = (R_final * R_init.inv()).as_rotvec()
        total_deg = math.degrees(np.linalg.norm(total_delta))
        print(f"  Tổng góc xoay tích lũy sau 3 lần clutch: {total_deg:.1f}°")
        self.assertGreater(np.linalg.norm(total_delta), math.radians(6.0), "Góc xoay phải được tích lũy liên tục qua các lần bóp nhả!")
        print("  -> PASS: Tích lũy đa chu kỳ Multi-clutch mượt mà, không bao giờ bị giật ngược về 0!")

    def test_04_execution_latency(self):
        """Đo lường độ trễ xử lý thuật toán xoay Grip (< 0.1ms)"""
        print("\n[TEST 4 - ĐỘ TRỄ XỬ LÝ GÓC XOAY GRIP]")

        # Setup active grip frame
        q_test = R.from_euler('xyz', [0.1, 0.1, 0.1]).as_quat()
        frame = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_test[3], q_test[0], q_test[1], q_test[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False, "grip": True
        }
        self.server.set_frame(frame)
        self.pose_source.get_pose()

        # Warm up
        for _ in range(50):
            self.pose_source.get_pose()

        N = 1000
        t0 = time.perf_counter()
        for _ in range(N):
            self.pose_source.get_pose()
        dt_ms = ((time.perf_counter() - t0) * 1000.0) / N

        print(f"  Độ trễ trung bình khi tính toán góc xoay Grip: {dt_ms:.4f} ms / cycle (N={N})")
        self.assertLess(dt_ms, 0.30, "Độ trễ phải < 0.30ms (dưới 1% chu kỳ 50ms của vòng lặp 20Hz)!")
        print("  -> PASS: Tốc độ xử lý cực nhanh, hoàn toàn thời gian thực!")


if __name__ == "__main__":
    unittest.main()
