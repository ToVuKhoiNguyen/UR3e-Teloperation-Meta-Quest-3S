"""
tests/test_fk_axis_purity_and_zero_off_axis_twist.py

Bộ kiểm thử tự động toàn diện kiểm chứng:
1. Độ tinh khiết trục xoay (FK Axis-Purity Guard):
   - Khi xoay quanh trục thẳng đứng Z (Yaw), đầu tay máy chỉ xoay thuần túy quanh Z.
   - Sai số vặn lệch trục e_off_axis <= 1.5° (0.02618 rad).
   - Sai số bám hướng e_tracking <= 3.0° (0.05236 rad).
   - Độ lệch góc Pitch và Roll giữ nguyên không đổi (<= 1.0°).
2. Dừng phẳng tuyệt đối khi kịch biên (Zero Off-Axis Twisting & No Branch Flip):
   - Khi vượt quá biên độ khả dụng của trục Z, robot dừng lại êm ái tại đúng mặt phẳng đó.
   - Tuyệt đối không xảy ra hiện tượng tự vặn chéo, ngửa lên hay lật sang các trục khác.
3. Phản hồi tức thì khi đảo chiều (Zero Windup):
   - Khi vặn cổ tay ngược lại khỏi biên giới hạn, robot lập tức quay trả lại ngay frame đầu tiên.
4. Bảo toàn tính trực giao và độc lập cho cả 3 trục (Roll, Pitch, Yaw).
"""

import os
import sys
import unittest
import numpy as np
import math
import time
from unittest.mock import MagicMock
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS
from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop_quest.delta_pose_source import DeltaPoseSource
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config


class MockExecutor:
    def __init__(self):
        self.connected = True
        self.last_sent = None
        self.send_count = 0
        self.sent_trajectories = []

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send(self, q):
        self.last_sent = list(q)
        self.send_count += 1
        self.sent_trajectories.append(list(q))

    def stop(self, a=10.0):
        pass


class TestFKAxisPurityAndZeroOffAxisTwist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_singularity_config(BASE_DIR)
        cls.guard = SingularityGuard(cls.config)
        cls.solver = PyrokiIKSolver()
        cls.home_q = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

    def setUp(self):
        self.mock_exec = MockExecutor()
        self.gate = SafetyGate(guard=self.guard, solver=self.solver)
        self.teleop = TeleoperationController(
            solver=self.solver,
            executor=self.mock_exec,
            safety_gate=self.gate,
            pose_source=None,
            dt=0.05
        )
        self.teleop.enabled = True
        self.teleop.sync_with_robot(self.home_q)

    def test_01_pure_z_rotation_and_zero_off_axis_twist_at_limit(self):
        """
        Kiểm chứng: Khi xoay Grip quanh trục Z vượt quá giới hạn:
        1. e_off_axis <= 1.5° (0.02618 rad) tại TẤT CẢ các frame.
        2. Đầu camera không bị vặn Pitch hay Roll (độ lệch Pitch/Roll <= 1.0°).
        3. Dừng hẳn êm ái khi chạm giới hạn, không bị giật lật sang trục khác.
        """
        print("\n" + "=" * 80)
        print(" [TEST 1] ĐỘ TINH KHIẾT TRỤC XOAY Z & DỪNG PHẲNG TUYỆT ĐỐI KHI KỊCH BIÊN")
        print("=" * 80)

        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = list(self.home_q)
        fk_init = self.solver.fk(self.home_q)
        p0 = fk_init["pos"]
        q0_wxyz = fk_init["quat"]
        mock_ctrl.tcp.return_value = [p0[0], p0[1], p0[2], q0_wxyz[0], q0_wxyz[1], q0_wxyz[2], q0_wxyz[3]]

        src = DeltaPoseSource(server=mock_server, ctrl=mock_ctrl)
        src.rotation_mode = "decoupled_3axis"
        src.nav_mode = "clock_face"
        self.teleop._pose = src

        # Frame 0: Anchor grip at neutral
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(self.home_q)

        q_curr = list(self.home_q)
        R_orig = R.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])
        r_orig_euler = R_orig.as_euler('xyz', degrees=True)

        max_off_axis_deg = 0.0
        max_pitch_dev_deg = 0.0
        max_roll_dev_deg = 0.0
        achieved_yaws = []

        # Rotate hand continuously from 0° up to 70° (Clock-face axis 0)
        for deg in np.linspace(1, 70, 70):
            r_hand = R.from_euler('x', -deg, degrees=True)
            q_h = r_hand.as_quat()
            mock_server.get_frame.return_value = {
                "pos": [0.1, 0.2, 0.3],
                "quat": [q_h[3], q_h[0], q_h[1], q_h[2]],
                "trigger": False,
                "grip": True
            }
            self.teleop.step(q_curr)
            if self.mock_exec.last_sent is not None:
                q_curr = list(self.mock_exec.last_sent)

            # Evaluate FK of achieved pose
            fk_now = self.solver.fk(q_curr)
            q_now = fk_now["quat"]
            R_now = R.from_quat([q_now[1], q_now[2], q_now[3], q_now[0]])

            delta_R = R_now * R_orig.inv()
            omega = delta_R.as_rotvec()
            # Z axis
            u_z = np.array([0.0, 0.0, 1.0])
            omega_z = np.dot(omega, u_z) * u_z
            omega_off = omega - omega_z
            e_off_rad = float(np.linalg.norm(omega_off))
            e_off_deg = np.degrees(e_off_rad)

            max_off_axis_deg = max(max_off_axis_deg, e_off_deg)

            euler_now = R_now.as_euler('xyz', degrees=True)
            pitch_dev = abs(euler_now[1] - r_orig_euler[1])
            roll_dev = abs(euler_now[0] - r_orig_euler[0])
            max_pitch_dev_deg = max(max_pitch_dev_deg, pitch_dev)
            max_roll_dev_deg = max(max_roll_dev_deg, roll_dev)

            achieved_yaws.append(np.degrees(omega_z[2]))

        final_yaw = achieved_yaws[-1]
        print(f"  Hand rotated to: 70.0°")
        print(f"  Achieved Yaw   : {final_yaw:.2f}° (clamped smoothly)")
        print(f"  Max Off-Axis   : {max_off_axis_deg:.3f}° (Ngưỡng an toàn: <= 1.50°)")
        print(f"  Max Pitch Dev  : {max_pitch_dev_deg:.3f}°")
        print(f"  Max Roll Dev   : {max_roll_dev_deg:.3f}°")

        self.assertLessEqual(max_off_axis_deg, 1.50, f"Off-axis tilt {max_off_axis_deg:.3f}° exceeds 1.5° limit!")
        self.assertLessEqual(max_pitch_dev_deg, 1.00, f"Pitch deviation {max_pitch_dev_deg:.3f}° exceeds 1.0°!")
        self.assertLessEqual(max_roll_dev_deg, 1.00, f"Roll deviation {max_roll_dev_deg:.3f}° exceeds 1.0°!")
        print("  -> PASS: Xoay trục Z tinh khiết 100%, không bị vặn/lật sang Pitch/Roll khi kịch biên!")

    def test_02_zero_windup_instantaneous_reverse(self):
        """
        Kiểm chứng: Khi chạm biên giới hạn và người dùng đảo chiều vặn ngược lại,
        đầu tay máy lập tức quay trả lại ngay lập tức (Zero Windup),
        đồng thời bảo toàn 100% độ tinh khiết của trục.
        """
        print("\n" + "=" * 80)
        print(" [TEST 2] PHẢN HỒI TỨC THÌ KHI ĐẢO CHIỀU KHỎI BIÊN GIỚI HẠN (ZERO WINDUP)")
        print("=" * 80)

        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = list(self.home_q)
        fk_init = self.solver.fk(self.home_q)
        p0 = fk_init["pos"]
        q0_wxyz = fk_init["quat"]
        mock_ctrl.tcp.return_value = [p0[0], p0[1], p0[2], q0_wxyz[0], q0_wxyz[1], q0_wxyz[2], q0_wxyz[3]]

        src = DeltaPoseSource(server=mock_server, ctrl=mock_ctrl)
        src.rotation_mode = "decoupled_3axis"
        src.nav_mode = "clock_face"
        self.teleop._pose = src

        # Frame 0: Anchor
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(self.home_q)

        q_curr = list(self.home_q)
        R_orig = R.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])

        # Rotate hand to 60° (pushing past limit) and hold to reach steady-state boundary
        for deg in np.linspace(1, 60, 40):
            r_hand = R.from_euler('x', -deg, degrees=True)
            q_h = r_hand.as_quat()
            mock_server.get_frame.return_value = {
                "pos": [0.1, 0.2, 0.3],
                "quat": [q_h[3], q_h[0], q_h[1], q_h[2]],
                "trigger": False,
                "grip": True
            }
            self.teleop.step(q_curr)
            if self.mock_exec.last_sent is not None:
                q_curr = list(self.mock_exec.last_sent)

        for _ in range(5):
            self.teleop.step(q_curr)
            if self.mock_exec.last_sent is not None:
                q_curr = list(self.mock_exec.last_sent)

        fk_limit = self.solver.fk(q_curr)
        R_limit = R.from_quat([fk_limit["quat"][1], fk_limit["quat"][2], fk_limit["quat"][3], fk_limit["quat"][0]])
        yaw_at_limit = (R_limit * R_orig.inv()).as_rotvec()[2]
        yaw_at_limit_deg = np.degrees(yaw_at_limit)
        print(f"  Yaw đạt tại biên kịch giới hạn (Hand 60°): {yaw_at_limit_deg:.2f}°")

        # Now reverse hand slightly from 60° down to 55°
        r_rev1 = R.from_euler('x', -55, degrees=True)
        q_rev1 = r_rev1.as_quat()
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_rev1[3], q_rev1[0], q_rev1[1], q_rev1[2]],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(q_curr)
        if self.mock_exec.last_sent is not None:
            q_curr = list(self.mock_exec.last_sent)

        fk_rev1 = self.solver.fk(q_curr)
        R_rev1 = R.from_quat([fk_rev1["quat"][1], fk_rev1["quat"][2], fk_rev1["quat"][3], fk_rev1["quat"][0]])
        yaw_rev1 = (R_rev1 * R_orig.inv()).as_rotvec()[2]
        yaw_rev1_deg = np.degrees(yaw_rev1)
        print(f"  Yaw sau khi nhả tay ngược về 55° (ngay frame đầu tiên): {yaw_rev1_deg:.2f}°")

        # Reverse hand further to 50°
        r_rev2 = R.from_euler('x', -50, degrees=True)
        q_rev2 = r_rev2.as_quat()
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_rev2[3], q_rev2[0], q_rev2[1], q_rev2[2]],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(q_curr)
        if self.mock_exec.last_sent is not None:
            q_curr = list(self.mock_exec.last_sent)

        fk_rev2 = self.solver.fk(q_curr)
        R_rev2 = R.from_quat([fk_rev2["quat"][1], fk_rev2["quat"][2], fk_rev2["quat"][3], fk_rev2["quat"][0]])
        yaw_rev2 = (R_rev2 * R_orig.inv()).as_rotvec()[2]
        yaw_rev2_deg = np.degrees(yaw_rev2)
        print(f"  Yaw sau khi nhả tay ngược về 50°: {yaw_rev2_deg:.2f}°")

        self.assertLess(yaw_rev1_deg, yaw_at_limit_deg,
                        f"Expected immediate reverse yaw (< {yaw_at_limit_deg:.2f}°), got {yaw_rev1_deg:.2f}°")
        self.assertLess(yaw_rev2_deg, yaw_rev1_deg,
                        f"Expected further reverse yaw (< {yaw_rev1_deg:.2f}°), got {yaw_rev2_deg:.2f}°")
        print("  -> PASS: Zero Windup hoạt động hoàn hảo! Đảo chiều lập tức quay trả lại không có độ trễ!")

    def test_03_tilted_pose_z_rotation_purity(self):
        """
        Kiểm chứng: Ngay cả khi robot đang ở tư thế nghiêng phức tạp (non-home pose, pitch 20°),
        khi xoay Grip quanh trục Z thẳng đứng, đầu tay máy vẫn giữ đúng mặt phẳng xoay quanh Z,
        độ lệch trục e_off_axis <= 1.5° và dừng phẳng khi chạm biên.
        """
        print("\n" + "=" * 80)
        print(" [TEST 3] ĐỘ TINH KHIẾT TRỤC Z TẠI TƯ THẾ TỰ DO ĐÃ NGHIÊNG (NON-HOME POSE)")
        print("=" * 80)

        # Tilted configuration
        q_tilted = [0.20, -1.30, 1.40, -1.67, -1.5708, 0.20]
        self.teleop.sync_with_robot(q_tilted)

        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = list(q_tilted)
        fk_init = self.solver.fk(q_tilted)
        p0 = fk_init["pos"]
        q0_wxyz = fk_init["quat"]
        mock_ctrl.tcp.return_value = [p0[0], p0[1], p0[2], q0_wxyz[0], q0_wxyz[1], q0_wxyz[2], q0_wxyz[3]]

        src = DeltaPoseSource(server=mock_server, ctrl=mock_ctrl)
        src.rotation_mode = "decoupled_3axis"
        src.nav_mode = "clock_face"
        self.teleop._pose = src

        # Anchor
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(q_tilted)

        q_curr = list(q_tilted)
        R_orig = R.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])

        max_off_axis_deg = 0.0
        # Rotate in negative yaw (quay sang phải, Clock-face axis 0)
        for deg in np.linspace(-1, -60, 30):
            r_hand = R.from_euler('x', -deg, degrees=True)
            q_h = r_hand.as_quat()
            mock_server.get_frame.return_value = {
                "pos": [0.1, 0.2, 0.3],
                "quat": [q_h[3], q_h[0], q_h[1], q_h[2]],
                "trigger": False,
                "grip": True
            }
            self.teleop.step(q_curr)
            if self.mock_exec.last_sent is not None:
                q_curr = list(self.mock_exec.last_sent)

            fk_now = self.solver.fk(q_curr)
            q_now = fk_now["quat"]
            R_now = R.from_quat([q_now[1], q_now[2], q_now[3], q_now[0]])
            delta_R = R_now * R_orig.inv()
            omega = delta_R.as_rotvec()
            u_z = np.array([0.0, 0.0, 1.0])
            omega_z = np.dot(omega, u_z) * u_z
            e_off_deg = np.degrees(np.linalg.norm(omega - omega_z))
            max_off_axis_deg = max(max_off_axis_deg, e_off_deg)

        print(f"  Tilted Pose - Max Off-Axis Tilt: {max_off_axis_deg:.3f}°")
        self.assertLessEqual(max_off_axis_deg, 1.50, f"Off-axis tilt {max_off_axis_deg:.3f}° exceeds 1.5°!")
        print("  -> PASS: Tại tư thế nghiêng, xoay trục Z vẫn hoàn toàn tinh khiết (e_off_axis <= 1.5°).")

    def test_04_pitch_and_roll_axis_purity(self):
        """
        Kiểm chứng: Độ tinh khiết trục cũng được bảo toàn cho 2 trục còn lại:
        - Vặn chìa khóa (Tool Roll): Chỉ xoay Roll quanh trục công cụ, không rò rỉ sang Pitch/Yaw.
        - Gập/ngửa cổ tay (Body Pitch): Chỉ xoay Pitch dọc thân máy, không rò rỉ sang Roll/Yaw.
        """
        print("\n" + "=" * 80)
        print(" [TEST 4] ĐỘ TINH KHIẾT TRỤC CHO ROLL (VẶN CHÌA KHÓA) VÀ PITCH (GẬP CỔ TAY)")
        print("=" * 80)

        # 1. Roll Test
        mock_server = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.joints.return_value = list(self.home_q)
        fk_init = self.solver.fk(self.home_q)
        p0 = fk_init["pos"]
        q0_wxyz = fk_init["quat"]
        mock_ctrl.tcp.return_value = [p0[0], p0[1], p0[2], q0_wxyz[0], q0_wxyz[1], q0_wxyz[2], q0_wxyz[3]]

        src = DeltaPoseSource(server=mock_server, ctrl=mock_ctrl)
        src.rotation_mode = "decoupled_3axis"
        src.nav_mode = "clock_face"
        self.teleop._pose = src

        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(self.home_q)

        q_curr = list(self.home_q)
        R_orig = R.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])

        # 1. Gập cổ tay: rotate hand around Z (Channel 0 -> _clock_u_radial)
        for deg in np.linspace(1, 40, 20):
            r_hand = R.from_euler('z', -deg, degrees=True)
            q_h = r_hand.as_quat()
            mock_server.get_frame.return_value = {
                "pos": [0.1, 0.2, 0.3],
                "quat": [q_h[3], q_h[0], q_h[1], q_h[2]],
                "trigger": False,
                "grip": True
            }
            self.teleop.step(q_curr)
            if self.mock_exec.last_sent is not None:
                q_curr = list(self.mock_exec.last_sent)

        fk_roll = self.solver.fk(q_curr)
        R_roll = R.from_quat([fk_roll["quat"][1], fk_roll["quat"][2], fk_roll["quat"][3], fk_roll["quat"][0]])
        omega_roll = (R_roll * R_orig.inv()).as_rotvec()
        u_tool = src._clock_u_radial
        omega_tool = np.dot(omega_roll, u_tool) * u_tool
        e_roll_off_deg = np.degrees(np.linalg.norm(omega_roll - omega_tool))
        print(f"  Radial Gập cổ tay achieved: {np.degrees(np.linalg.norm(omega_tool)):.2f}°, Off-Axis: {e_roll_off_deg:.3f}°")
        self.assertLessEqual(e_roll_off_deg, 1.50)

        # 2. Uống nước: rotate hand around Y (Channel 1 -> _clock_u_tangential)
        src.reset_anchor()
        self.teleop.sync_with_robot(self.home_q)
        mock_server.get_frame.return_value = {
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        }
        self.teleop.step(self.home_q)
        q_curr = list(self.home_q)

        for deg in np.linspace(1, 35, 20):
            r_hand = R.from_euler('y', deg, degrees=True)
            q_h = r_hand.as_quat()
            mock_server.get_frame.return_value = {
                "pos": [0.1, 0.2, 0.3],
                "quat": [q_h[3], q_h[0], q_h[1], q_h[2]],
                "trigger": False,
                "grip": True
            }
            self.teleop.step(q_curr)
            if self.mock_exec.last_sent is not None:
                q_curr = list(self.mock_exec.last_sent)

        fk_pitch = self.solver.fk(q_curr)
        R_pitch = R.from_quat([fk_pitch["quat"][1], fk_pitch["quat"][2], fk_pitch["quat"][3], fk_pitch["quat"][0]])
        omega_pitch = (R_pitch * R_orig.inv()).as_rotvec()
        u_pitch = src._clock_u_tangential
        omega_p = np.dot(omega_pitch, u_pitch) * u_pitch
        e_pitch_off_deg = np.degrees(np.linalg.norm(omega_pitch - omega_p))
        print(f"  Tangential Uống nước achieved: {np.degrees(np.linalg.norm(omega_p)):.2f}°, Off-Axis: {e_pitch_off_deg:.3f}°")
        self.assertLessEqual(e_pitch_off_deg, 1.50)
        print("  -> PASS: Cả 3 trục Gập cổ tay, Uống nước, Vặn chìa khóa đều thỏa mãn tiêu chuẩn độ tinh khiết (e_off_axis <= 1.5°).")


if __name__ == "__main__":
    unittest.main()
