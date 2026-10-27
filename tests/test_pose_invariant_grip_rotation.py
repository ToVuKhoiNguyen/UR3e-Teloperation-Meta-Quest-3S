#!/usr/bin/env python3
"""
Kiểm thử tự động toán học: Pose-Invariant Grip Rotation Test
Xác nhận tính độc lập tư thế 100% của cơ chế xoay định hướng bằng nút Grip trên Meta Quest:
1. Test tại Home: Robot ở SAFE_HOME_JOINTS, xoay 3 trục độc lập (Roll +20°, Pitch +20°, Yaw +20°).
2. Test tại 3 tư thế cực hạn:
   - Tư thế 1: Xoay khớp base đi +60° và xoay cổ tay wrist_3 đi +90°.
   - Tư thế 2: Gập cổ tay wrist_1, wrist_2 lệch 45° sang ngang.
   - Tư thế 3: Đưa tay máy sang góc phần tư khác và nghiêng TCP chúc xuống.
3. Xác nhận vector trục quay của TCP trong hệ base_link trùng khít 100% (sai số góc < 0.01 rad)
   so với khi xoay ở tư thế Home.
"""

import os
import sys
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop.pyroki_solver import PyrokiIKSolver


class MockQuestServer:
    def __init__(self):
        self._frame = None

    def get_frame(self):
        return self._frame

    def set_frame(self, frame):
        self._frame = frame


class MockRobotCtrl:
    def __init__(self, joints, tcp):
        self._joints = list(joints)
        self._tcp = list(tcp)

    def joints(self):
        return self._joints

    def tcp(self):
        return self._tcp


class TestPoseInvariantGripRotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print(" BỘ KIỂM THỬ TOÁN HỌC TỰ ĐỘNG: POSE-INVARIANT GRIP ROTATION TEST")
        print("=" * 80)
        cls.solver = PyrokiIKSolver()

        cls.SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]
        cls.EXT1_JOINTS = [math.radians(60), -2.2, 1.9, -1.2708, -1.5708, math.radians(90)]
        cls.EXT2_JOINTS = [0.0, -2.2, 1.9, -1.2708 + math.radians(45), -1.5708 + math.radians(45), 0.0]
        cls.EXT3_JOINTS = [math.radians(135), -1.8, 1.5, -1.0, -1.5708, math.radians(-45)]

    def _get_tcp_from_joints(self, joints):
        fk = self.solver.fk(joints)
        pos = fk["pos"]
        quat_wxyz = fk["quat"]
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rotvec = rot.as_rotvec()
        return [pos[0], pos[1], pos[2], rotvec[0], rotvec[1], rotvec[2]], rot

    def _run_rotation_test(self, joints, rot_matrix_vr, steps=10):
        tcp, R_origin = self._get_tcp_from_joints(joints)
        server = MockQuestServer()
        ctrl = MockRobotCtrl(joints, tcp)
        pose_source = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )
        pose_source.rot_scale = 1.0
        pose_source.rotation_mode = "decoupled_3axis"

        # Frame 1: Bóp Grip (Anchor Engage)
        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z]
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })
        p0 = pose_source.get_pose()
        self.assertIsNone(p0, "Frame đầu tiên bóp Grip phải chốt mỏ neo và trả về None")

        # Các frame tiếp theo: Xoay cổ tay và chạy bộ lọc EMA đến ổn định
        q_vr = rot_matrix_vr.as_quat()  # [x, y, z, w]
        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [q_vr[3], q_vr[0], q_vr[1], q_vr[2]],  # [w, x, y, z]
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": True
        })

        res = None
        for _ in range(steps):
            res = pose_source.get_pose()

        self.assertIsNotNone(res)
        # Kiểm tra vị trí TCP không bị dịch chuyển
        pos_err = np.linalg.norm(np.array(res["pos"]) - np.array(tcp[:3]))
        self.assertLess(pos_err, 1e-4, "Vị trí robot không được dịch chuyển khi chỉ bóp xoay Grip!")

        # Tính góc xoay TCP trong hệ base_link
        q_target = res["quat"]  # [w, x, y, z]
        R_target = R.from_quat([q_target[1], q_target[2], q_target[3], q_target[0]])
        R_delta_base = R_target * R_origin.inv()
        rotvec_base = R_delta_base.as_rotvec()

        # Nhả Grip (Release)
        server.set_frame({
            "pos": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0],
            "trigger": False,
            "grip": False
        })
        p_rel = pose_source.get_pose()
        self.assertIsNone(p_rel)

        return rotvec_base

    def test_01_pose_invariant_roll_rotation(self):
        """Kiểm chứng tính bất biến tư thế của thao tác ROLL (+20° quanh trục nhìn tới)"""
        print("\n[TEST 1 - BẤT BIẾN TƯ THẾ TRỤC ROLL (+20°)]")
        # Roll: Vặn chìa khóa theo chiều kim đồng hồ quanh u_fwd = [0, 0, -1]
        R_roll_vr = R.from_rotvec([0.0, 0.0, -math.radians(20.0)])

        rotvec_home = self._run_rotation_test(self.SAFE_HOME_JOINTS, R_roll_vr)
        rotvec_ext1 = self._run_rotation_test(self.EXT1_JOINTS, R_roll_vr)
        rotvec_ext2 = self._run_rotation_test(self.EXT2_JOINTS, R_roll_vr)
        rotvec_ext3 = self._run_rotation_test(self.EXT3_JOINTS, R_roll_vr)

        print(f"  Home TCP Rotvec in Base: {np.round(np.degrees(rotvec_home), 2)}°")
        print(f"  Ext1 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext1), 2)}°")
        print(f"  Ext2 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext2), 2)}°")
        print(f"  Ext3 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext3), 2)}°")

        err1 = float(np.linalg.norm(rotvec_ext1 - rotvec_home))
        err2 = float(np.linalg.norm(rotvec_ext2 - rotvec_home))
        err3 = float(np.linalg.norm(rotvec_ext3 - rotvec_home))

        print(f"  Sai số góc: Ext1={err1:.6f} rad | Ext2={err2:.6f} rad | Ext3={err3:.6f} rad")
        self.assertLess(err1, 0.01, f"Ext1 Roll sai số vượt quá 0.01 rad! ({err1:.6f})")
        self.assertLess(err2, 0.01, f"Ext2 Roll sai số vượt quá 0.01 rad! ({err2:.6f})")
        self.assertLess(err3, 0.01, f"Ext3 Roll sai số vượt quá 0.01 rad! ({err3:.6f})")
        print("  -> PASS: Trục Roll bất biến 100% trên toàn bộ các tư thế cực hạn!")

    def test_02_pose_invariant_pitch_rotation(self):
        """Kiểm chứng tính bất biến tư thế của thao tác PITCH (+20° gập cổ tay)"""
        print("\n[TEST 2 - BẤT BIẾN TƯ THẾ TRỤC PITCH (+20°)]")
        # Pitch: Gập cổ tay quanh u_right = [1, 0, 0]
        R_pitch_vr = R.from_rotvec([math.radians(20.0), 0.0, 0.0])

        rotvec_home = self._run_rotation_test(self.SAFE_HOME_JOINTS, R_pitch_vr)
        rotvec_ext1 = self._run_rotation_test(self.EXT1_JOINTS, R_pitch_vr)
        rotvec_ext2 = self._run_rotation_test(self.EXT2_JOINTS, R_pitch_vr)
        rotvec_ext3 = self._run_rotation_test(self.EXT3_JOINTS, R_pitch_vr)

        print(f"  Home TCP Rotvec in Base: {np.round(np.degrees(rotvec_home), 2)}°")
        print(f"  Ext1 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext1), 2)}°")
        print(f"  Ext2 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext2), 2)}°")
        print(f"  Ext3 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext3), 2)}°")

        err1 = float(np.linalg.norm(rotvec_ext1 - rotvec_home))
        err2 = float(np.linalg.norm(rotvec_ext2 - rotvec_home))
        err3 = float(np.linalg.norm(rotvec_ext3 - rotvec_home))

        print(f"  Sai số góc: Ext1={err1:.6f} rad | Ext2={err2:.6f} rad | Ext3={err3:.6f} rad")
        self.assertLess(err1, 0.01, f"Ext1 Pitch sai số vượt quá 0.01 rad! ({err1:.6f})")
        self.assertLess(err2, 0.01, f"Ext2 Pitch sai số vượt quá 0.01 rad! ({err2:.6f})")
        self.assertLess(err3, 0.01, f"Ext3 Pitch sai số vượt quá 0.01 rad! ({err3:.6f})")
        print("  -> PASS: Trục Pitch bất biến 100% trên toàn bộ các tư thế cực hạn!")

    def test_03_pose_invariant_yaw_rotation(self):
        """Kiểm chứng tính bất biến tư thế của thao tác YAW (+20° quét trái/phải)"""
        print("\n[TEST 3 - BẤT BIẾN TƯ THẾ TRỤC YAW (+20°)]")
        # Yaw: Quét cổ tay quanh u_up = [0, 1, 0]
        R_yaw_vr = R.from_rotvec([0.0, math.radians(20.0), 0.0])

        rotvec_home = self._run_rotation_test(self.SAFE_HOME_JOINTS, R_yaw_vr)
        rotvec_ext1 = self._run_rotation_test(self.EXT1_JOINTS, R_yaw_vr)
        rotvec_ext2 = self._run_rotation_test(self.EXT2_JOINTS, R_yaw_vr)
        rotvec_ext3 = self._run_rotation_test(self.EXT3_JOINTS, R_yaw_vr)

        print(f"  Home TCP Rotvec in Base: {np.round(np.degrees(rotvec_home), 2)}°")
        print(f"  Ext1 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext1), 2)}°")
        print(f"  Ext2 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext2), 2)}°")
        print(f"  Ext3 TCP Rotvec in Base: {np.round(np.degrees(rotvec_ext3), 2)}°")

        err1 = float(np.linalg.norm(rotvec_ext1 - rotvec_home))
        err2 = float(np.linalg.norm(rotvec_ext2 - rotvec_home))
        err3 = float(np.linalg.norm(rotvec_ext3 - rotvec_home))

        print(f"  Sai số góc: Ext1={err1:.6f} rad | Ext2={err2:.6f} rad | Ext3={err3:.6f} rad")
        self.assertLess(err1, 0.01, f"Ext1 Yaw sai số vượt quá 0.01 rad! ({err1:.6f})")
        self.assertLess(err2, 0.01, f"Ext2 Yaw sai số vượt quá 0.01 rad! ({err2:.6f})")
        self.assertLess(err3, 0.01, f"Ext3 Yaw sai số vượt quá 0.01 rad! ({err3:.6f})")
        print("  -> PASS: Trục Yaw bất biến 100% trên toàn bộ các tư thế cực hạn!")

    def test_04_smooth_release_and_multi_clutch_continuity(self):
        """Kiểm chứng thả nút Grip giữ nguyên hướng và tái gắn mỏ neo (re-anchor) không bị giật"""
        print("\n[TEST 4 - THẢ NÚT GRIP GIỮ HƯỚNG & TÁI GẮN MỎ NEO KHÔNG GIẬT]")
        tcp, R_origin = self._get_tcp_from_joints(self.SAFE_HOME_JOINTS)
        server = MockQuestServer()
        ctrl = MockRobotCtrl(self.SAFE_HOME_JOINTS, tcp)
        pose_source = DeltaPoseSource(
            server, ctrl, scale=1.0,
            anchor_reference_func=lambda: list(ctrl.tcp())
        )

        # Chu kỳ 1: Bóp và xoay Pitch 15°
        server.set_frame({
            "pos": [0.1, 0.2, 0.3], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        pose_source.get_pose()

        q_rot1 = R.from_rotvec([math.radians(15.0), 0.0, 0.0]).as_quat()
        server.set_frame({
            "pos": [0.1, 0.2, 0.3], "quat": [q_rot1[3], q_rot1[0], q_rot1[1], q_rot1[2]],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        res1 = None
        for _ in range(8):
            res1 = pose_source.get_pose()

        self.assertIsNotNone(res1)
        q1_saved = res1["quat"]

        # Thả nút Grip
        server.set_frame({
            "pos": [0.1, 0.2, 0.3], "quat": [1.0, 0.0, 0.0, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": False
        })
        p_rel = pose_source.get_pose()
        self.assertIsNone(p_rel)

        # Cập nhật controller thực tế với hướng đã xoay res1
        R_res1 = R.from_quat([q1_saved[1], q1_saved[2], q1_saved[3], q1_saved[0]])
        rotvec_res1 = R_res1.as_rotvec()
        ctrl._tcp = [tcp[0], tcp[1], tcp[2], rotvec_res1[0], rotvec_res1[1], rotvec_res1[2]]

        # Chu kỳ 2: Bóp Grip lại ở tư thế mới (Re-engage)
        server.set_frame({
            "pos": [0.5, 0.5, 0.5], "quat": [0.7071, 0.0, 0.7071, 0.0],  # Vị trí tay cầm VR bất kỳ
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        p_reanchor = pose_source.get_pose()
        self.assertIsNone(p_reanchor, "Frame đầu tái gắn mỏ neo phải trả về None")

        # Frame tiếp theo giữ nguyên tay cầm (delta=0)
        server.set_frame({
            "pos": [0.5, 0.5, 0.5], "quat": [0.7071, 0.0, 0.7071, 0.0],
            "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
        })
        res2 = pose_source.get_pose()
        self.assertIsNotNone(res2)
        q2 = res2["quat"]

        # Bước nhảy hướng phải bằng 0.0000°
        R_res2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
        R_step = R_res2 * R_res1.inv()
        jump_deg = float(np.degrees(np.linalg.norm(R_step.as_rotvec())))
        print(f"  Bước nhảy góc khi Clutch Re-engage: {jump_deg:.5f}°")
        self.assertLess(jump_deg, 0.001, f"Bị giật góc khi Clutch re-engage! ({jump_deg:.5f}°)")
        print("  -> PASS: Thả và bóp lại Grip hoàn toàn êm ái, 0 trôi, 0 giật!")


if __name__ == "__main__":
    unittest.main()
