"""
test_viser_gizmo_mouse_and_vr_seamless_coexistence.py
======================================================
Kiểm chứng triệt để:
1. Khi bật Teleop và dùng chuột kéo Viser 3D Gizmo, tay máy lập tức bám mượt theo Gizmo với trạng thái OK.
2. Chuyển đổi mượt mà giữa Chuột (Viser Gizmo) và Tay cầm VR (Quest Multi-Clutch):
   - Kéo Gizmo bằng chuột -> robot bám theo.
   - Bóp cò/Grip trên Quest -> robot và Gizmo bám theo VR.
   - Nhả cò/Grip -> chốt vị trí trên Gizmo.
   - Kéo lại Gizmo bằng chuột -> robot tiếp tục bám theo chuột liền mạch, không giật, không kẹt.
3. Không gọi servoStop() trong vòng lặp Teleop khi nhả phím hay khi Gizmo đứng yên.
4. Đảm bảo toàn bộ thuộc tính phục vụ cơ chế xoay Grip (grip_engaged, locked_axis_idx, R_ctrl_origin, R_tcp_origin, prev_target_quat, reset) được khởi tạo đầy đủ và an toàn.
"""

import unittest
import numpy as np
import time
from teleop.teleoperation import TeleoperationController, SafetyGate
from teleop.pyroki_solver import PyrokiIKSolver
from teleop_quest.pose_sources import WidgetPoseSource
from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop_quest.quest_server import QuestServer


class MockRobotController:
    def __init__(self, solver, q_init):
        self.solver = solver
        self._q = list(q_init)

    def joints(self):
        return list(self._q)

    def tcp(self):
        fk = self.solver.fk(self._q)
        return list(fk["pos"]) + [0.0, 0.0, 0.0]


class MockExecutor:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.servo_stopped = False
        self.commands_sent = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def send(self, q):
        self.ctrl._q = list(q)
        self.commands_sent.append(list(q))

    def stop(self, a=10.0):
        self.servo_stopped = True


class MockWidget:
    def __init__(self, pos, quat):
        self.position = tuple(pos)
        self.wxyz = tuple(quat)
        self.is_suppressed = False

    def update_pose(self, pos=None, wxyz=None, suppress=False):
        if pos is not None:
            self.position = tuple(pos)
        if wxyz is not None:
            self.wxyz = tuple(wxyz)


class MockServer:
    def __init__(self):
        self._frame = None
        self._connected = True

    def set_frame(self, frame):
        self._frame = frame

    def get_frame(self):
        return self._frame

    def get_controller_frame(self):
        return self._frame

    def is_connected(self):
        return self._connected

    def is_cockpit_jogging(self):
        return False


class TestViserGizmoAndVRSeamlessCoexistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = PyrokiIKSolver()
        cls.gate = SafetyGate(guard=None, solver=cls.solver)

    def setUp(self):
        self.start_joints = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.ctrl = MockRobotController(self.solver, self.start_joints)
        self.executor = MockExecutor(self.ctrl)
        self.server = MockServer()

        fk_init = self.solver.fk(self.start_joints)
        self.init_pos = np.array(fk_init["pos"])
        self.init_quat = np.array(fk_init["quat"])

        self.widget = MockWidget(self.init_pos, self.init_quat)
        self.widget_pose_source = WidgetPoseSource(self.widget)
        self.widget_pose_source.sync_with_robot(pos=self.init_pos, quat=self.init_quat)

        self.teleop = TeleoperationController(
            pose_source=self.widget_pose_source,
            pyroki_solver=self.solver,
            safety_gate=self.gate,
            executor=self.executor
        )
        self.teleop.sync_with_robot(self.start_joints)
        self.teleop.enabled = True

        self.pose_source = DeltaPoseSource(
            server=self.server,
            ctrl=self.ctrl,
            scale=1.0,
            anchor_reference_func=lambda: list(self.ctrl.tcp()[:3]) + list(self.init_quat),
            alpha=0.50,
            max_delta_m=0.30
        )
        self.pose_source.nav_mode = "cartesian"

    def test_01_delta_pose_source_initialization_safety(self):
        """Kiểm tra toàn bộ biến trạng thái xoay Grip được khởi tạo an toàn và phương thức reset() tồn tại."""
        self.assertFalse(self.pose_source.grip_engaged)
        self.assertIsNone(self.pose_source.locked_axis_idx)
        self.assertIsNone(self.pose_source.R_ctrl_origin)
        self.assertIsNone(self.pose_source.R_tcp_origin)
        self.assertIsNone(self.pose_source.prev_target_quat)
        self.assertTrue(hasattr(self.pose_source, "reset"))

        # Kiểm tra khi chưa có frame kết nối (frame=None)
        self.server.set_frame(None)
        pose_none = self.pose_source.get_pose()
        self.assertIsNone(pose_none, "Khi chưa kết nối VR, get_pose() phải trả về None")

        # Gọi reset() không được văng lỗi
        self.pose_source.reset()
        self.assertFalse(self.pose_source.grip_engaged)

    def test_02_mouse_gizmo_drag_tracking(self):
        """Khi Teleop bật và kéo Viser Gizmo bằng chuột, robot lập tức di chuyển bám theo Gizmo với trạng thái OK."""
        # VR controller ở trạng thái nghỉ
        self.server.set_frame({"pos": [0.0, 0.0, 0.0], "trigger": False, "grip": False})
        vr_pose = self.pose_source.get_pose()
        self.assertIsNone(vr_pose, "Khi không bấm nút VR, DeltaPoseSource phải nhường quyền bằng None")

        # Người dùng kéo Gizmo sang phải +25mm (trục Y)
        new_target_pos = self.init_pos + np.array([0.0, 0.025, 0.0])
        self.widget.position = tuple(new_target_pos)

        # Lấy target pose từ WidgetPoseSource
        gizmo_target = self.widget_pose_source.get_pose()
        self.assertIsNotNone(gizmo_target)
        np.testing.assert_allclose(gizmo_target["pos"], new_target_pos, atol=1e-5)

        # Điều khiển 10 chu kỳ 20Hz
        for _ in range(10):
            self.teleop.step(self.ctrl.joints(), pose=gizmo_target)
            self.assertEqual(self.teleop.get_status(), "OK")

        fk_after_gizmo = self.solver.fk(self.ctrl.joints())
        error_p = np.linalg.norm(np.array(fk_after_gizmo["pos"]) - new_target_pos)
        self.assertLess(error_p, 0.005, f"Robot phải bám sát Gizmo trong vòng 5mm (actual error: {error_p*1000:.2f}mm)")
        self.assertFalse(self.executor.servo_stopped, "Không được gọi servoStop() khi kéo Gizmo!")

    def test_03_seamless_switching_between_mouse_and_vr(self):
        """
        Chu trình chuyển đổi liên tục Chuột (Gizmo) <-> Tay cầm VR:
        1. Kéo Gizmo bằng chuột -> robot bám theo.
        2. Bóp cò VR -> robot và Gizmo bám theo VR.
        3. Nhả cò VR -> chốt vị trí Gizmo tại robot TCP.
        4. Kéo Gizmo bằng chuột lần nữa -> robot tiếp tục bám theo chuột liền mạch không giật.
        """
        # BƯỚC 1: Kéo Gizmo bằng chuột (+20mm theo X)
        pos_step1 = self.init_pos + np.array([0.020, 0.0, 0.0])
        self.widget.position = tuple(pos_step1)
        for _ in range(10):
            target = self.widget_pose_source.get_pose()
            self.teleop.step(self.ctrl.joints(), pose=target)
        fk_step1 = self.solver.fk(self.ctrl.joints())
        self.assertAlmostEqual(fk_step1["pos"][0], pos_step1[0], delta=0.003)

        # BƯỚC 2: Người dùng bóp cò Quest VR và kéo tay (-30mm Z Quest -> +30mm X robot)
        hand_start = np.array([0.0, 0.0, -0.20])
        self.server.set_frame({"pos": hand_start.tolist(), "trigger": True, "grip": False})
        vr_pose_start = self.pose_source.get_pose()
        self.assertIsNotNone(vr_pose_start, "Frame đầu bóp cò VR phải có pose neo hợp lệ")

        # Kéo tay thêm 3cm
        hand_moved = hand_start + np.array([0.0, 0.0, -0.030])
        self.server.set_frame({"pos": hand_moved.tolist(), "trigger": True, "grip": False})
        for _ in range(10):
            vr_pose = self.pose_source.get_pose()
            self.assertIsNotNone(vr_pose)
            self.teleop.step(self.ctrl.joints(), pose=vr_pose)
            # Giả lập đồng bộ Gizmo đi theo VR (do UIWorker thực hiện)
            self.widget.update_pose(pos=vr_pose["pos"], wxyz=vr_pose["quat"])

        fk_step2 = self.solver.fk(self.ctrl.joints())
        self.assertGreater(fk_step2["pos"][0], fk_step1["pos"][0] + 0.015, "Robot phải tiến theo VR")

        # BƯỚC 3: Người dùng nhả cò VR
        self.server.set_frame({"pos": hand_moved.tolist(), "trigger": False, "grip": False})
        self.pose_source.reset_anchor()
        self.teleop.notify_clutch_released(self.ctrl.joints())
        self.widget_pose_source.sync_with_robot(pos=fk_step2["pos"], quat=fk_step2["quat"])

        # Xác nhận DeltaPoseSource trả về None sau khi nhả
        self.assertIsNone(self.pose_source.get_pose())

        # BƯỚC 4: Người dùng tiếp tục dùng chuột kéo Gizmo (+20mm theo Y)
        pos_step4 = np.array(fk_step2["pos"]) + np.array([0.0, 0.020, 0.0])
        self.widget.position = tuple(pos_step4)

        target_gizmo_step4 = self.widget_pose_source.get_pose()
        self.assertIsNotNone(target_gizmo_step4)
        for _ in range(10):
            self.teleop.step(self.ctrl.joints(), pose=target_gizmo_step4)
            self.assertEqual(self.teleop.get_status(), "OK")

        fk_final = self.solver.fk(self.ctrl.joints())
        self.assertAlmostEqual(fk_final["pos"][1], pos_step4[1], delta=0.003,
                               msg="Robot phải bám mượt theo thao tác kéo chuột sau khi nhả VR!")


if __name__ == "__main__":
    unittest.main()
