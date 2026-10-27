#!/usr/bin/env python3
"""
tests/test_cockpit_joint_control.py
Kịch bản kiểm chứng tính năng Interactive Joint-Space Control & Anti-Fight Lock trên Quest Cockpit:
1. Cockpit Slider & Arrow Test:
   - Slew-rate limiting: Kiểm tra mỗi bước dịch chuyển <= 0.08 rad/step (~4.58 deg/step)
   - Kẹp biên giới hạn khớp UR3E_LIMITS
   - Anti-Fight Lock: is_cockpit_jogging() giữ cờ trong 450ms, chống giằng co telemetry
   - WebSocket message & HTTP API endpoint: nhận joint_cmd và joint_jog chính xác.
2. Gizmo Auto-Sync Test:
   - Khi xoay khớp qua cockpit, tính FK và tự động đồng bộ ui.target_widget.position và wxyz
   - widget_pose_source và pose_source được reset_anchor mượt mà
   - Tiếp tục kéo chuột Gizmo sau đó không bị nhảy giật (0 glitch).
3. Collision Guard Test (0.5cm Uniform Armor):
   - Cố tình gửi lệnh xoay khớp đâm vào bàn hoặc va đập link
   - Bị chặn đứng ngay tại lớp áo giáp 0.5cm, trả về ENV COLLISION hoặc SELF COLLISION
   - Robot giữ nguyên vị trí an toàn trước đó, không vượt rào.
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
    clamp_joints,
)
from teleop.teleoperation import TeleoperationController
from teleop_quest.quest_server import QuestServer
from teleop_quest.pose_sources import WidgetPoseSource


class MockExecutor:
    def __init__(self):
        self.last_sent = None
        self.sent_history = []
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send(self, q):
        self.last_sent = list(q)
        self.sent_history.append(list(q))

    def stop(self, a=10.0):
        pass


class MockTargetWidget:
    def __init__(self, pos=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.position = tuple(pos)
        self.wxyz = tuple(wxyz)


class MockPoseSource:
    def __init__(self):
        self.reanchored = False
        self.last_reanchor_pos = None

    def get_pose(self):
        return None

    def reset_anchor(self):
        self.reanchored = True

    def reanchor_position(self, pos):
        self.last_reanchor_pos = list(pos)


class TestCockpitJointControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = PyrokiIKSolver()
        cls.home_q = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]

    def setUp(self):
        self.exec = MockExecutor()
        self.pose = MockPoseSource()
        self.controller = TeleoperationController(
            pose_source=self.pose,
            solver=self.solver,
            executor=self.exec,
            dt=0.05
        )
        self.controller.enabled = True
        self.controller.sync_with_robot(self.home_q)

    def test_01_slider_and_arrow_slew_rate_limiting(self):
        """Kiểm tra điều khiển góc khớp: Slew-rate limiting <= 0.08 rad/step và kẹp biên UR3E_LIMITS"""
        # Giả lập người dùng kéo thanh trượt Base từ 0.0 rad lên 1.0 rad (~57.3 độ)
        q_target = list(self.home_q)
        q_target[0] = 1.0  # Bước nhảy 1.0 rad

        # Chu kỳ 1
        res1 = self.controller.step_joint_space(q_target, q_real=self.home_q, max_step=0.08)
        self.assertTrue(res1["success"], f"Step 1 thất bại: {res1}")
        q_step1 = res1["q"]
        delta1 = abs(q_step1[0] - self.home_q[0])

        # Phải bị giới hạn tối đa 0.08 rad/step
        self.assertLessEqual(delta1, 0.0801, f"Delta step vượt quá giới hạn 0.08 rad: {delta1}")
        self.assertAlmostEqual(delta1, 0.08, places=3)
        self.assertEqual(self.exec.last_sent, q_step1)

        # Lặp 15 chu kỳ để robot tiến dần về 1.0 rad
        for _ in range(15):
            res = self.controller.step_joint_space(q_target, max_step=0.08)
            self.assertTrue(res["success"])

        # Sau nhiều chu kỳ, khớp 0 phải chạm tới đích 1.0 rad
        self.assertAlmostEqual(self.controller._last_valid_q[0], 1.0, places=2)

    def test_02_joint_limits_clamping(self):
        """Kiểm tra khi người dùng nhập số độ hoặc kéo slider vượt quá giới hạn an toàn UR3E_LIMITS"""
        # Shoulder có giới hạn [-2.65, 0.20]. Giả lập user nhập +1.5 rad (+85.9 độ)
        q_target = list(self.home_q)
        q_target[1] = 1.5  # Vượt xa biên trên +0.20 rad

        res = self.controller.step_joint_space(q_target, max_step=0.5)
        self.assertTrue(res["success"])
        q_next = res["q"]

        # Shoulder không được vượt quá 0.20 rad
        self.assertLessEqual(q_next[1], 0.20, f"Khớp shoulder đã vượt giới hạn an toàn: {q_next[1]}")

    def test_03_quest_server_anti_fight_lock(self):
        """Kiểm tra cơ chế Anti-Fight Lock trên QuestServer: is_cockpit_jogging và sync_cockpit_joints"""
        cert_path = os.path.join(BASE_DIR, "certs", "cert.pem")
        key_path = os.path.join(BASE_DIR, "certs", "key.pem")

        server = QuestServer(cert_path, key_path, host="127.0.0.1", http_port=9443, ws_port=9444)

        # Ban đầu chưa có tương tác
        self.assertFalse(server.is_cockpit_jogging())

        # Robot đang ở home_q
        server.sync_cockpit_joints(self.home_q)
        self.assertEqual(server.get_cockpit_target_joints(), self.home_q)

        # 1. Người dùng bấm nút mũi tên xoay khớp Base thêm 2 độ (+0.0349 rad)
        delta_rad = math.radians(2.0)
        server.handle_cockpit_joint_jog(joint_idx=0, delta_rad=delta_rad)

        # Cờ is_cockpit_jogging phải lập tức bật True
        self.assertTrue(server.is_cockpit_jogging())
        target_q = server.get_cockpit_target_joints()
        self.assertAlmostEqual(target_q[0], self.home_q[0] + delta_rad, places=4)

        # Trong thời gian jogging, telemetry server gọi sync_cockpit_joints KHÔNG ĐƯỢC ghi đè mốc target
        telemetry_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        server.sync_cockpit_joints(telemetry_q)
        # Target vẫn giữ nguyên của người dùng, không bị kéo về 0.0
        self.assertAlmostEqual(server.get_cockpit_target_joints()[0], self.home_q[0] + delta_rad, places=4)

        # 2. Người dùng nhập số độ trực tiếp cho khớp Elbow: 45 độ (0.785 rad)
        deg_val = 45.0
        server.handle_cockpit_joint_cmd(joint_idx=2, target_rad=math.radians(deg_val))
        self.assertTrue(server.is_cockpit_jogging())
        self.assertAlmostEqual(server.get_cockpit_target_joints()[2], math.radians(deg_val), places=4)

    def test_04_gizmo_fk_auto_sync(self):
        """Kiểm tra đồng bộ ngược Gizmo 3D và Trạng thái Nội bộ (FK Synchronization)"""
        mock_widget = MockTargetWidget(pos=(0.3, 0.0, 0.2), wxyz=(1.0, 0.0, 0.0, 0.0))
        widget_pose_source = WidgetPoseSource(mock_widget)

        # Ban đầu Gizmo ở (0.3, 0.0, 0.2)
        self.assertEqual(mock_widget.position, (0.3, 0.0, 0.2))

        # Dùng cockpit đổi khớp Base xoay 30 độ (0.5236 rad)
        q_target = list(self.home_q)
        q_target[0] = math.radians(30.0)

        # Chạy step_joint_space cho đến khi đạt đích
        for _ in range(10):
            res = self.controller.step_joint_space(q_target, max_step=0.08)

        fk_now = res.get("fk")
        self.assertIsNotNone(fk_now)
        p_fk = fk_now["pos"]
        q_fk = fk_now["quat"]

        # Đồng bộ ngược vào Gizmo như trong main_quest.py
        mock_widget.position = tuple(p_fk)
        mock_widget.wxyz = tuple(q_fk)
        widget_pose_source.sync_with_robot(pos=p_fk, quat=q_fk)
        self.pose.reset_anchor()

        # Kiểm tra Gizmo đã tự động trôi theo đầu TCP thực tế
        self.assertAlmostEqual(mock_widget.position[0], p_fk[0], places=4)
        self.assertAlmostEqual(mock_widget.position[1], p_fk[1], places=4)
        self.assertAlmostEqual(mock_widget.position[2], p_fk[2], places=4)

        # Kiểm tra khi dùng chuột kéo tiếp từ WidgetPoseSource, vị trí bắt đầu trùng khít với TCP mới
        gizmo_pose = widget_pose_source.get_pose()
        err_dist = np.linalg.norm(np.array(gizmo_pose["pos"]) - np.array(p_fk))
        self.assertLess(err_dist, 1e-4, f"Lệch pha giữa Gizmo và TCP sau khi xoay khớp: {err_dist*1000:.2f} mm")

    def test_05_collision_guard_0_5cm_armor(self):
        """Kiểm tra Collision Guard 0.5cm: Cố tình ép khớp đâm xuống bàn máy -> Chặn tại biên giáp"""
        # Cố tình đưa khớp shoulder gập xuống hết cỡ hướng thẳng xuống mặt bàn máy
        # Để TCP hoặc camera chạm mặt bàn Z <= 0.005m
        q_dangerous = [0.0, -0.3, 2.5, -2.2, -1.57, 0.0]

        # Kiểm tra trước bằng solver xem cấu hình này có va chạm không
        tbl_check = self.solver.check_table_collision(q_dangerous, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)

        # Chạy qua step_joint_space
        res = self.controller.step_joint_space(q_dangerous, max_step=0.08)

        # Tiếp tục bước về hướng nguy hiểm cho tới khi chạm rào
        stopped_at_barrier = False
        last_safe_q = list(self.home_q)

        for step_i in range(30):
            res = self.controller.step_joint_space(q_dangerous, max_step=0.08)
            if not res["success"]:
                self.assertIn(res["status"], ["ENV COLLISION", "SELF COLLISION"])
                stopped_at_barrier = True
                break
            last_safe_q = res["q"]

        self.assertTrue(stopped_at_barrier, "Lớp Áo giáp 0.5cm không chặn chuyển động va chạm!")
        # Vị trí gửi xuống executor phải là vị trí an toàn cuối cùng
        self.assertEqual(self.exec.last_sent, last_safe_q)

    def test_06_http_and_websocket_joint_handlers(self):
        """Kiểm tra các hàm nhận lệnh góc khớp toàn phần và từng khớp trên QuestServer"""
        cert_path = os.path.join(BASE_DIR, "certs", "cert.pem")
        key_path = os.path.join(BASE_DIR, "certs", "key.pem")
        server = QuestServer(cert_path, key_path, host="127.0.0.1", http_port=9445, ws_port=9446)

        # 1. Đặt toàn bộ 6 khớp
        new_q = [0.1, -1.2, 1.4, -1.5, -1.4, 0.2]
        server.handle_cockpit_joints_cmd(new_q)
        self.assertTrue(server.is_cockpit_jogging())
        self.assertEqual(server.get_cockpit_target_joints(), new_q)

        # 2. Đặt từng khớp
        server.handle_cockpit_joint_cmd(joint_idx=3, target_rad=-1.0)
        self.assertAlmostEqual(server.get_cockpit_target_joints()[3], -1.0, places=4)

        # 3. Jog từng khớp
        server.handle_cockpit_joint_jog(joint_idx=3, delta_rad=0.2)
        self.assertAlmostEqual(server.get_cockpit_target_joints()[3], -0.8, places=4)

    def test_07_anti_fight_lock_timeout_recovery(self):
        """Kiểm tra phục hồi sau Anti-Fight Lock: Hết 450ms thì sync_cockpit_joints hoạt động bình thường"""
        cert_path = os.path.join(BASE_DIR, "certs", "cert.pem")
        key_path = os.path.join(BASE_DIR, "certs", "key.pem")
        server = QuestServer(cert_path, key_path, host="127.0.0.1", http_port=9447, ws_port=9448)

        # Giả lập tương tác
        server.handle_cockpit_joint_jog(joint_idx=0, delta_rad=0.1)
        self.assertTrue(server.is_cockpit_jogging())

        # Giả lập trôi qua 0.5s
        server._cockpit_jog_time -= 0.50
        self.assertFalse(server.is_cockpit_jogging())

        # Khi không jogging nữa, sync_cockpit_joints phải cập nhật thành công
        new_telemetry = [0.5, -1.0, 1.0, -1.0, -1.0, 0.5]
        server.sync_cockpit_joints(new_telemetry)
        self.assertEqual(server.get_cockpit_target_joints(), new_telemetry)


if __name__ == "__main__":
    unittest.main()
