import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop_quest.quest_server import QuestServer
from teleop.teleoperation import TeleoperationController


class MockQuestServer:
    def __init__(self):
        self._latest_frame = None
        self._last_seen = 0.0
        self._connected = True
        self._new_session_requested = False
        self._broadcast_reset_called = False

    def get_frame(self):
        if self._latest_frame and self._latest_frame.get("tracking_lost"):
            safe = dict(self._latest_frame)
            safe["trigger"] = False
            safe["grip"] = False
            return safe
        return self._latest_frame

    def last_seen(self):
        return self._last_seen

    def is_connected(self):
        return self._connected

    def check_and_clear_new_session(self):
        if self._new_session_requested:
            self._new_session_requested = False
            return True
        return False

    def broadcast_reset_session(self):
        self._broadcast_reset_called = True


class MockRobotCtrl:
    def __init__(self, pos=(0.3, 0.0, 0.2), quat=(1.0, 0.0, 0.0, 0.0)):
        self._pos = list(pos)
        self._quat = list(quat)
        self._joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    def tcp(self):
        # [x, y, z, rx, ry, rz]
        return [self._pos[0], self._pos[1], self._pos[2], 0.0, 3.14159, 0.0]

    def joints(self):
        return list(self._joints)


def test_controller_occlusion_keeps_session_alive():
    """
    Test 1: Khi tay cầm bị khuất tầm nhìn (tracking_lost=True hoặc pos=None):
    - Phiên WebXR không bị hủy (broadcast_reset_session KHÔNG được gọi)
    - Robot đứng yên an toàn (trigger=False, grip=False, clutch_engaged=False)
    - Khi có lại tracking, điều khiển khôi phục trơn tru ngay tại vị trí hiện tại.
    """
    server = MockQuestServer()
    ctrl = MockRobotCtrl()
    
    # Anchor reference returns current robot TCP
    delta_source = DeltaPoseSource(
        server, ctrl, scale=1.0,
        anchor_reference_func=lambda: [0.35, -0.10, 0.25, 1.0, 0.0, 0.0, 0.0]
    )

    # 1. Tay cầm tracking bình thường, bóp cò tịnh tiến
    server._latest_frame = {
        "t": 100.0,
        "pos": [0.1, 0.2, 0.3],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "trigger": True,
        "grip": False
    }
    server._last_seen = time.time()
    
    pose1 = delta_source.get_pose()
    assert pose1 is not None, "Khi có tracking và bóp cò, pose phải khác None"
    assert delta_source.clutch_engaged is True, "Clutch phải được kích hoạt"

    # 2. Tay cầm bị khuất tầm nhìn kính Quest (tracking_lost: True)
    server._latest_frame = {
        "t": 100.5,
        "tracking_lost": True,
        "pos": None,
        "trigger": False,
        "grip": False
    }
    pose_lost = delta_source.get_pose()
    assert pose_lost is None, "Khi mất tracking, get_pose phải trả về None để giữ robot an toàn"
    assert delta_source.clutch_engaged is False, "Clutch phải tự động nhả khi mất tracking"
    assert delta_source.grip_engaged is False
    assert delta_source._last_button_state == 0, "Cờ trạng thái nút bấm phải được xả sạch về 0"
    assert server._broadcast_reset_called is False, "Tuyệt đối không gửi broadcast_reset_session làm sập WebXR"

    # 3. Tay cầm trở lại tầm nhìn tại tọa độ mới [0.15, 0.22, 0.35]
    # và người dùng tiếp tục bóp cò
    server._latest_frame = {
        "t": 101.0,
        "tracking_lost": False,
        "pos": [0.15, 0.22, 0.35],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "trigger": True,
        "grip": False
    }
    pose_regained = delta_source.get_pose()
    assert pose_regained is not None, "Khi có lại tracking và bóp cò, phải tiếp tục điều khiển bình thường"
    assert delta_source.clutch_engaged is True
    # Anchor mới phải khớp với vị trí mới của tay cầm
    np.testing.assert_allclose(delta_source._anchor_pos_quest, [0.15, 0.22, 0.35], atol=1e-5)


def test_enter_vr_auto_resync_without_teleop_toggle():
    """
    Test 2: Khi người dùng Enter VR lại (hoặc reconnect WebXR) mà nút Teleop đang Bật:
    - delta_pose_source tự động xả sạch các cờ kẹt cũ
    - Ngay khi người dùng bóp cò/Grip, robot lập tức nhận lệnh điều khiển
    - Hoàn toàn KHÔNG cần phải bấm Tắt rồi Bật lại nút Teleop.
    """
    server = MockQuestServer()
    ctrl = MockRobotCtrl()
    current_robot_tcp = [0.40, -0.15, 0.20, 1.0, 0.0, 0.0, 0.0]

    delta_source = DeltaPoseSource(
        server, ctrl, scale=1.0,
        anchor_reference_func=lambda: current_robot_tcp
    )

    # Giả lập phiên cũ trước đó: đang bóp cò ở tọa độ cũ của WebXR
    server._latest_frame = {
        "t": 50.0,
        "pos": [0.5, 0.5, 0.5],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "trigger": True,
        "grip": False
    }
    pose_old = delta_source.get_pose()
    assert pose_old is not None
    assert delta_source.clutch_engaged is True

    # Người dùng thoát VR và bấm Enter VR lại -> WebSocket gửi tín hiệu session_start
    server._new_session_requested = True

    # Phiên WebXR mới có hệ quy chiếu mới (gốc tọa độ tay cầm bắt đầu từ vị trí mới)
    server._latest_frame = {
        "t": 60.0,
        "pos": [-0.2, 0.1, 0.0],  # Tọa độ hoàn toàn khác trong WebXR refSpace mới
        "quat": [1.0, 0.0, 0.0, 0.0],
        "trigger": True,  # Người dùng bóp cò ngay khi vào VR
        "grip": False
    }

    # Gọi get_pose() trong vòng lặp điều khiển tiếp theo
    pose_new_session = delta_source.get_pose()
    assert pose_new_session is not None, "Phiên VR mới phải nhận pose ngay lập tức"
    assert delta_source.clutch_engaged is True

    # Gốc mỏ neo tay cầm phải tự động chốt vào tọa độ mới [-0.2, 0.1, 0.0]
    np.testing.assert_allclose(delta_source._anchor_pos_quest, [-0.2, 0.1, 0.0], atol=1e-5)
    # Gốc TCP neo phải tự động chốt vào vị trí robot hiện tại
    np.testing.assert_allclose(delta_source._anchor_tcp[:3], current_robot_tcp[:3], atol=1e-5)

    # Vì vừa mới bóp cò ở frame đầu tiên của session mới, độ dời tịnh tiến delta phải xấp xỉ 0 (không nhảy cóc)
    pos_cmd = pose_new_session["pos"] if isinstance(pose_new_session, dict) else pose_new_session[0]
    np.testing.assert_allclose(pos_cmd, current_robot_tcp[:3], atol=1e-3)


def test_grip_rotation_auto_resync_on_reconnect():
    """
    Test 3: Tương tự với thao tác xoay bằng nút Grip:
    - Khi reconnect / Enter VR mới, cờ grip_engaged và locked_axis_idx xả sạch
    - Ngay khi bóp Grip lần đầu trong phiên mới, sườn lên chốt lại R_ctrl_origin và R_tcp_origin
    - Không bị khóa cứng hoặc xoay giật góc.
    """
    server = MockQuestServer()
    ctrl = MockRobotCtrl()
    current_robot_tcp = [0.35, 0.0, 0.30, 0.0, 1.0, 0.0, 0.0]

    delta_source = DeltaPoseSource(
        server, ctrl, scale=1.0,
        anchor_reference_func=lambda: current_robot_tcp
    )

    # Reconnect báo phiên mới
    server._new_session_requested = True
    assert delta_source.grip_engaged is False
    assert delta_source.locked_axis_idx is None

    # Frame 1: Người dùng bắt đầu bóp Grip trong phiên mới -> chốt mỏ neo xoay
    server._latest_frame = {
        "t": 10.0,
        "pos": [0.0, 0.0, 0.0],
        "quat": [0.7071, 0.0, 0.7071, 0.0], # Pitch 90 deg
        "trigger": False,
        "grip": True
    }
    pose_rot_frame1 = delta_source.get_pose()
    # Ở frame đầu tiên khi bắt đầu bóp Grip, hệ thống chốt mỏ neo gốc và trả về None an toàn
    assert pose_rot_frame1 is None
    assert delta_source.grip_engaged is True
    assert delta_source.R_ctrl_origin is not None
    assert delta_source.R_tcp_origin is not None

    # Frame 2: Người dùng xoay cổ tay (15 độ pitch) -> hệ thống tính delta và xuất target_pose
    from scipy.spatial.transform import Rotation as R
    r_init = R.from_quat([0.0, 0.7071, 0.0, 0.7071])
    r_step = R.from_euler('x', 15.0, degrees=True) * r_init
    q_xyzw = r_step.as_quat()
    server._latest_frame = {
        "t": 10.05,
        "pos": [0.0, 0.0, 0.0],
        "quat": [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], # [w, x, y, z]
        "trigger": False,
        "grip": True
    }
    pose_rot_frame2 = delta_source.get_pose()
    assert pose_rot_frame2 is not None, "Frame tiếp theo khi xoay cổ tay phải xuất target_pose"


def test_teleoperation_controller_reset_session():
    """
    Test 4: Kiểm tra reset_teleop_session() trong TeleoperationController:
    - Giữ nguyên enabled = True
    - Xả cờ _was_clutch_released = True để target pose đầu tiên bắt dính mượt mà
    - Triệt tiêu hoàn toàn vận tốc tồn đọng _prev_linear_vel = 0.
    """
    mock_solver = MagicMock()
    mock_solver.fk.return_value = {"pos": [0.3, 0.0, 0.2], "quat": [1.0, 0.0, 0.0, 0.0]}
    mock_exec = MagicMock()
    mock_pose_src = MagicMock()

    controller = TeleoperationController(solver=mock_solver, executor=mock_exec, pose_source=mock_pose_src)
    controller.enabled = True
    controller._prev_linear_vel = np.array([0.05, 0.02, 0.01])
    controller._was_clutch_released = False

    # Gọi reset_teleop_session
    controller.reset_teleop_session(q_current=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

    assert controller.enabled is True, "TeleoperationController phải giữ nguyên trạng thái enabled"
    assert controller._was_clutch_released is True, "_was_clutch_released phải là True"
    np.testing.assert_allclose(controller._prev_linear_vel, [0.0, 0.0, 0.0], atol=1e-6)
