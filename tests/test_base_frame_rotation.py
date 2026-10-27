#!/usr/bin/env python3
"""
Kiểm thử tự động tính năng Đảo chiều (Invert Direction) các trục xoay VR:
1. Gập cổ tay VR xuống (Flexion) -> Tay máy cụp xuống.
2. Ngửa cổ tay VR lên (Extension) -> Tay máy vươn/ngửa lên.
3. Vặn tay như vặn chìa khóa sang phải -> Đầu tay máy xoay nghiêng sang phải (Clockwise).
4. Quét cổ tay sang phải -> Đầu tay máy xoay quét sang phải quanh trục Base Z.
5. Kiểm tra tính bất biến theo Base Frame trên toàn bộ không gian làm việc (nhiều góc cánh tay và camera).
"""
import os
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from teleop_quest.delta_pose_source import DeltaPoseSource

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

def test_pitch_flexion_and_extension_direction():
    print("=" * 80)
    print(" [TEST 1] PITCH: GẬP CỔ TAY XUỐNG -> CỤP XUỐNG | NGỬA LÊN -> NGỬA LÊN ")
    print("=" * 80)
    
    server = MockQuestServer()
    test_arm_angles_deg = [0.0, 30.0, 45.0, 90.0, 135.0, -60.0, -120.0]
    test_camera_yaw_deg = [0.0, 25.0, 45.0, 90.0, -60.0, 180.0]

    for arm_deg in test_arm_angles_deg:
        arm_rad = math.radians(arm_deg)
        r_dist = 0.35
        tcp_x = r_dist * math.cos(arm_rad)
        tcp_y = r_dist * math.sin(arm_rad)
        tcp_z = 0.20
        
        u_reach = np.array([math.cos(arm_rad), math.sin(arm_rad), 0.0]) # Hướng vươn cánh tay

        for cam_yaw in test_camera_yaw_deg:
            cam_rad = math.radians(cam_yaw)
            R_tool_anchor = R.from_euler("zyx", [cam_rad, 0.0, np.pi], degrees=False)
            rotvec_anchor = R_tool_anchor.as_rotvec()
            
            mock_tcp = [tcp_x, tcp_y, tcp_z, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
            mock_joints = [arm_rad, -1.57, 1.57, -1.57, -1.57, cam_rad]
            ctrl = MockRobotCtrl(joints=mock_joints, tcp=mock_tcp)
            
            pose_source = DeltaPoseSource(server, ctrl, scale=1.0)
            pose_source.nav_mode = "clock_face"
            pose_source.rotation_mode = "decoupled_3axis"
            pose_source.enable_axis_snapping = False

            # Khởi tạo Anchor
            server.set_frame({
                "pos": [0.1, 0.2, 0.3], "quat": [1.0, 0.0, 0.0, 0.0],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            pose_source.get_pose() # Reset Anchor

            # 1. GẬP CỔ TAY XUỐNG (+0.14 rad quanh trục u_right)
            # Theo phản xạ tự nhiên: Đầu ngàm phải cụp chúc xuống theo phương vươn u_reach
            R_flex = R.from_rotvec([0.14, 0.0, 0.0])
            q_flex = R_flex.as_quat()
            server.set_frame({
                "pos": [0.1, 0.2, 0.3], "quat": [q_flex[3], q_flex[0], q_flex[1], q_flex[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            res_flex = pose_source.get_pose()
            q_res = res_flex["quat"]
            R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])
            
            v_anchor = R_tool_anchor.apply([0.0, 0.0, 1.0]) # [0, 0, -1]
            v_new = R_target.apply([0.0, 0.0, 1.0])
            delta_v_flex = v_new - v_anchor
            delta_xy_flex = np.array([delta_v_flex[0], delta_v_flex[1], 0.0])
            dir_flex = delta_xy_flex / np.linalg.norm(delta_xy_flex)
            
            # Gập xuống phải làm vector tip chúc theo hướng +u_reach (cụp về phía trước)
            cos_flex = float(np.dot(dir_flex, u_reach))
            assert cos_flex >= 0.999, f"Lỗi: Gập cổ tay không làm cụp ngàm! CosSim={cos_flex}"

            # 2. NGỬA CỔ TAY LÊN (-0.14 rad quanh trục u_right)
            # Theo phản xạ tự nhiên: Đầu ngàm phải ngửa lên về phía sau (-u_reach)
            R_ext = R.from_rotvec([-0.14, 0.0, 0.0])
            q_ext = R_ext.as_quat()
            server.set_frame({
                "pos": [0.1, 0.2, 0.3], "quat": [q_ext[3], q_ext[0], q_ext[1], q_ext[2]],
                "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
            })
            res_ext = pose_source.get_pose()
            q_res_ext = res_ext["quat"]
            R_target_ext = R.from_quat([q_res_ext[1], q_res_ext[2], q_res_ext[3], q_res_ext[0]])
            
            v_new_ext = R_target_ext.apply([0.0, 0.0, 1.0])
            delta_v_ext = v_new_ext - v_anchor
            delta_xy_ext = np.array([delta_v_ext[0], delta_v_ext[1], 0.0])
            dir_ext = delta_xy_ext / np.linalg.norm(delta_xy_ext)
            
            cos_ext = float(np.dot(dir_ext, -u_reach))
            assert cos_ext >= 0.999, f"Lỗi: Ngửa cổ tay không làm ngửa ngàm! CosSim={cos_ext}"

    print("[PASS] Động tác Pitch: Gập cổ tay -> Cụp xuống; Ngửa cổ tay -> Ngửa lên (100% chuẩn xác trên 42 góc test)!")

def test_roll_clockwise_direction():
    print("\n" + "=" * 80)
    print(" [TEST 2] ROLL: VẶN CHÌA KHÓA SANG PHẢI -> ĐẦU TAY MÁY NGHIÊNG SANG PHẢI ")
    print("=" * 80)

    server = MockQuestServer()
    arm_rad = math.radians(45.0)
    u_reach = np.array([math.cos(arm_rad), math.sin(arm_rad), 0.0])

    R_tool_anchor = R.from_euler("zyx", [0.0, 0.0, np.pi], degrees=False)
    rotvec_anchor = R_tool_anchor.as_rotvec()
    
    mock_tcp = [0.25, 0.25, 0.20, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
    mock_joints = [arm_rad, -1.57, 1.57, -1.57, -1.57, 0.0]
    ctrl = MockRobotCtrl(joints=mock_joints, tcp=mock_tcp)

    pose_source = DeltaPoseSource(server, ctrl, scale=1.0)
    pose_source.nav_mode = "clock_face"
    pose_source.rotation_mode = "decoupled_3axis"
    pose_source.enable_axis_snapping = False

    server.set_frame({
        "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
        "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
    })
    pose_source.get_pose() # Anchor

    # Vặn chìa khóa sang phải (Clockwise quanh trục nhìn u_fwd = [0, 0, -1])
    # Theo quy tắc bàn tay phải nhìn từ người dùng tới bàn: quay quanh -Z
    R_roll_right = R.from_rotvec([0.0, 0.0, -0.15])
    q_roll = R_roll_right.as_quat()
    server.set_frame({
        "pos": [0.0, 0.0, 0.0], "quat": [q_roll[3], q_roll[0], q_roll[1], q_roll[2]],
        "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
    })
    pose_res = pose_source.get_pose()
    q_res = pose_res["quat"]
    R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])

    # Trục xoay thực tế trong Base Frame: R_delta = R_target * R_anchor.inv()
    R_delta_base = R_target * R_tool_anchor.inv()
    rotvec_base = R_delta_base.as_rotvec()
    rot_mag = np.linalg.norm(rotvec_base)
    axis_base = rotvec_base / rot_mag
    
    # Kiểm tra góc xoay mang giá trị dương cùng chiều kim đồng hồ quanh u_reach
    dot_axis = float(np.dot(axis_base, u_reach))
    print(f"[TEST Roll] Trục xoay Base: {axis_base} | Trục chuẩn cánh tay u_reach: {u_reach} | Dot: {dot_axis:+.5f}")
    assert dot_axis >= 0.999, f"Lỗi: Vặn phải không làm xoay nghiêng sang phải quanh cánh tay! Dot={dot_axis}"
    print("[PASS] Động tác Roll: Vặn chìa khóa sang phải -> Đầu tay máy xoay nghiêng sang phải chuẩn 1:1!")

def test_yaw_sweep_direction():
    print("\n" + "=" * 80)
    print(" [TEST 3] YAW: QUÉT CỔ TAY SANG PHẢI -> ĐẦU TAY MÁY QUÉT SANG PHẢI ")
    print("=" * 80)

    server = MockQuestServer()
    R_tool_anchor = R.from_euler("zyx", [0.0, 0.0, np.pi], degrees=False)
    rotvec_anchor = R_tool_anchor.as_rotvec()
    mock_tcp = [0.3, 0.0, 0.2, rotvec_anchor[0], rotvec_anchor[1], rotvec_anchor[2]]
    mock_joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    ctrl = MockRobotCtrl(joints=mock_joints, tcp=mock_tcp)

    pose_source = DeltaPoseSource(server, ctrl, scale=1.0)
    pose_source.nav_mode = "clock_face"
    pose_source.rotation_mode = "decoupled_3axis"
    pose_source.enable_axis_snapping = False

    server.set_frame({
        "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
        "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
    })
    pose_source.get_pose() # Anchor

    # Quét cổ tay sang phải quanh trục thẳng đứng u_up = [0, 1, 0] (Clockwise từ trên nhìn xuống: vector âm)
    R_yaw_right = R.from_rotvec([0.0, -0.15, 0.0])
    q_yaw = R_yaw_right.as_quat()
    server.set_frame({
        "pos": [0.0, 0.0, 0.0], "quat": [q_yaw[3], q_yaw[0], q_yaw[1], q_yaw[2]],
        "head_quat": [1.0, 0.0, 0.0, 0.0], "trigger": False, "grip": True
    })
    pose_res = pose_source.get_pose()
    q_res = pose_res["quat"]
    R_target = R.from_quat([q_res[1], q_res[2], q_res[3], q_res[0]])

    R_delta_base = R_target * R_tool_anchor.inv()
    rotvec_base = R_delta_base.as_rotvec()
    axis_base = rotvec_base / np.linalg.norm(rotvec_base)
    
    # Quét sang phải tương ứng quay quanh Base -Z (chiều kim đồng hồ từ trên nhìn xuống)
    u_yaw_right = np.array([0.0, 0.0, -1.0])
    dot_yaw = float(np.dot(axis_base, u_yaw_right))
    print(f"[TEST Yaw] Trục xoay Base: {axis_base} | Trục chuẩn Yaw Right [0, 0, -1]: {u_yaw_right} | Dot: {dot_yaw:+.5f}")
    assert dot_yaw >= 0.999, f"Lỗi: Quét cổ tay sang phải không làm xoay ngàm sang phải! Dot={dot_yaw}"
    print("[PASS] Động tác Yaw: Quét cổ tay sang phải -> Đầu tay máy xoay sang phải chuẩn 1:1!")

if __name__ == "__main__":
    test_pitch_flexion_and_extension_direction()
    test_roll_clockwise_direction()
    test_yaw_sweep_direction()
    print("\n" + "=" * 80)
    print(" TOÀN BỘ 3 KỊCH BẢN KIỂM CHỨNG ĐẢO CHIỀU 1:1: 100% PASS! ")
    print("=" * 80)
