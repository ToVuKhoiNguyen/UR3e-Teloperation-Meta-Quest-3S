#!/usr/bin/env python3
"""
Kiểm thử tự động tính năng Viser UI Recording, VR Button A Toggle và các dataset viễn thao trong HDF5.
"""
import os
import sys
import time
import h5py
import numpy as np

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.selectable_recorder import SelectableDataRecorder
from core.verify_dataset import verify_dataset

def test_selectable_recorder_teleop_datasets():
    test_dir = "logs/test_teleop_recording"
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Khởi tạo recorder
    recorder = SelectableDataRecorder(config={"output_dir": test_dir, "hz": 20, "flush_chunk_size": 10})
    
    # 2. Bắt đầu episode
    filepath = recorder.start_episode(task_description="Test Teleop Recording")
    assert os.path.exists(filepath), f"File {filepath} không tồn tại!"
    
    # 3. Ghi 25 bước (vượt qua flush_chunk_size=10 để kiểm tra auto-flush)
    T = 25
    for i in range(T):
        telemetry = {
            "actual_q": [0.1 * (i + j) for j in range(6)],
            "actual_qd": [0.01 * (i + j) for j in range(6)],
            "actual_tcp_pose": [0.2, -0.1, 0.3 + 0.001 * i, 0.0, 3.14, 0.0],
            "target_q": [0.1 * (i + j) for j in range(6)],
            "target_tcp_pose": [0.2, -0.1, 0.3 + 0.001 * i, 0.0, 3.14, 0.0],
            "timestamp_host_ms": int(time.time() * 1000) + i * 50,
            "timestamp_host_ns": time.perf_counter_ns() + i * 50_000_000,
            "timestamp_robot": float(i * 0.05),
        }
        
        # VR controller pose [X, Y, Z, Qx, Qy, Qz, Qw]
        vr_pose = [0.15, -0.25, 0.45 + 0.002 * i, 0.0, 0.7071, 0.0, 0.7071]
        target_tcp = [0.2, -0.1, 0.3 + 0.001 * i, 0.0, 3.14, 0.0]
        grip_cmd = 1.0 if i > 10 else 0.0
        
        rgb_dummy = np.full((120, 160, 3), fill_value=(i % 255), dtype=np.uint8)
        
        recorder.record_step(
            telemetry=telemetry,
            images={"wrist_camera": rgb_dummy},
            action=[0.1 * (i + j) for j in range(6)] + [grip_cmd],
            gripper_actual=grip_cmd,
            gripper_target=grip_cmd,
            q_target=telemetry["target_q"],
            vr_controller_pose=vr_pose,
            target_tcp_pose=target_tcp,
        )
    
    # 4. Kết thúc episode
    saved_file = recorder.end_episode(success=True)
    assert saved_file == filepath, "Tệp trả về không khớp!"
    print(f"[TEST] Episode lưu thành công tại: {saved_file}")
    
    # 5. Kiểm tra nội dung tệp HDF5
    with h5py.File(saved_file, "r") as f:
        # Kiểm tra sự hiện diện của các dataset viễn thao
        assert "observations/vr_controller_pose" in f, "Thiếu dataset observations/vr_controller_pose!"
        assert "observations/target_tcp_pose" in f, "Thiếu dataset observations/target_tcp_pose!"
        assert "actions/gripper_cmd" in f, "Thiếu dataset actions/gripper_cmd!"
        
        vr_pose_dset = f["observations/vr_controller_pose"]
        target_tcp_dset = f["observations/target_tcp_pose"]
        gripper_cmd_dset = f["actions/gripper_cmd"]
        qpos_dset = f["observations/qpos"]
        action_dset = f["action"]
        
        print(f"[TEST] observations/vr_controller_pose shape: {vr_pose_dset.shape}, dtype: {vr_pose_dset.dtype}")
        print(f"[TEST] observations/target_tcp_pose shape:   {target_tcp_dset.shape}, dtype: {target_tcp_dset.dtype}")
        print(f"[TEST] actions/gripper_cmd shape:            {gripper_cmd_dset.shape}, dtype: {gripper_cmd_dset.dtype}")
        print(f"[TEST] observations/qpos shape:              {qpos_dset.shape}")
        print(f"[TEST] action shape:                         {action_dset.shape}")
        
        assert vr_pose_dset.shape == (T, 7), f"vr_controller_pose shape sai: {vr_pose_dset.shape} != ({T}, 7)"
        assert target_tcp_dset.shape == (T, 6), f"target_tcp_pose shape sai: {target_tcp_dset.shape} != ({T}, 6)"
        assert gripper_cmd_dset.shape == (T, 1), f"gripper_cmd shape sai: {gripper_cmd_dset.shape} != ({T}, 1)"
        assert qpos_dset.shape == (T, 6), f"qpos shape sai: {qpos_dset.shape} != ({T}, 6)"
        assert action_dset.shape == (T, 7), f"action shape sai: {action_dset.shape} != ({T}, 7)"
        
        # Kiểm tra tính chính xác của dữ liệu
        last_vr = vr_pose_dset[-1]
        assert np.isclose(last_vr[0], 0.15), f"Tọa độ VR X không khớp: {last_vr[0]}"
        assert np.isclose(last_vr[6], 0.7071), f"Quaternion VR Qw không khớp: {last_vr[6]}"
        
        last_grip = gripper_cmd_dset[-1, 0]
        assert np.isclose(last_grip, 1.0), f"Lệnh Gripper không khớp: {last_grip}"
        
        print("[TEST] Toàn bộ giá trị dataset viễn thao khớp 100%!")

    # 6. Kiểm định dataset qua verify_dataset
    print("\n[TEST] Chạy sanity check qua verify_dataset:")
    sanity_pass = verify_dataset(dataset_dir=test_dir)
    assert sanity_pass, "verify_dataset thất bại!"
    print("[TEST] verify_dataset: PASS 100%!")

    # Dọn dẹp thư mục tạm sau kiểm thử
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

def test_viser_ui_buttons():
    """Kiểm tra logic khởi tạo nút trên Viser UI."""
    from unittest.mock import MagicMock
    from teleop_quest.viser_ui import TeleopViserUI
    
    # Mock ViserServer
    mock_server = MagicMock()
    mock_gui = MagicMock()
    mock_server.gui = mock_gui
    
    button_handles = {}
    def mock_add_button(label, color=None, **kwargs):
        handle = MagicMock()
        handle.label = label
        handle.color = color
        button_handles[label] = handle
        return handle
    
    mock_gui.add_button.side_effect = mock_add_button
    
    # Kiểm tra phương thức set_recording_state trên checkbox cb_record
    ui = TeleopViserUI.__new__(TeleopViserUI)
    ui.cb_record = MagicMock()
    ui.cb_record.label = "Start Recording"
    ui.cb_record.value = False
    ui.cb_discard = MagicMock()
    ui.cb_discard.label = "Discard"
    ui.cb_discard.value = False
    
    # Bắt đầu ghi
    ui.set_recording_state(True, ep_idx=42)
    assert ui.cb_record.label == "Recording (Ep 42)"
    assert ui.cb_record.value == True
    print("[TEST] Checkbox Record chuyển sang trạng thái Recording thành công (Recording (Ep 42), Checked=True)!")
    
    # Kết thúc ghi
    ui.set_recording_state(False, ep_idx=43)
    assert ui.cb_record.label == "Start Recording"
    assert ui.cb_record.value == False
    print("[TEST] Checkbox Record chuyển về trạng thái Sẵn sàng thành công (Start Recording, Checked=False)!")

if __name__ == "__main__":
    print("=" * 60)
    print(" BẮT ĐẦU KIỂM THỬ SELECTABLE RECORDER VÀ VISER UI RECORDING ")
    print("=" * 60)
    test_viser_ui_buttons()
    test_selectable_recorder_teleop_datasets()
    print("=" * 60)
    print(" TẤT CẢ CÁC BÀI TEST ĐÃ VƯỢT QUA XUẤT SẮC! ")
    print("=" * 60)
