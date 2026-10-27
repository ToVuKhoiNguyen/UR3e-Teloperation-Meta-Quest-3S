#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_visualize_data.py
Kiểm thử tự động công cụ trực quan hóa dữ liệu đa phương thức core/visualize_data.py:
1. Đọc đúng và đầy đủ cấu trúc HDF5 (Metadata, qpos, gripper, tcp_pose, wrist_camera).
2. Xử lý chính xác việc trích xuất ảnh frame RGB từ dataset camera.
3. Tạo và lưu tệp ảnh PNG tổng hợp 4 đồ thị (2x2) thành công mà không gây lỗi.
4. Khả năng chống chịu lỗi (Fault-tolerance) khi tệp thiếu trường dữ liệu ngàm hoặc camera.
5. Kiểm thử giao diện dòng lệnh CLI (argparse --file, --no-gui, --save).
"""

import os
import sys
import subprocess
import pytest
import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.visualize_data import HDF5Visualizer


def create_dummy_hdf5(file_path: str, num_steps: int = 25, has_camera: bool = True, has_gripper: bool = True):
    """Tạo tệp HDF5 giả lập đa phương thức chuẩn."""
    w, h = 64, 48
    with h5py.File(file_path, "w") as f:
        f.attrs["hz"] = 20
        f.attrs["task_description"] = "Test Tomato Pick & Place"
        f.attrs["duration_sec"] = float(num_steps / 20.0)
        f.attrs["num_steps"] = num_steps
        f.attrs["success"] = True

        obs = f.create_group("observations")
        # 6-DoF qpos
        t_vals = np.linspace(0, np.pi, num_steps, dtype=np.float32)
        qpos_data = np.stack([np.sin(t_vals + i * 0.5) for i in range(6)], axis=-1)
        obs.create_dataset("qpos", data=qpos_data)

        # 6-DoF tcp_pose (x, y, z xoay theo vòng tròn)
        tcp_x = 0.3 * np.cos(t_vals)
        tcp_y = 0.3 * np.sin(t_vals)
        tcp_z = 0.2 + 0.1 * t_vals
        tcp_data = np.zeros((num_steps, 6), dtype=np.float32)
        tcp_data[:, 0] = tcp_x
        tcp_data[:, 1] = tcp_y
        tcp_data[:, 2] = tcp_z
        obs.create_dataset("tcp_pose", data=tcp_data)

        # Timestamps
        obs.create_dataset("timestamps_ms", data=np.arange(num_steps, dtype=np.int64) * 50)

        # Gripper
        if has_gripper:
            acts = f.create_group("actions")
            grip_cmd = np.ones((num_steps, 1), dtype=np.float32)
            grip_cmd[: num_steps // 2] = 0.0
            acts.create_dataset("gripper_cmd", data=grip_cmd)
            obs.create_dataset("gripper", data=grip_cmd * 0.95)

        # Camera
        if has_camera:
            img_grp = obs.create_group("images")
            rgb_frames = (np.random.rand(num_steps, h, w, 3) * 255).astype(np.uint8)
            img_grp.create_dataset("wrist_camera", data=rgb_frames)


def test_hdf5_visualizer_complete(tmp_path):
    h5_path = str(tmp_path / "complete_ep.hdf5")
    png_path = str(tmp_path / "custom_output.png")
    create_dummy_hdf5(h5_path, num_steps=30, has_camera=True, has_gripper=True)

    viz = HDF5Visualizer(h5_path)
    data = viz.load_data()

    assert data["num_steps"] == 30
    assert data["hz"] == 20.0
    assert data["success"] is True
    assert data["qpos"].shape == (30, 6)
    assert len(data["gripper_cmd"]) == 30
    assert data["tcp_pose"].shape == (30, 6)
    assert data["has_wrist_camera"] is True

    # Kiểm tra trích xuất frame RGB
    frame, f_idx = viz.get_wrist_camera_frame(10)
    assert f_idx == 10
    assert frame is not None
    assert frame.shape == (48, 64, 3)

    # Kiểm tra vẽ và lưu ảnh PNG
    out_file = viz.plot(save_path=png_path, show_gui=False, frame_idx=10)
    assert out_file == png_path
    assert os.path.isfile(png_path)
    assert os.path.getsize(png_path) > 10000  # Ảnh PNG có dung lượng hợp lệ


def test_hdf5_visualizer_missing_fields(tmp_path):
    """Kiểm tra hoạt động bền bỉ khi thiếu camera hoặc ngàm kẹp."""
    h5_path = str(tmp_path / "minimal_ep.hdf5")
    create_dummy_hdf5(h5_path, num_steps=15, has_camera=False, has_gripper=False)

    viz = HDF5Visualizer(h5_path)
    data = viz.load_data()
    assert data["gripper_cmd"] is None
    assert data["has_wrist_camera"] is False

    frame, f_idx = viz.get_wrist_camera_frame()
    assert frame is None
    assert f_idx == -1

    # Vẫn phải vẽ và lưu ảnh thành công
    png_path = str(tmp_path / "minimal_viz.png")
    out_file = viz.plot(save_path=png_path, show_gui=False)
    assert os.path.isfile(out_file)
    assert os.path.getsize(out_file) > 5000


def test_hdf5_visualizer_cli(tmp_path):
    """Kiểm tra lệnh gọi CLI từ terminal."""
    h5_path = str(tmp_path / "cli_ep.hdf5")
    create_dummy_hdf5(h5_path, num_steps=20, has_camera=True, has_gripper=True)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "core", "visualize_data.py"),
        "--file", h5_path,
        "--no-gui"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "[Thành công]" in res.stdout

    # File PNG mặc định cùng thư mục tệp HDF5
    expected_png = str(tmp_path / "cli_ep_visualization.png")
    assert os.path.isfile(expected_png)
