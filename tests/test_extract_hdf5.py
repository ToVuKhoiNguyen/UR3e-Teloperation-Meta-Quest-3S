import os
import sys
import shutil
import pytest
import numpy as np
import h5py
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.extract_hdf5 import HDF5Extractor, SEMANTIC_HEADERS


def test_hdf5_extractor_comprehensive(tmp_path):
    h5_path = str(tmp_path / "test_episode.hdf5")
    out_dir = str(tmp_path / "test_episode_extracted")

    # 1. Tạo tệp HDF5 giả lập đầy đủ luồng: RGB, Depth, 1D/2D Kinematics/Teleop
    num_steps = 15
    w, h = 64, 48
    with h5py.File(h5_path, "w") as f:
        f.attrs["hz"] = 30
        f.attrs["task_description"] = "Test HDF5 Extraction"
        f.attrs["duration_sec"] = 0.5
        f.attrs["num_steps"] = num_steps

        # Observations
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.ones((num_steps, 6), dtype=np.float32) * 0.1)
        obs.create_dataset("tcp_pose", data=np.ones((num_steps, 6), dtype=np.float32) * 0.2)
        obs.create_dataset("vr_controller_pose", data=np.ones((num_steps, 7), dtype=np.float32) * 0.3)
        obs.create_dataset("gripper", data=np.ones((num_steps, 1), dtype=np.float32) * 0.5)
        obs.create_dataset("timestamps_ms", data=np.arange(num_steps, dtype=np.int64) * 33)

        # Actions
        f.create_dataset("action", data=np.ones((num_steps, 7), dtype=np.float32))
        act_grp = f.create_group("actions")
        act_grp.create_dataset("gripper_cmd", data=np.ones((num_steps, 1), dtype=np.float32))

        # Images
        img_grp = obs.create_group("images")
        # RGB [T, H, W, 3]
        rgb_data = (np.random.rand(num_steps, h, w, 3) * 255).astype(np.uint8)
        img_grp.create_dataset("wrist_camera", data=rgb_data)
        # Depth [T, H, W] uint16
        depth_data = (np.random.rand(num_steps, h, w) * 1000 + 300).astype(np.uint16)
        img_grp.create_dataset("wrist_camera_depth", data=depth_data)

    # 2. Khởi chạy trích xuất
    extractor = HDF5Extractor(h5_path, output_dir=out_dir, verbose=True)
    res_dir = extractor.extract_all()
    assert res_dir == out_dir
    assert os.path.exists(out_dir)

    # 3. Kiểm tra Metadata
    meta_file = os.path.join(out_dir, "episode_metadata.json")
    assert os.path.exists(meta_file)
    summary_file = os.path.join(out_dir, "summary.txt")
    assert os.path.exists(summary_file)

    # 4. Kiểm tra Video RGB
    mp4_path = os.path.join(out_dir, "wrist_camera.mp4")
    assert os.path.exists(mp4_path)
    cap = cv2.VideoCapture(mp4_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == num_steps
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h
    cap.release()

    # 5. Kiểm tra Thư mục Depth và từng ảnh PNG 16-bit
    depth_folder = os.path.join(out_dir, "wrist_camera_depth")
    assert os.path.isdir(depth_folder)
    depth_info = os.path.join(depth_folder, "depth_info.json")
    assert os.path.exists(depth_info)
    assert len([f for f in os.listdir(depth_folder) if f.endswith(".png")]) == num_steps

    frame0_path = os.path.join(depth_folder, "frame_00000.png")
    depth_img0 = cv2.imread(frame0_path, cv2.IMREAD_UNCHANGED)
    assert depth_img0 is not None
    assert depth_img0.dtype == np.uint16
    assert depth_img0.shape == (h, w)

    # 6. Kiểm tra các tệp CSV
    csv_files = ["qpos.csv", "tcp_pose.csv", "vr_controller_pose.csv", "action.csv", "gripper_cmd.csv"]
    for cf in csv_files:
        c_path = os.path.join(out_dir, cf)
        assert os.path.exists(c_path), f"Thiếu file {cf}"
        with open(c_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            # Dòng 0 là header + num_steps dòng dữ liệu
            assert len(lines) == num_steps + 1
            assert lines[0].startswith("step,")
