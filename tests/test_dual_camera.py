#!/usr/bin/env python3
"""
Test kịch bản xác thực kiến trúc Camera đa phương thức (USB & Ethernet):
1. Chế độ USB mặc định: Factory gọi đúng RealSenseWristCamera hoặc SyntheticWristCameraWrapper.
2. Chế độ Ethernet: Factory gọi đúng EthernetWristCamera.
3. Kiểm tra tính tương thích 100% với giao diện BaseWristCamera:
   - get_frame('rgb') trả về numpy ndarray (480, 640, 3).
   - get_frame('side_by_side') trả về numpy ndarray (480, 1280, 3).
   - add_osd_preview() vẽ OSD thành công.
   - release() giải phóng thread an toàn.
4. Kiểm tra argparse trong main_quest.py với --cam-source usb và --cam-source eth.
5. Đo độ trễ khởi tạo (< 50ms).
"""

import sys
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.camera_driver import (
    BaseWristCamera,
    SyntheticWristCameraWrapper,
    RealSenseWristCamera,
    EthernetWristCamera,
    create_wrist_camera,
)
from teleop_quest.main_quest import parse_args


def run_tests():
    print("=" * 65)
    print(" BẮT ĐẦU KIỂM THỬ MODULE CAMERA ĐA PHƯƠNG THỨC (USB & ETHERNET)")
    print("=" * 65)

    # 1. TEST KHỞI TẠO USB MẶC ĐỊNH (Cam kết tương thích ngược 100%)
    t0 = time.perf_counter()
    cam_usb_synth = create_wrist_camera(camera_type="synthetic", cam_source="usb")
    dt_usb = (time.perf_counter() - t0) * 1000.0
    print(f"\n[Test 1 - Khởi tạo USB Synthetic] Loại: {type(cam_usb_synth).__name__} (Thời gian: {dt_usb:.2f}ms)")
    assert isinstance(cam_usb_synth, SyntheticWristCameraWrapper), "LỖI: cam_source='usb' không trả về SyntheticWristCameraWrapper!"
    assert dt_usb < 50.0, f"LỖI: Thời gian khởi tạo ({dt_usb:.2f}ms) vượt quá 50ms!"
    print("  -> PASS: Khởi tạo USB gốc hoàn toàn chuẩn xác.")

    # 2. TEST KHỞI TẠO ETHERNET QUA FACTORY
    t0 = time.perf_counter()
    cam_eth = create_wrist_camera(
        camera_type="synthetic",  # Khi cam_source="eth", cờ cam_source sẽ ưu tiên kích hoạt Ethernet
        cam_source="eth",
        cam_url="tcp://192.168.1.100:5555",
        width=640,
        height=480
    )
    dt_eth = (time.perf_counter() - t0) * 1000.0
    print(f"\n[Test 2 - Khởi tạo Ethernet] Loại: {type(cam_eth).__name__} (Thời gian: {dt_eth:.2f}ms)")
    assert isinstance(cam_eth, EthernetWristCamera), "LỖI: cam_source='eth' không trả về EthernetWristCamera!"
    assert cam_eth.url == "tcp://192.168.1.100:5555"
    print("  -> PASS: Khởi tạo Ethernet qua Factory thành công.")

    # 3. TEST GIAO DIỆN HÀM BASEWRISTCAMERA CỦA ETHERNET CAMERA
    print(f"\n[Test 3 - Kiểm tra giao diện BaseWristCamera của EthernetWristCamera]")
    # 3a. Khung hình RGB khi chưa có stream (Placeholder)
    frame_rgb = cam_eth.get_frame("rgb")
    assert isinstance(frame_rgb, np.ndarray), "LỖI: get_frame('rgb') không trả về numpy array!"
    assert frame_rgb.shape == (480, 640, 3), f"LỖI: Sai kích thước RGB: {frame_rgb.shape}"
    print(f"  3a. Placeholder RGB: shape = {frame_rgb.shape}, dtype = {frame_rgb.dtype} -> OK")

    # 3b. Khung hình Side-by-side
    frame_sbs = cam_eth.get_frame("side_by_side")
    assert frame_sbs.shape == (480, 1280, 3), f"LỖI: Sai kích thước Side-by-side: {frame_sbs.shape}"
    print(f"  3b. Placeholder Side-by-side: shape = {frame_sbs.shape} -> OK")

    # 3c. Lớp phủ OSD
    osd_frame = cam_eth.add_osd_preview(frame_rgb, gripper_closed=True, stream_mode="rgb")
    assert isinstance(osd_frame, np.ndarray), "LỖI: add_osd_preview không trả về numpy array!"
    assert osd_frame.shape == (480, 640, 3)
    print("  3c. Lớp phủ OSD -> OK")

    # 3d. Thuộc tính is_real
    assert cam_eth.is_real() is True
    print("  3d. is_real() -> True (OK)")

    # 3e. Giải phóng an toàn
    cam_eth.release()
    print("  3e. release() an toàn -> OK")

    # 4. TEST ARGPARSE CỦA MAIN_QUEST.PY
    print(f"\n[Test 4 - Kiểm tra Argument Parsing trong main_quest.py]")
    saved_argv = sys.argv.copy()
    try:
        # Giả lập lệnh 1: Chạy USB mặc định
        sys.argv = ["main_quest.py", "--cam-source", "usb", "--camera", "synthetic"]
        args1 = parse_args()
        assert args1.cam_source == "usb"
        print(f"  4a. CLI: --cam-source usb -> args.cam_source = '{args1.cam_source}' (OK)")

        # Giả lập lệnh 2: Chạy Ethernet với URL tùy chỉnh
        sys.argv = ["main_quest.py", "--cam-source", "eth", "--cam-url", "tcp://192.168.1.50:6666"]
        args2 = parse_args()
        assert args2.cam_source == "eth"
        assert args2.cam_url == "tcp://192.168.1.50:6666"
        print(f"  4b. CLI: --cam-source eth --cam-url {args2.cam_url} -> OK")

    finally:
        sys.argv = saved_argv

    print("\n" + "=" * 65)
    print(" TẤT CẢ 4 KỊCH BẢN KIỂM CHỨNG ĐỀU PASS 100%!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
