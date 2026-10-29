#!/usr/bin/env python3
"""
Kiểm thử kết nối và truyền nhận luồng video qua giao thức WebSocket (RFC 6455):
1. Khởi động WebSocket Camera Server (OpenCV source hoặc synthetic frame) trên cổng 8767.
2. Khởi tạo EthernetWristCamera với URL ws://127.0.0.1:8767 qua Factory create_wrist_camera.
3. Kiểm tra kết nối thành công và nhận khung hình get_frame() dạng RGB numpy array.
4. Kiểm tra an toàn giải phóng tài nguyên release().
"""

import sys
import os
import time
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.camera_driver import create_wrist_camera, EthernetWristCamera


def run_tests():
    print("=" * 65)
    print(" BẮT ĐẦU KIỂM THỬ MODULE WEBSOCKET CAMERA STREAMING ")
    print("=" * 65)

    test_port = 8767
    server_cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "teleop_quest", "websocket_camera_server.py"),
        "--camera", "opencv",
        "--device", "0",
        "--port", str(test_port),
        "--fps", "30"
    ]

    print(f"[Test WS 1] Khởi động WebSocket Camera Server trên cổng {test_port}...")
    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.2)  # Đợi server bind port

    try:
        # Kiểm tra server có đang chạy không
        if server_proc.poll() is not None:
            err = server_proc.stderr.read().decode()
            raise RuntimeError(f"Server không khởi động được: {err}")

        print("[Test WS 2] Khởi tạo Client qua Factory create_wrist_camera(cam_source='ws')...")
        cam = create_wrist_camera(
            camera_type="synthetic",
            cam_source="ws",
            cam_url=f"ws://127.0.0.1:{test_port}",
            width=640,
            height=480
        )
        assert isinstance(cam, EthernetWristCamera), "Lỗi: cam_source='ws' phải trả về EthernetWristCamera"

        print("[Test WS 3] Chờ nhận frame đầu tiên qua WebSocket...")
        t_start = time.time()
        frame = None
        while time.time() - t_start < 4.0:
            frame = cam.get_frame("rgb")
            # Nếu không phải placeholder màu đen xám (hoặc đã kết nối)
            if cam._is_connected and frame is not None and not np.all(frame == (18, 22, 28)):
                break
            time.sleep(0.1)

        assert cam._is_connected, "Lỗi: Client không kết nối được tới WebSocket Server!"
        assert frame is not None, "Lỗi: Không nhận được frame từ WebSocket!"
        assert frame.shape == (480, 640, 3), f"Lỗi: Kích thước frame không đúng {frame.shape}"
        print(f"  -> PASS: Đã nhận frame WebSocket chuẩn kích thước: {frame.shape}, FPS đo được: {cam._fps_actual:.1f}")

        print("[Test WS 4] Kiểm tra giải phóng camera...")
        cam.release()
        print("  -> PASS: Giải phóng WebSocket Client an toàn.")

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[Test WS] Đã dừng WebSocket Server.")

    print("=" * 65)
    print(" TẤT CẢ TEST WEBSOCKET CAMERA ĐỀU THÀNH CÔNG (PASS 100%)! ")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
