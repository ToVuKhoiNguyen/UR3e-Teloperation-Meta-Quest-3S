#!/usr/bin/env python3
"""
Script kiểm tra Wrist Camera mô phỏng (Synthetic Wrist Camera) và kiểm tra di chuyển quả Cà chua.

Cách chạy:
  1. Chế độ kiểm tra nhanh (Render & lưu 4 góc độ ảnh vào logs/):
     python tests/test_camera.py

  2. Chế độ giao diện Web trực quan (kéo thanh trượt di chuyển cà chua trực tiếp):
     python tests/test_camera.py --gui
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Them thu muc goc vao PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop_quest.synthetic_camera import SyntheticWristCamera

def test_camera_render():
    print("=" * 60)
    print("Kiểm tra Wrist Camera (Mô phỏng 3D Eye-in-Hand) ")
    print("=" * 60)

    # 1. Khởi tạo Camera ảo (RealSense D435i - 640x480, FOV 65 độ)
    cam = SyntheticWristCamera(width=640, height=480, fov_deg=65.0)

    # Giả định đầu gắp robot đang ở vị trí Home an toàn chuẩn UR3e:
    # [x=0.136, y=0.327, z=0.120] (hướng nhìn thẳng xuống mặt bàn)
    tool_pos = [0.1357, 0.3270, 0.1200]
    tool_quat = [0.01244, 0.99984, 0.00039, -0.01254] # [w, x, y, z]
    basket_pos = np.array([0.35, 0.15, 0.02], dtype=np.float32)

    os.makedirs("logs", exist_ok=True)

    test_cases = [
        ("Chính giữa (Dưới đầu gắp)", np.array([0.1357, 0.3270, 0.035]), "logs/test_cam_center.jpg"),
        ("Lệch sang TRÁI (-10cm)",   np.array([0.0357, 0.3270, 0.035]), "logs/test_cam_left.jpg"),
        ("Lệch sang PHẢI (+10cm)",  np.array([0.2357, 0.3270, 0.035]), "logs/test_cam_right.jpg"),
        ("Đẩy ra XA (+10cm)",       np.array([0.1357, 0.4270, 0.035]), "logs/test_cam_far.jpg"),
    ]

    print(f"\n[Robot] Vị trí đầu gắp TCP: X={tool_pos[0]:.3f}, Y={tool_pos[1]:.3f}, Z={tool_pos[2]:.3f}")
    print("[Mặt bàn] Z = 0.000m, Cà chua Z = 0.035m\n")

    for desc, tomato_pos, save_path in test_cases:
        img_rgb = cam.render(
            tool_pos=tool_pos,
            tool_quat=tool_quat,
            tomato_pos=tomato_pos,
            basket_pos=basket_pos,
            gripper_closed=False
        )
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        print(f" -> {desc:30s} | Tọa độ 3D: [{tomato_pos[0]:.3f}, {tomato_pos[1]:.3f}, {tomato_pos[2]:.3f}] -> Lưu tại: {save_path}")

    print("Đã render thành công cả 4 trường hợp! Bạn có thể xem các file ảnh trong thư mục 'logs/'.")
    print("=" * 60)

def test_interactive_gui():
    """Khởi động một mini server Viser trên cổng 8089 để kéo slider thử nghiệm trực tiếp."""
    import viser
    import subprocess
    print("=" * 60)
    print("KHỞI ĐỘNG GIAO DIỆN TEST CAMERA TƯƠNG TÁC TRÊN VISER ")
    print("=" * 60)

    subprocess.run("fuser -k 8089/tcp 2>/dev/null", shell=True)
    server = viser.ViserServer(host="0.0.0.0", port=8089)
    cam = SyntheticWristCamera(width=640, height=480)

    tool_pos = np.array([0.1357, 0.3270, 0.1200], dtype=np.float32)
    tool_quat = [0.01244, 0.99984, 0.00039, -0.01254]
    basket_pos = np.array([0.35, 0.15, 0.02], dtype=np.float32)

    # Tạo mesh 3D trong không gian Viser
    tomato_mesh = server.scene.add_icosphere(
        "/world/tomato",
        radius=0.032,
        color=(235, 45, 45),
        position=(tool_pos[0], tool_pos[1], 0.035)
    )
    server.scene.add_box(
        "/world/basket",
        dimensions=(0.15, 0.15, 0.03),
        color=(60, 150, 230),
        position=tuple(basket_pos)
    )

    with server.gui.add_folder("Điều chỉnh vị trí Cà chua"):
        slider_x = server.gui.add_slider("Tọa độ X (Trái / Phải)", min=-0.40, max=0.40, step=0.01, initial_value=float(tool_pos[0]))
        slider_y = server.gui.add_slider("Tọa độ Y (Gần / Xa)", min=0.10, max=0.60, step=0.01, initial_value=float(tool_pos[1]))
        btn_center = server.gui.add_button("Đặt ngay dưới đầu gắp")

    with server.gui.add_folder("Wrist Camera Live Preview"):
        init_frame = cam.render(tool_pos, tool_quat, np.array([tool_pos[0], tool_pos[1], 0.035]), basket_pos)
        cam_preview = server.gui.add_image(cv2.resize(init_frame, (320, 240)), label="Live Camera Preview")

    def update_view():
        tx = slider_x.value
        ty = slider_y.value
        tomato_p = np.array([tx, ty, 0.035], dtype=np.float32)
        tomato_mesh.position = (tx, ty, 0.035)
        frame = cam.render(tool_pos, tool_quat, tomato_p, basket_pos)
        cam_preview.image = cv2.resize(frame, (320, 240))

    @slider_x.on_update
    def _(_): update_view()

    @slider_y.on_update
    def _(_): update_view()

    @btn_center.on_click
    def _(_):
        slider_x.value = float(tool_pos[0])
        slider_y.value = float(tool_pos[1])
        update_view()

    print("Mở trình duyệt truy cập: http://localhost:8089")
    print("Kéo các thanh trượt 'Tọa độ X' và 'Tọa độ Y' để xem quả cà chua di chuyển trên bàn 3D và trực tiếp trên camera!")
    print("Nhấn Ctrl+C để kết thúc.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nĐã tắt server test.")

def main():
    parser = argparse.ArgumentParser(description="Test Wrist Camera & Tomato Movement")
    parser.add_argument("--gui", action="store_true", help="Chạy giao diện Viser test tương tác (Port 8089)")
    args = parser.parse_args()

    if args.gui:
        test_interactive_gui()
    else:
        test_camera_render()

if __name__ == "__main__":
    main()
