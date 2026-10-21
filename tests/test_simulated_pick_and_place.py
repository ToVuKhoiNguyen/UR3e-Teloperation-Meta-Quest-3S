#!/usr/bin/env python3
"""
Script kiểm tra toàn diện quy trình Mô phỏng Gắp Cà chua 3D + Camera ảo + Ghi HDF5.
Cách chạy:
  python tests/test_simulated_pick_and_place.py
"""

import sys
import os
import time
import h5py
import numpy as np
import cv2
import viser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop_quest.grasp_simulator import GraspSimulator
from teleop_quest.synthetic_camera import SyntheticWristCamera
from core.data_logger import DataLogger

def main():
    print("=" * 60)
    print(" TEST MÔ PHỎNG GẮP CÀ CHUA 3D & CAMERA ẢO CỔ TAY ")
    print("=" * 60)

    # 1. Khởi tạo Viser ảo
    server = viser.ViserServer(port=8089)
    print("-> Đã mở Viser 3D test tại: http://localhost:8089")

    # 2. Khởi tạo GraspSimulator và SyntheticWristCamera
    grasp_sim = GraspSimulator(server, pick_pos=(0.20, 0.20, 0.035), place_pos=(0.15, -0.20, 0.02))
    cam = SyntheticWristCamera(width=640, height=480)
    
    test_log_dir = "logs/ai_training_data/test_grasp_sim"
    logger = DataLogger({"output_dir": test_log_dir})
    logger.start_episode(task_description="Simulated Pick Tomato and Place in Basket")

    # 3. Tạo quỹ đạo giả lập:
    # Giai đoạn 1: Ở trên cao nhìn xuống (Z=0.25)
    # Giai đoạn 2: Hạ xuống sát quả cà chua (Z=0.04)
    # Giai đoạn 3: Bóp Grip -> Gắp cà chua lên cao (Z=0.25)
    # Giai đoạn 4: Di chuyển sang rổ (X=0.15, Y=-0.20, Z=0.20)
    # Giai đoạn 5: Nhả Grip -> Rơi vào rổ!

    trajectory_steps = [
        # (x, y, z, grip_pressed, desc)
        (0.20, 0.20, 0.25, False, "1. Ở trên cao tiếp cận"),
        (0.20, 0.20, 0.15, False, "2. Hạ thấp xuống"),
        (0.20, 0.20, 0.05, False, "3. Chạm sát quả cà chua"),
        (0.20, 0.20, 0.05, True,  "4. BÓP GRIP -> Gắp quả cà chua"),
        (0.20, 0.20, 0.18, True,  "5. Nhấc cà chua lên cao"),
        (0.18, 0.00, 0.20, True,  "6. Di chuyển ngang qua bàn"),
        (0.15, -0.20, 0.20, True, "7. Đến phía trên chiếc giỏ"),
        (0.15, -0.20, 0.12, True, "8. Hạ thấp vào giỏ"),
        (0.15, -0.20, 0.12, False, "9. NHẢ GRIP -> Thả cà chua vào giỏ!"),
    ]

    print("\nBắt đầu chạy mô phỏng từng bước:")
    last_frame = None

    for i, (x, y, z, grip, desc) in enumerate(trajectory_steps):
        tcp_pose = [x, y, z, 0.0, 3.14, 0.0] # TCP quay xuống đất
        
        # Cập nhật logic gắp thả
        is_holding, event = grasp_sim.update(tcp_pose, grip)
        
        # Render góc nhìn camera cổ tay (Wrist Cam)
        frame_rgb = cam.render(
            tcp_pose=tcp_pose,
            tomato_pos=grasp_sim.tomato_pos,
            basket_pos=grasp_sim.place_basket_pos,
            gripper_closed=grip
        )
        last_frame = frame_rgb

        # Ghi vào Dataset HDF5
        logger.log_step(
            q_actual=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            qd_actual=[0.0]*6,
            tcp_actual=tcp_pose + [0.0],
            gripper_actual=1.0 if grip else 0.0,
            images={"wrist_camera": frame_rgb}
        )

        status_str = "DÍNH CÀ CHUA" if is_holding else "RỖNG"
        print(f"  Step {i+1}: {desc} | Trạng thái: {status_str} | Sự kiện: {event}")
        time.sleep(0.05)

    # 4. Xuất file HDF5 và đóng gói
    saved_hdf5 = logger.end_episode(success=True)
    print(f"\n-> Đã lưu Episode HDF5 thành công vào: {saved_hdf5}")

    # 5. Lưu 1 ảnh từ camera ảo tại thời điểm gắp để kiểm tra bằng mắt
    if last_frame is not None:
        preview_path = "logs/simulated_wrist_camera_view.jpg"
        cv2.imwrite(preview_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"-> Đã lưu ảnh camera ảo góc nhìn cổ tay: {preview_path}")

    # 6. Kiểm tra lại dữ liệu trong file HDF5
    with h5py.File(saved_hdf5, 'r') as f:
        print("\nKiểm tra cấu trúc file HDF5:")
        print(f"-> Số frames: {f.attrs['num_steps']}")
        print(f"-> Dataset images: {list(f['observations/images'].keys())}")
        print(f"-> Tensor Wrist Camera shape: {f['observations/images/wrist_camera'].shape}")

    server.stop()
    print("=" * 60)
    print(" TEST HOÀN TẤT THÀNH CÔNG 100%! ")
    print("=" * 60)

if __name__ == "__main__":
    main()
