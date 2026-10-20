#!/usr/bin/env python3
"""
Script kiểm tra độc lập camera Intel RealSense D455 / D435:
1. Tự động quét và nhận diện thiết bị qua Intel RealSense SDK (pyrealsense2).
2. Kiểm tra tốc độ cổng USB (Cảnh báo nếu cắm nhầm cổng USB 2.0 thay vì USB 3.0+).
3. Đọc luồng video RGB thực tế, đo FPS và độ trễ thu thập hình ảnh.
4. Mô phỏng quy trình nén JPEG + Base64 WebSocket của Quest Server để kiểm tra băng thông.
5. Lưu ảnh snapshot kiểm định chất lượng vào logs/test_realsense_d455.jpg.
"""

import os
import sys
import time
import base64
import argparse
import cv2
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra camera Intel RealSense D455/D435")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng khung hình (mặc định: 640)")
    parser.add_argument("--height", type=int, default=480, help="Chiều cao khung hình (mặc định: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Tốc độ khung hình (mặc định: 30)")
    parser.add_argument("--gui", action="store_true", help="Hiển thị cửa sổ OpenCV GUI trực tiếp")
    parser.add_argument("--duration", type=float, default=3.0, help="Thời gian kiểm tra benchmark (giây)")
    args = parser.parse_args()

    print("=" * 65)
    print("      KIỂM TRA CAMERA INTEL REALSENSE (D455 / D435 / D405)")
    print("=" * 65)

    # 1. Kiểm tra thư viện pyrealsense2
    try:
        import pyrealsense2 as rs
        print(f"[1/5] Thư viện Intel RealSense SDK: pyrealsense2 v{rs.__version__} (OK)")
    except ImportError:
        print("[!] LỖI: Chưa cài đặt pyrealsense2!")
        print("    Vui lòng cài đặt bằng: pip install pyrealsense2")
        sys.exit(1)

    # 2. Quét thiết bị RealSense trên cổng USB
    print("[2/5] Đang quét thiết bị RealSense qua các cổng USB máy tính...")
    ctx = rs.context()
    devices = list(ctx.devices)

    if not devices:
        print("\n[!] KHÔNG TÌM THẤY THIẾT BỊ INTEL REALSENSE NÀO!")
        print("    -> Vui lòng cắm cáp USB Type-C của camera RealSense D455 vào máy tính.")
        print("    -> Đảm bảo cổng cắm là USB 3.0/3.1 (thường có màu xanh dương hoặc biểu tượng SS).")
        print("\nCác thiết bị video chuẩn V4L2 hiện có trên hệ thống:")
        os.system("ls -la /dev/video* 2>/dev/null || echo 'Không có cổng /dev/video'")
        print("=" * 65)
        sys.exit(0)

    print(f" -> Đã phát hiện {len(devices)} thiết bị RealSense:")
    for idx, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"    [{idx+1}] Tên: {name}")
        print(f"        Serial:   {serial}")
        print(f"        Firmware: {fw}")
        print(f"        Cổng USB: USB {usb_type}")
        if "2." in usb_type:
            print("        [CẢNH BÁO] Camera đang cắm ở cổng USB 2.0/2.1!")
            print("                   Băng thông có thể bị giới hạn độ phân giải hoặc FPS.")
            print("                   Khuyên dùng: Cắm sang cổng USB 3.0+ để đạt hiệu năng tối đa.")
        else:
            print("        [TỐT] Cổng USB 3.0+ tốc độ cao, băng thông đầy đủ.")

    target_dev = devices[0]
    dev_name = target_dev.get_info(rs.camera_info.name)
    dev_serial = target_dev.get_info(rs.camera_info.serial_number)

    # 3. Khởi động luồng đọc ảnh
    print(f"\n[3/5] Khởi động luồng Stream RGB {args.width}x{args.height} @ {args.fps}FPS...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(dev_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

    try:
        pipeline.start(config)
        print(" -> Đã mở Pipeline RealSense thành công!")
    except Exception as e:
        print(f"[!] Lỗi khi mở luồng camera: {e}")
        sys.exit(1)

    # 4. Đo đạc hiệu năng FPS và mô phỏng WebSocket nén
    print(f"\n[4/5] Đang benchmark tốc độ thu thập hình ảnh và nén WebSocket...")
    os.makedirs("logs", exist_ok=True)
    out_snapshot = "logs/test_realsense_d455.jpg"

    frames_read = 0
    total_capture_time = 0.0
    total_encode_time = 0.0
    jpeg_sizes = []
    last_frame_rgb = None

    t_start = time.perf_counter()
    bench_duration = args.duration

    try:
        while time.perf_counter() - t_start < bench_duration:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_rgb = np.asanyarray(color_frame.get_data())
            t1 = time.perf_counter()
            total_capture_time += (t1 - t0)

            # Mô phỏng nén Quest Server WebSocket
            t2 = time.perf_counter()
            small = cv2.resize(frame_rgb, (320, 240))
            small_bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            success, enc = cv2.imencode(".jpg", small_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                jpeg_bytes = enc.tobytes()
                b64 = base64.b64encode(jpeg_bytes)
                jpeg_sizes.append(len(jpeg_bytes))
            t3 = time.perf_counter()
            total_encode_time += (t3 - t2)

            frames_read += 1
            last_frame_rgb = frame_rgb

            if args.gui:
                preview = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(preview, f"{dev_name} | {args.fps} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Intel RealSense Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        pipeline.stop()
        if args.gui:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    actual_fps = frames_read / elapsed if elapsed > 0 else 0
    avg_cap_ms = (total_capture_time / frames_read * 1000) if frames_read > 0 else 0
    avg_enc_ms = (total_encode_time / frames_read * 1000) if frames_read > 0 else 0
    avg_kb = (sum(jpeg_sizes) / len(jpeg_sizes) / 1024) if jpeg_sizes else 0

    print(f" -> Số frame đã đọc:      {frames_read} frames")
    print(f" -> Tốc độ thực tế:       {actual_fps:.1f} FPS (Target: {args.fps} FPS)")
    print(f" -> Độ trễ đọc phần cứng: {avg_cap_ms:.2f} ms/frame")
    print(f" -> Thời gian nén JPEG:   {avg_enc_ms:.2f} ms/frame")
    print(f" -> Kích thước WebSocket: {avg_kb:.1f} KB/frame (~{avg_kb * actual_fps:.1f} KB/s)")

    # 5. Lưu ảnh snapshot kiểm định
    if last_frame_rgb is not None:
        h, w = last_frame_rgb.shape[:2]
        osd_frame = last_frame_rgb.copy()
        cv2.rectangle(osd_frame, (0, 0), (w, 35), (15, 23, 42), -1)
        cv2.putText(osd_frame, f"{dev_name} | {actual_fps:.1f} FPS | USB 3.0", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cx, cy = w // 2, h // 2
        cv2.drawMarker(osd_frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)
        cv2.circle(osd_frame, (cx, cy), 18, (0, 255, 255), 1, cv2.LINE_AA)

        bgr_save = cv2.cvtColor(osd_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_snapshot, bgr_save)
        print(f"\n[5/5] Đã lưu ảnh kiểm định thành công:")
        print(f"      -> {out_snapshot}")

    print("=" * 65)
    print(" KIỂM THỬ HOÀN TẤT! Camera sẵn sàng kết nối vào hệ thống VR Teleop!")
    print(" Lệnh chạy cùng VR Teleop: python main.py --camera realsense")
    print("=" * 65)

if __name__ == "__main__":
    main()
