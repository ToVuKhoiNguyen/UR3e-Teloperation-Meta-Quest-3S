#!/usr/bin/env python3
"""
Camera Publisher Node (Dành cho máy tính nhúng gắn trên Robot):
- Thu thập luồng hình ảnh từ Camera Intel RealSense (pyrealsense2) hoặc USB Webcam (OpenCV).
- Nén JPEG tối ưu băng thông mạng LAN Gigabit/Fast Ethernet.
- Phát stream thời gian thực qua giao thức ZeroMQ PUB (mặc định cổng 5555).
- Tương thích 100% với EthernetWristCamera trên máy tính điều khiển Teleoperation.

Hướng dẫn sử dụng trên máy tính nhúng:
  python3 camera_server_node.py --camera realsense --port 5555 --fps 30
  python3 camera_server_node.py --camera opencv --device 0 --port 5555
"""

import sys
import time
import argparse
import signal
import cv2
import numpy as np

try:
    import zmq
except ImportError:
    print("[CameraServer] LỖI: Chưa cài đặt thư viện 'pyzmq'.")
    print("Vui lòng cài đặt bằng lệnh: pip install pyzmq")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Ethernet Streaming Server (Robot Onboard Node)")
    parser.add_argument("--port", type=int, default=5555, help="Cổng mạng ZeroMQ PUB (mặc định: 5555)")
    parser.add_argument("--camera", type=str, choices=["realsense", "opencv"], default="realsense", help="Driver camera: realsense hoặc opencv")
    parser.add_argument("--device", type=int, default=0, help="Device ID khi dùng OpenCV (mặc định: 0)")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng khung hình (mặc định: 640)")
    parser.add_argument("--height", type=int, default=480, help="Chiều cao khung hình (mặc định: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Tốc độ khung hình mục tiêu (mặc định: 30)")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="Chất lượng nén JPEG 1-100 (mặc định: 85)")
    parser.add_argument("--enable-depth", action="store_true", help="Bật phát kèm luồng độ sâu Depth (chỉ RealSense)")
    return parser.parse_args()


class RealSenseSource:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, enable_depth: bool = False):
        import pyrealsense2 as rs
        self.rs = rs
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        print(f"[RealSenseSource] Đang khởi động camera RealSense {width}x{height}@{fps}FPS...")
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color) if enable_depth else None
        print("[RealSenseSource] Camera RealSense đã sẵn sàng phát stream.")

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None

        frame_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_raw = np.asanyarray(depth_frame.get_data())

        return frame_bgr, depth_raw

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVSource:
    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        print(f"[OpenCVSource] Đang mở Camera /dev/video{device}...")
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera OpenCV device ID {device}")
        print(f"[OpenCVSource] Camera OpenCV ID {device} đã sẵn sàng.")

    def get_frames(self):
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return None, None
        return frame_bgr, None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def main():
    args = parse_args()

    # Thiết lập ZeroMQ PUB socket
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    # Hàng đợi chỉ lưu 1 frame mới nhất, drop frame cũ để chống tích lũy trễ
    sock.setsockopt(zmq.SNDHWM, 2)
    bind_address = f"tcp://*:{args.port}"
    sock.bind(bind_address)

    print("=" * 65)
    print(" CAMERA ETHERNET STREAMING SERVER (ROBOT ONBOARD NODE) ")
    print("=" * 65)
    print(f"  Giao thức:      ZeroMQ PUB")
    print(f"  Cổng mạng:      {args.port} (Đang lắng nghe: {bind_address})")
    print(f"  Độ phân giải:   {args.width}x{args.height} @ {args.fps} FPS")
    print(f"  Chất lượng JPEG:{args.jpeg_quality}%")
    print(f"  Nguồn camera:   {args.camera.upper()}")
    print("=" * 65)

    # Khởi tạo nguồn camera tương ứng
    cam_source = None
    if args.camera == "realsense":
        try:
            cam_source = RealSenseSource(
                width=args.width,
                height=args.height,
                fps=args.fps,
                enable_depth=args.enable_depth
            )
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khởi động RealSense: {e}. Đang chuyển sang dự phòng OpenCV ID 0...")
            cam_source = OpenCVSource(device=0, width=args.width, height=args.height, fps=args.fps)
    else:
        cam_source = OpenCVSource(device=args.device, width=args.width, height=args.height, fps=args.fps)

    # Đăng ký xử lý tín hiệu thoát (Ctrl+C)
    running = [True]
    def _sig_handler(sig, frame):
        print("\n[CameraServer] Đang dừng máy chủ phát hình...")
        running[0] = False
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_count = 0
    fps_start = time.perf_counter()
    target_dt = 1.0 / max(1, args.fps)

    print("[CameraServer] Đang phát stream qua mạng LAN...")
    try:
        while running[0]:
            t0 = time.perf_counter()

            frame_bgr, depth_raw = cam_source.get_frames()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            # Nén ảnh màu sang JPEG
            ret, jpg_buffer = cv2.imencode(".jpg", frame_bgr, encode_params)
            if not ret:
                continue

            # Đóng gói multipart message
            msg = [jpg_buffer.tobytes()]
            if depth_raw is not None:
                # Gửi buffer depth dạng raw bytes (uint16)
                msg.append(depth_raw.tobytes())

            sock.send_multipart(msg)

            frame_count += 1
            now = time.perf_counter()
            if now - fps_start >= 3.0:
                cur_fps = frame_count / (now - fps_start)
                kb_size = len(jpg_buffer) / 1024.0
                mbps = (cur_fps * len(jpg_buffer) * 8) / (1024.0 * 1024.0)
                print(f"[CameraServer] Đang phát: {cur_fps:.1f} FPS | Kích thước frame: {kb_size:.1f} KB | Băng thông: {mbps:.2f} Mbps")
                frame_count = 0
                fps_start = now

            elapsed = time.perf_counter() - t0
            sleep_time = target_dt - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    finally:
        cam_source.release()
        sock.close(linger=0)
        ctx.term()
        print("[CameraServer] Đã giải phóng socket và tắt camera an toàn.")


if __name__ == "__main__":
    main()
