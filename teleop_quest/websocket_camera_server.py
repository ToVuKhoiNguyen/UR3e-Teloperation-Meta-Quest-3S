#!/usr/bin/env python3
"""
Camera WebSocket Streaming Server:
- Thu thập luồng hình ảnh từ Camera Intel RealSense (pyrealsense2) hoặc USB/Laptop Webcam (OpenCV).
- Nén JPEG tối ưu băng thông với độ trễ siêu thấp (Low Latency).
- Phát stream thời gian thực qua giao thức chuẩn WebSocket (RFC 6455).
- Tương thích 100% với EthernetWristCamera trên máy tính điều khiển Teleoperation.

Hướng dẫn sử dụng:
  # Phát từ Intel RealSense:
  python3 teleop_quest/websocket_camera_server.py --camera realsense --port 8765 --fps 30

  # Phát từ Webcam (mặc định /dev/video0):
  python3 teleop_quest/websocket_camera_server.py --camera opencv --device 0 --port 8765 --fps 30
"""

import sys
import time
import argparse
import asyncio
import signal
import struct
import cv2
import numpy as np

try:
    import websockets
except ImportError:
    print("[WebSocketServer] LỖI: Chưa cài đặt thư viện 'websockets'.")
    print("Vui lòng cài đặt bằng lệnh: pip install websockets")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Camera WebSocket Streaming Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ IP lắng nghe (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Cổng WebSocket (mặc định: 8765)")
    parser.add_argument("--camera", type=str, choices=["realsense", "opencv"], default="realsense", help="Driver camera: realsense hoặc opencv")
    parser.add_argument("--device", type=int, default=0, help="Device ID khi dùng OpenCV (mặc định: 0)")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng khung hình (mặc định: 640)")
    parser.add_argument("--height", type=int, default=480, help="Chiều cao khung hình (mặc định: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Tốc độ khung hình mục tiêu (mặc định: 30)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Chất lượng nén JPEG 1-100 (mặc định: 80)")
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
        try:
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
        except Exception:
            return None, None

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


CLIENTS = set()

async def ws_handler(websocket):
    client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}" if websocket.remote_address else "unknown"
    CLIENTS.add(websocket)
    print(f"[WebSocketServer] Client kết nối thành công: {client_addr} (Tổng clients: {len(CLIENTS)})")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"[WebSocketServer] Client ngắt kết nối: {client_addr} (Còn lại: {len(CLIENTS)})")


async def streamer_task(cam_source, fps: int, quality: int):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    target_dt = 1.0 / max(1, fps)
    frame_count = 0
    fps_start = time.perf_counter()
    loop = asyncio.get_running_loop()

    while True:
        t0 = time.perf_counter()

        # Đọc frame từ camera trong executor để không block event loop
        frame_bgr, depth_raw = await loop.run_in_executor(None, cam_source.get_frames)
        if frame_bgr is None:
            await asyncio.sleep(0.01)
            continue

        # Nén JPEG trong executor
        ret, jpg_buffer = cv2.imencode(".jpg", frame_bgr, encode_params)
        if not ret:
            await asyncio.sleep(0.005)
            continue

        rgb_bytes = jpg_buffer.tobytes()
        depth_bytes = depth_raw.tobytes() if depth_raw is not None else b""

        # Đóng gói packet: [4B len_rgb][4B len_depth][rgb_bytes][depth_bytes]
        header = struct.pack("!II", len(rgb_bytes), len(depth_bytes))
        packet = header + rgb_bytes + depth_bytes

        # Broadcast tới tất cả client đang kết nối
        if CLIENTS:
            dead_clients = []
            for client in list(CLIENTS):
                try:
                    await client.send(packet)
                except websockets.exceptions.ConnectionClosed:
                    dead_clients.append(client)
                except Exception:
                    pass
            for dc in dead_clients:
                CLIENTS.discard(dc)

        frame_count += 1
        now = time.perf_counter()
        if now - fps_start >= 3.0:
            cur_fps = frame_count / (now - fps_start)
            kb_size = len(rgb_bytes) / 1024.0
            mbps = (cur_fps * len(packet) * 8) / (1024.0 * 1024.0)
            if CLIENTS:
                print(f"[WebSocketServer] Đang phát: {cur_fps:.1f} FPS | Frame: {kb_size:.1f} KB | Băng thông: {mbps:.2f} Mbps | Clients: {len(CLIENTS)}")
            frame_count = 0
            fps_start = now

        elapsed = time.perf_counter() - t0
        sleep_time = target_dt - elapsed
        if sleep_time > 0.001:
            await asyncio.sleep(sleep_time)
        else:
            await asyncio.sleep(0.001)


async def main_async(args):
    # Khởi tạo nguồn camera
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
            print(f"[Cảnh báo] Không thể mở RealSense ({e}). Đang chuyển sang dự phòng Webcam OpenCV ID 0...")
            cam_source = OpenCVSource(device=0, width=args.width, height=args.height, fps=args.fps)
    else:
        cam_source = OpenCVSource(device=args.device, width=args.width, height=args.height, fps=args.fps)

    print("=" * 65)
    print(" CAMERA WEBSOCKET STREAMING SERVER ")
    print("=" * 65)
    print(f"  Giao thức:      WebSocket (RFC 6455)")
    print(f"  Endpoint:       ws://{args.host}:{args.port}")
    print(f"  Độ phân giải:   {args.width}x{args.height} @ {args.fps} FPS")
    print(f"  Chất lượng JPEG:{args.jpeg_quality}%")
    print(f"  Nguồn camera:   {args.camera.upper()}")
    print("=" * 65)

    try:
        async with websockets.serve(ws_handler, args.host, args.port, max_size=None):
            print(f"[WebSocketServer] Server đang lắng nghe kết nối tại: ws://{args.host}:{args.port}")
            await streamer_task(cam_source, args.fps, args.jpeg_quality)
    finally:
        cam_source.release()
        print("[WebSocketServer] Đã tắt camera và đóng WebSocket Server an toàn.")


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, SystemExit):
        print("\n[WebSocketServer] Dừng máy chủ thành công.")


if __name__ == "__main__":
    main()
