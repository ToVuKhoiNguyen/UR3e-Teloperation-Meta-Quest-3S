import threading
import time
from typing import Optional, Union, Tuple, Dict, Any
import cv2
import numpy as np

from teleop_quest.synthetic_camera import SyntheticWristCamera


def draw_camera_osd(
    frame: np.ndarray,
    device_name: str,
    fps: float,
    gripper_closed: bool,
    stream_mode: str = "rgb"
) -> np.ndarray:
    """
    Trả về khung hình video stream thuần túy sạch 100%:
    - Không vẽ thanh bar đen hay nhãn trạng thái OSD.
    - Không vẽ chữ đè (text) hay tâm ngắm (crosshair).
    """
    if frame is None:
        return frame
    return frame.copy()


class BaseWristCamera:
    """Lớp cơ sở cho Camera gắn cổ tay robot (Eye-in-Hand)."""
    def get_frame(self, stream_mode: str = "rgb", **kwargs) -> np.ndarray:
        raise NotImplementedError

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Trả về dữ liệu bản đồ độ sâu thô dạng uint16 (mm) nếu có."""
        return None

    def add_osd_preview(
        self,
        frame_rgb: np.ndarray,
        gripper_closed: bool = False,
        stream_mode: str = "rgb",
        **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    def is_real(self) -> bool:
        return False

    def get_frame_metadata(self) -> Dict[str, Any]:
        """Trả về metadata khung hình: timestamp phần cứng, timestamp toàn cục (ns), frame number."""
        t_ns = time.perf_counter_ns()
        return {
            "timestamp_hw_ms": t_ns / 1e6,
            "timestamp_global_ns": t_ns,
            "frame_number": 0,
            "domain": "system",
            "latency_ms": 0.0,
        }

    def release(self):
        pass


class SyntheticWristCameraWrapper(BaseWristCamera):
    """Bọc Camera ảo 3D mô phỏng góc nhìn cổ tay robot UR3e."""
    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 65.0):
        self.cam = SyntheticWristCamera(width=width, height=height, fov_deg=fov_deg)
        self.width = width
        self.height = height
        self._latest_depth_raw = None
        self._fps_actual = 20.0
        self._frame_seq = 0

    def get_frame(
        self,
        tool_pos: Optional[Union[list, np.ndarray]] = None,
        tool_quat: Optional[Union[list, np.ndarray]] = None,
        tomato_pos: Optional[np.ndarray] = None,
        basket_pos: Optional[np.ndarray] = None,
        gripper_closed: bool = False,
        stream_mode: str = "rgb",
        **kwargs
    ) -> np.ndarray:
        self._frame_seq += 1
        if stream_mode == "depth":
            depth_rgb, depth_raw = self.cam.render_depth(
                tool_pos=tool_pos,
                tool_quat=tool_quat,
                tomato_pos=tomato_pos,
                basket_pos=basket_pos,
                gripper_closed=gripper_closed
            )
            self._latest_depth_raw = depth_raw
            self._latest_ts_ns = time.perf_counter_ns()
            return depth_rgb
        elif stream_mode == "side_by_side":
            rgb = self.cam.render(
                tool_pos=tool_pos,
                tool_quat=tool_quat,
                tomato_pos=tomato_pos,
                basket_pos=basket_pos,
                gripper_closed=gripper_closed
            )
            depth_rgb, depth_raw = self.cam.render_depth(
                tool_pos=tool_pos,
                tool_quat=tool_quat,
                tomato_pos=tomato_pos,
                basket_pos=basket_pos,
                gripper_closed=gripper_closed
            )
            self._latest_depth_raw = depth_raw
            self._latest_ts_ns = time.perf_counter_ns()
            return np.hstack([rgb, depth_rgb])
        else: # "rgb"
            frame = self.cam.render(
                tool_pos=tool_pos,
                tool_quat=tool_quat,
                tomato_pos=tomato_pos,
                basket_pos=basket_pos,
                gripper_closed=gripper_closed
            )
            self._latest_ts_ns = time.perf_counter_ns()
            return frame

    def get_depth_frame(self) -> Optional[np.ndarray]:
        return self._latest_depth_raw.copy() if self._latest_depth_raw is not None else None

    def get_frame_metadata(self) -> Dict[str, Any]:
        t_ns = getattr(self, "_latest_ts_ns", time.perf_counter_ns())
        return {
            "timestamp_hw_ms": t_ns / 1e6,
            "timestamp_global_ns": t_ns,
            "frame_number": self._frame_seq,
            "domain": "synthetic_sim",
            "latency_ms": 0.0,
        }

    def add_osd_preview(
        self,
        frame_rgb: np.ndarray,
        gripper_closed: bool = False,
        stream_mode: str = "rgb",
        **kwargs
    ) -> np.ndarray:
        return draw_camera_osd(
            frame_rgb,
            device_name="EYE-IN-HAND (SIMULATION)",
            fps=self._fps_actual,
            gripper_closed=gripper_closed,
            stream_mode=stream_mode
        )

    def is_real(self) -> bool:
        return False


class RealSenseWristCamera(BaseWristCamera):
    """
    Driver chuyên dụng cho Intel RealSense (D455 / D435 / D405 / D415):
    - Sử dụng Intel RealSense SDK (pyrealsense2) chính thức.
    - Cấu hình đồng thời 2 luồng: RGB Stream (rgb8) và Depth Stream (z16) @ 30 FPS.
    - Căn chỉnh pixel-to-pixel bằng rs.align(rs.stream.color).
    - Tự động mã hóa Depth sang bản đồ nhiệt màu qua Intel RealSense Colorizer (Jet/Turbo).
    - Hỗ trợ linh hoạt 3 chế độ stream_mode: 'rgb', 'depth', 'side_by_side'.
    - Tự động kết nối lại (Auto-reconnect) nếu cáp USB bị lỏng hoặc chập chờn.
    - Đảm bảo Zero-Block: thu nhận hình ảnh hoàn toàn trong background thread độc lập.
    """
    def __init__(
        self,
        serial_number: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps

        self._lock = threading.Lock()
        self._running = True
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_raw: Optional[np.ndarray] = None
        self._fps_actual = 0.0
        self._is_connected = False
        self._device_name = "RealSense Camera"
        self._hw_time_offset_ns: Optional[int] = None
        self._latest_metadata: Dict[str, Any] = {}

        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("[RealSense] LỖI: Chưa cài đặt thư viện 'pyrealsense2'. Hãy chạy: pip install pyrealsense2")
            return

        pipeline = None
        align = None
        colorizer = None
        frame_count = 0
        fps_start = time.perf_counter()

        while self._running:
            if pipeline is None:
                try:
                    ctx = rs.context()
                    devices = list(ctx.devices)
                    if not devices:
                        with self._lock:
                            self._is_connected = False
                            self._device_name = "RealSense (Chưa kết nối)"
                        time.sleep(1.0)
                        continue

                    # Tìm thiết bị phù hợp
                    target_dev = None
                    if self.serial_number:
                        for d in devices:
                            if d.get_info(rs.camera_info.serial_number) == str(self.serial_number):
                                target_dev = d
                                break
                    else:
                        target_dev = devices[0]

                    if target_dev is None:
                        with self._lock:
                            self._is_connected = False
                        time.sleep(1.0)
                        continue

                    dev_name = target_dev.get_info(rs.camera_info.name)
                    dev_sn = target_dev.get_info(rs.camera_info.serial_number)
                    dev_usb = target_dev.get_info(rs.camera_info.usb_type_descriptor)

                    config = rs.config()
                    config.enable_device(dev_sn)
                    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
                    config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

                    p = rs.pipeline()
                    p.start(config)

                    align = rs.align(rs.stream.color)
                    colorizer = rs.colorizer()

                    pipeline = p
                    with self._lock:
                        self._is_connected = True
                        self._device_name = f"{dev_name} (USB {dev_usb})"
                    print(f"[RealSense] ĐÃ KẾT NỐI: {dev_name} | Serial: {dev_sn} | Cổng: USB {dev_usb} | RGB+Depth {self.width}x{self.height}@{self.fps}FPS")
                except Exception as e:
                    with self._lock:
                        self._is_connected = False
                    pipeline = None
                    time.sleep(1.5)
                    continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                t_arrival_ns = time.perf_counter_ns()
                # Căn chỉnh pixel-to-pixel giữa Depth và RGB
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Đọc Hardware Timestamp & Domain từ ASIC RealSense
                hw_ts_ms = float(color_frame.get_timestamp())
                hw_domain = str(color_frame.get_frame_timestamp_domain())
                frame_num = int(color_frame.get_frame_number())

                # Tính toán Offset ở frame đầu tiên để bù trôi pha
                hw_ts_ns = int(hw_ts_ms * 1_000_000)
                if self._hw_time_offset_ns is None:
                    self._hw_time_offset_ns = t_arrival_ns - hw_ts_ns
                    print(f"[RealSense HW-Sync] Đã khóa pha Hardware Timestamp: Offset={self._hw_time_offset_ns} ns ({hw_domain})")

                # Timestamp toàn cục tính theo Master Clock đã bù trôi
                global_ts_ns = hw_ts_ns + self._hw_time_offset_ns

                # pyrealsense2 rs.format.rgb8 trả về mảng shape (H, W, 3) hệ màu RGB
                frame_rgb = np.asanyarray(color_frame.get_data())
                # pyrealsense2 colorizer chuyển depth z16 thành mảng RGB màu
                colored_depth_frame = colorizer.colorize(depth_frame)
                frame_depth = np.asanyarray(colored_depth_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())

                if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                    frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
                if frame_depth.shape[1] != self.width or frame_depth.shape[0] != self.height:
                    frame_depth = cv2.resize(frame_depth, (self.width, self.height))

                frame_count += 1
                now = time.perf_counter()
                if now - fps_start >= 1.0:
                    self._fps_actual = frame_count / (now - fps_start)
                    frame_count = 0
                    fps_start = now

                with self._lock:
                    self._latest_rgb = frame_rgb
                    self._latest_depth = frame_depth
                    self._latest_depth_raw = depth_raw
                    self._latest_metadata = {
                        "timestamp_hw_ms": hw_ts_ms,
                        "timestamp_global_ns": global_ts_ns,
                        "frame_number": frame_num,
                        "domain": hw_domain,
                        "latency_ms": (t_arrival_ns - global_ts_ns) / 1e6
                    }
                    self._is_connected = True

            except Exception:
                with self._lock:
                    self._is_connected = False
                try:
                    if pipeline is not None:
                        pipeline.stop()
                except Exception:
                    pass
                pipeline = None
                time.sleep(1.0)

        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _create_placeholder(self, stream_mode: str) -> np.ndarray:
        """Tạo ảnh chờ placeholder thông báo trực quan khi thiết bị chưa kết nối"""
        def make_box(title: str, subtitle: str, color=(255, 180, 50)):
            box = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            box[:] = (18, 22, 28)
            cv2.putText(box, title, (25, self.height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(box, subtitle, (25, self.height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
            return box

        if stream_mode == "depth":
            return make_box("INTEL REALSENSE DEPTH STREAM", "Please plug Intel RealSense USB 3.0 cable", (100, 220, 255))
        elif stream_mode == "side_by_side":
            b_rgb = make_box("RGB STREAM WAITING...", "Plug Intel RealSense USB 3.0", (255, 180, 50))
            b_depth = make_box("DEPTH STREAM WAITING...", "Pixel-to-Pixel Aligned Depth", (100, 220, 255))
            return np.hstack([b_rgb, b_depth])
        else:
            return make_box("INTEL REALSENSE D455/D435 WAITING...", "Please plug Intel RealSense USB 3.0 cable into PC", (255, 180, 50))

    def get_frame(self, stream_mode: str = "rgb", **kwargs) -> np.ndarray:
        with self._lock:
            if self._is_connected and self._latest_rgb is not None:
                if stream_mode == "depth" and self._latest_depth is not None:
                    return self._latest_depth.copy()
                elif stream_mode == "side_by_side" and self._latest_depth is not None:
                    return np.hstack([self._latest_rgb, self._latest_depth])
                else:
                    return self._latest_rgb.copy()

        return self._create_placeholder(stream_mode)

    def get_depth_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_depth_raw.copy() if self._latest_depth_raw is not None else None

    def get_frame_metadata(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_metadata)

    def add_osd_preview(
        self,
        frame_rgb: np.ndarray,
        gripper_closed: bool = False,
        stream_mode: str = "rgb",
        **kwargs
    ) -> np.ndarray:
        return draw_camera_osd(
            frame_rgb,
            device_name=self._device_name,
            fps=self._fps_actual,
            gripper_closed=gripper_closed,
            stream_mode=stream_mode
        )

    def is_real(self) -> bool:
        return True

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class RealWristCamera(BaseWristCamera):
    """
    Driver đọc luồng trực tiếp thời gian thực từ Camera vật lý thật:
    - Hỗ trợ bất kỳ USB / UVC Camera qua OpenCV V4L2.
    - Hỗ trợ mô phỏng pseudo-depth map khi chọn chế độ depth.
    - Chạy luồng nền chuyên dụng (Capture Thread) tách biệt khỏi vòng lặp điều khiển 20Hz của robot.
    - Tự động bắt lại kết nối nếu dây cắm camera bị lỏng hoặc chập chờn.
    """
    def __init__(
        self,
        device_index: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps

        self._lock = threading.Lock()
        self._running = True
        self._latest_frame: Optional[np.ndarray] = None
        self._fps_actual = 0.0
        self._is_connected = False

        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self):
        cap = None
        frame_count = 0
        fps_start = time.perf_counter()

        while self._running:
            if cap is None or not cap.isOpened():
                try:
                    cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(self.device_index)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
                        with self._lock:
                            self._is_connected = True
                        print(f"[RealCamera] Đã mở thành công camera thiết bị: {self.device_index}")
                    else:
                        with self._lock:
                            self._is_connected = False
                        time.sleep(1.0)
                        continue
                except Exception:
                    with self._lock:
                        self._is_connected = False
                    time.sleep(1.0)
                    continue

            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                with self._lock:
                    self._is_connected = False
                if cap is not None:
                    cap.release()
                cap = None
                time.sleep(0.5)
                continue

            if bgr_frame.shape[1] != self.width or bgr_frame.shape[0] != self.height:
                bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            frame_count += 1
            now = time.perf_counter()
            if now - fps_start >= 1.0:
                self._fps_actual = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

            with self._lock:
                self._latest_frame = rgb_frame
                self._is_connected = True

        if cap is not None:
            cap.release()

    def _make_pseudo_depth(self, frame_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        colored = cv2.applyColorMap(inv, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def get_frame(self, stream_mode: str = "rgb", **kwargs) -> np.ndarray:
        with self._lock:
            if self._latest_frame is not None and self._is_connected:
                rgb = self._latest_frame.copy()
                if stream_mode == "depth":
                    return self._make_pseudo_depth(rgb)
                elif stream_mode == "side_by_side":
                    return np.hstack([rgb, self._make_pseudo_depth(rgb)])
                return rgb

        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        placeholder[:] = (20, 25, 30)
        cv2.putText(
            placeholder,
            f"REAL CAMERA [DEV: {self.device_index}] WAITING...",
            (40, self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 50),
            2
        )
        cv2.putText(
            placeholder,
            "Connecting hardware sensor / RealSense...",
            (40, self.height // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )
        if stream_mode == "side_by_side":
            return np.hstack([placeholder, placeholder])
        return placeholder

    def add_osd_preview(
        self,
        frame_rgb: np.ndarray,
        gripper_closed: bool = False,
        stream_mode: str = "rgb",
        **kwargs
    ) -> np.ndarray:
        return draw_camera_osd(
            frame_rgb,
            device_name=f"V4L2 CAM [{self.device_index}]",
            fps=self._fps_actual,
            gripper_closed=gripper_closed,
            stream_mode=stream_mode
        )

    def is_real(self) -> bool:
        return True

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class EthernetWristCamera(BaseWristCamera):
    """
    Driver chuyên dụng nhận luồng hình ảnh qua mạng Ethernet từ máy tính nhúng gắn trên robot:
    - Hỗ trợ ZeroMQ SUB (khuyến nghị cho robotics): Cờ ZMQ_CONFLATE=1 triệt tiêu lag tích lũy.
    - Hỗ trợ fallback OpenCV VideoCapture qua HTTP / RTSP stream.
    - Tách biệt thu nhận hình ảnh trong background thread độc lập (Zero-Block).
    - Tự động kết nối lại (Auto-reconnect) khi ngắt cáp LAN hoặc server khởi động lại.
    - Trả về frame RGB và Depth tương thích 100% với BaseWristCamera.
    """
    def __init__(
        self,
        url: str = "tcp://192.168.1.100:5555",
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps

        self._lock = threading.Lock()
        self._running = True
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_raw: Optional[np.ndarray] = None
        self._fps_actual = 0.0
        self._is_connected = False
        self._device_name = f"Ethernet Cam ({url})"

        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self):
        """Worker thread chọn ZMQ SUB nếu url là tcp://, ngược lại dùng OpenCV stream"""
        if self.url.startswith("tcp://"):
            self._run_zmq_receiver()
        else:
            self._run_opencv_receiver()

    def _run_zmq_receiver(self):
        try:
            import zmq
        except ImportError:
            print("[EthernetCam] Cảnh báo: Chưa cài đặt thư viện 'pyzmq'. Hãy chạy: pip install pyzmq")
            print("[EthernetCam] Đang chờ thư viện pyzmq...")
            with self._lock:
                self._is_connected = False
                self._device_name = "Ethernet Cam (Thiếu pyzmq)"
            while self._running:
                time.sleep(2.0)
            return

        ctx = zmq.Context()
        frame_count = 0
        fps_start = time.perf_counter()

        while self._running:
            sock = None
            try:
                sock = ctx.socket(zmq.SUB)
                # CONFLATE = 1: Chỉ giữ đúng 1 frame mới nhất trong hàng đợi socket, drop toàn bộ frame cũ
                sock.setsockopt(zmq.CONFLATE, 1)
                sock.setsockopt(zmq.RCVTIMEO, 1500)  # Timeout 1.5s
                sock.setsockopt(zmq.SUBSCRIBE, b"")  # Lắng nghe mọi topic
                sock.connect(self.url)

                print(f"[EthernetCam] Đang lắng nghe luồng ZeroMQ tại: {self.url} (CONFLATE=1)...")

                while self._running:
                    try:
                        # Nhận tin nhắn (có thể là multipart [rgb_jpeg, depth_raw] hoặc single frame)
                        msg_parts = sock.recv_multipart()
                        if not msg_parts:
                            continue

                        # Giải mã RGB JPEG
                        jpg_bytes = msg_parts[0]
                        np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame_bgr is None:
                            continue
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        # Giải mã Depth nếu có multipart (part 2)
                        depth_frame = None
                        depth_raw = None
                        if len(msg_parts) >= 2 and len(msg_parts[1]) > 0:
                            depth_bytes = msg_parts[1]
                            if len(depth_bytes) == self.width * self.height * 2:
                                depth_raw = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((self.height, self.width))
                                depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                depth_frame = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                            else:
                                d_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
                                depth_frame = cv2.imdecode(d_arr, cv2.IMREAD_COLOR)

                        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

                        frame_count += 1
                        now = time.perf_counter()
                        if now - fps_start >= 1.0:
                            self._fps_actual = frame_count / (now - fps_start)
                            frame_count = 0
                            fps_start = now

                        with self._lock:
                            self._latest_rgb = frame_rgb
                            self._latest_depth = depth_frame
                            self._latest_depth_raw = depth_raw
                            self._is_connected = True
                            self._device_name = f"Ethernet ZMQ ({self.url})"

                    except zmq.Again:
                        # Hết thời gian chờ 1.5s không nhận được frame mới
                        with self._lock:
                            self._is_connected = False
                        continue

            except Exception as e:
                with self._lock:
                    self._is_connected = False
                time.sleep(1.0)
            finally:
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception:
                        pass

        ctx.term()

    def _run_opencv_receiver(self):
        """Dự phòng: Nhận stream qua OpenCV VideoCapture (HTTP MJPEG / RTSP)"""
        frame_count = 0
        fps_start = time.perf_counter()

        while self._running:
            cap = None
            try:
                cap = cv2.VideoCapture(self.url)
                if not cap.isOpened():
                    with self._lock:
                        self._is_connected = False
                    time.sleep(1.5)
                    continue

                while self._running:
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        with self._lock:
                            self._is_connected = False
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                        frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

                    frame_count += 1
                    now = time.perf_counter()
                    if now - fps_start >= 1.0:
                        self._fps_actual = frame_count / (now - fps_start)
                        frame_count = 0
                        fps_start = now

                    with self._lock:
                        self._latest_rgb = frame_rgb
                        self._is_connected = True
                        self._device_name = f"Ethernet Stream ({self.url})"

            except Exception:
                with self._lock:
                    self._is_connected = False
                time.sleep(1.0)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

    def _create_placeholder(self, stream_mode: str) -> np.ndarray:
        """Tạo ảnh placeholder thông báo trực quan khi chưa nhận được luồng qua mạng"""
        box = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        box[:] = (18, 22, 28)
        cv2.putText(box, "ETHERNET CAMERA WAITING...", (20, self.height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 200, 255), 2)
        cv2.putText(box, f"Connecting: {self.url}", (20, self.height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        if stream_mode == "side_by_side":
            b_depth = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            b_depth[:] = (18, 22, 28)
            cv2.putText(b_depth, "DEPTH NETWORK WAITING...", (20, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1)
            return np.hstack([box, b_depth])
        return box

    def get_frame(self, stream_mode: str = "rgb", **kwargs) -> np.ndarray:
        with self._lock:
            if self._is_connected and self._latest_rgb is not None:
                if stream_mode == "depth" and self._latest_depth is not None:
                    return self._latest_depth.copy()
                elif stream_mode == "side_by_side" and self._latest_depth is not None:
                    return np.hstack([self._latest_rgb, self._latest_depth])
                else:
                    return self._latest_rgb.copy()
        return self._create_placeholder(stream_mode)

    def get_depth_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_depth_raw.copy() if self._latest_depth_raw is not None else None

    def add_osd_preview(
        self,
        frame_rgb: np.ndarray,
        gripper_closed: bool = False,
        stream_mode: str = "rgb",
        **kwargs
    ) -> np.ndarray:
        return draw_camera_osd(
            frame_rgb,
            device_name=self._device_name,
            fps=self._fps_actual,
            gripper_closed=gripper_closed,
            stream_mode=stream_mode
        )

    def is_real(self) -> bool:
        return True

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def create_wrist_camera(
    camera_type: str = "synthetic",
    cam_source: str = "usb",
    cam_url: str = "tcp://192.168.1.100:5555",
    device_index: Union[int, str] = 0,
    serial_number: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 65.0,
    fps: int = 30
) -> BaseWristCamera:
    """
    Factory khởi tạo camera cổ tay UR3e:
    - cam_source == 'eth' hoặc camera_type in ('eth', 'ethernet'): Nhận stream qua cáp mạng Ethernet (ZMQ / RTSP / HTTP).
    - cam_source == 'usb' (Mặc định):
      + camera_type in ('realsense', 'd455', 'd435', 'd405'): Sử dụng SDK pyrealsense2 chính thức cho RealSense qua USB.
      + camera_type == 'real': Ưu tiên kiểm tra RealSense nếu có thiết bị cắm sẵn, nếu không sẽ dùng OpenCV V4L2.
      + camera_type == 'synthetic': Sử dụng Camera ảo mô phỏng 3D Eye-in-Hand.
    """
    # ── Phân nhánh 1: Kết nối qua mạng Ethernet ─────────────────────────────
    if cam_source == "eth" or camera_type.lower() in ("eth", "ethernet"):
        print(f"[Hệ thống] Khởi tạo Camera qua mạng Ethernet (Endpoint: {cam_url})")
        return EthernetWristCamera(url=cam_url, width=width, height=height, fps=fps)

    # ── Phân nhánh 2: Kết nối USB trực tiếp (GIỮ NGUYÊN 100% MÃ NGUỒN CŨ) ───
    c_type = camera_type.lower()
    if c_type in ("realsense", "d455", "d435", "d405"):
        print(f"[Hệ thống] Khởi tạo Driver Intel RealSense chuyên dụng (pyrealsense2, Serial: {serial_number or 'Auto'})")
        return RealSenseWristCamera(serial_number=serial_number, width=width, height=height, fps=fps)

    elif c_type == "real":
        # Tự động kiểm tra xem có thiết bị RealSense nào đang cắm không
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            if len(ctx.devices) > 0:
                print(f"[Hệ thống] Phát hiện {len(ctx.devices)} thiết bị Intel RealSense cắm qua USB -> Tự động dùng RealSense Driver!")
                return RealSenseWristCamera(serial_number=serial_number, width=width, height=height, fps=fps)
        except Exception:
            pass

        print(f"[Hệ thống] Khởi tạo Camera vật lý thật qua OpenCV V4L2 (thiết bị index: {device_index})")
        return RealWristCamera(device_index=device_index, width=width, height=height, fps=fps)

    else:
        print("[Hệ thống] Khởi tạo Camera ảo mô phỏng 3D (Synthetic Wrist Camera)")
        return SyntheticWristCameraWrapper(width=width, height=height, fov_deg=fov_deg)
