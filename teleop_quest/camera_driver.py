import threading
import time
from typing import Optional, Union, Tuple
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
    Vẽ lớp phủ hiển thị OSD hiện đại lên khung hình xem trước:
    - Tự động nhận diện ảnh đơn (RGB / Depth) hoặc ảnh ghép đôi (Side-by-side).
    - Hiển thị tên thiết bị, FPS, chế độ hiển thị, trạng thái ngàm kẹp và tâm ngắm crosshair.
    """
    if frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]

    # Thanh header sẫm màu bán trong suốt Slate
    cv2.rectangle(out, (0, 0), (w, 32), (15, 23, 42), -1)

    is_sbs = (w >= 1.5 * h) or (stream_mode == "side_by_side" and w > 640)

    if is_sbs:
        mid_x = w // 2
        # Vạch kẻ phân cách giữa RGB và Depth
        cv2.line(out, (mid_x, 0), (mid_x, h), (0, 220, 255), 1, cv2.LINE_AA)

        # Tiêu đề nửa trái (RGB)
        left_text = f"RGB | {device_name.upper()} | {fps:.1f} FPS"
        cv2.putText(out, left_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Tiêu đề nửa phải (Depth Aligned)
        right_text = "DEPTH (ALIGNED) | TURBO COLORIZER"
        cv2.putText(out, right_text, (mid_x + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 255), 1, cv2.LINE_AA)

        # Trạng thái ngàm kẹp ở góc phải
        status_text = "GRIPPER: CLOSED [100%]" if gripper_closed else "GRIPPER: OPEN [0%]"
        status_color = (255, 80, 80) if gripper_closed else (80, 255, 120)
        cv2.putText(out, status_text, (w - 230, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        # Tâm ngắm crosshair ở tâm mỗi nửa ảnh
        cx1, cy1 = mid_x // 2, h // 2
        cx2, cy2 = mid_x + mid_x // 2, h // 2
        for (cx, cy) in ((cx1, cy1), (cx2, cy2)):
            cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 15, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        mode_str = "DEPTH" if stream_mode == "depth" else "RGB"
        title_text = f"{device_name.upper()} [{mode_str}] | {fps:.1f} FPS"
        cv2.putText(out, title_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        status_text = "GRIPPER: CLOSED [100%]" if gripper_closed else "GRIPPER: OPEN [0%]"
        status_color = (255, 80, 80) if gripper_closed else (80, 255, 120)
        cv2.putText(out, status_text, (w - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        cx, cy = w // 2, h // 2
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 18, (0, 255, 255), 1, cv2.LINE_AA)

    return out


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
        if stream_mode == "depth":
            depth_rgb, depth_raw = self.cam.render_depth(
                tool_pos=tool_pos,
                tool_quat=tool_quat,
                tomato_pos=tomato_pos,
                basket_pos=basket_pos,
                gripper_closed=gripper_closed
            )
            self._latest_depth_raw = depth_raw
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
            return np.hstack([rgb, depth_rgb])
        else: # "rgb"
            return self.cam.render(
                tool_pos=tool_pos,
                tool_quat=tool_quat,
                tomato_pos=tomato_pos,
                basket_pos=basket_pos,
                gripper_closed=gripper_closed
            )

    def get_depth_frame(self) -> Optional[np.ndarray]:
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
                # Căn chỉnh pixel-to-pixel giữa Depth và RGB
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

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


def create_wrist_camera(
    camera_type: str = "synthetic",
    device_index: Union[int, str] = 0,
    serial_number: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 65.0,
    fps: int = 30
) -> BaseWristCamera:
    """
    Factory khởi tạo camera cổ tay UR3e:
    - camera_type = 'synthetic': Sử dụng Camera ảo mô phỏng 3D Eye-in-Hand.
    - camera_type in ('realsense', 'd455', 'd435', 'd405'): Sử dụng SDK pyrealsense2 chính thức cho RealSense.
    - camera_type = 'real': Ưu tiên kiểm tra RealSense nếu có thiết bị cắm sẵn, nếu không sẽ dùng OpenCV V4L2.
    """
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
