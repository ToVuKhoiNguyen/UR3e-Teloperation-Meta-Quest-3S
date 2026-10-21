import asyncio
import http.server
import json
import os
import queue
import ssl
import threading
import time
import websockets

class QuestServer:
    def __init__(self, cert: str, key: str, host: str = "0.0.0.0", http_port: int = 8443, ws_port: int = 8444, ws_port_plain: int = 8445, record_file: str = None):
        self.cert = cert
        self.key = key
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.ws_port_plain = ws_port_plain

        if record_file is None:
            record_file = os.environ.get("QUEST_RECORD_FILE", None)
        if record_file is None:
            import sys
            if "--record" in sys.argv:
                try:
                    idx = sys.argv.index("--record")
                    if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                        record_file = sys.argv[idx + 1]
                except Exception:
                    pass
        self.record_file = record_file

        self._lock = threading.RLock()
        self._frame_cond = threading.Condition(self._lock)
        self._running = True
        self._latest_frame = None
        self._last_seen = 0.0
        self._connected = False
        self._clients = set()
        self._loop = None
        self._estop_triggered = False
        self._home_requested = False
        self._gripper_toggle_requested = False
        self._latest_jpeg = None
        self._drive_mode = False
        self._latest_telemetry_bytes = None
        self._latest_telemetry_dict = None

        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        self._cam_file = os.path.join(self.static_dir, "wrist_camera.jpg")

        self._record_queue = queue.Queue() if self.record_file else None
        self._record_thread = None
        self._prev_trigger = False
        self._latest_q0 = None
        self._q0_provider = None

    def start(self):
        # Khởi động HTTPS server (chạy daemon thread)
        http_thread = threading.Thread(target=self._run_http, daemon=True)
        http_thread.start()

        # Khởi động WSS / WS server (chạy daemon thread)
        ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        ws_thread.start()

        # Khởi động luồng ghi log buffer nếu cờ --record được bật
        if self._record_queue is not None:
            self._record_thread = threading.Thread(target=self._record_worker, daemon=True)
            self._record_thread.start()
            print(f"[Recorder] Đang ghi gói tin thô JSONL vào: {self.record_file}")

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    @property
    def last_frame(self):
        return self.get_frame()

    def last_seen(self) -> float:
        with self._lock:
            return self._last_seen

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected
            
    def get_drive_mode(self) -> bool:
        with self._lock:
            return self._drive_mode

    def set_drive_mode(self, enabled: bool):
        """Cập nhật trạng thái Drive Mode từ hệ thống và đồng bộ sang kính Quest."""
        with self._lock:
            self._drive_mode = bool(enabled)
            if self._latest_telemetry_dict is not None and "controller" in self._latest_telemetry_dict:
                self._latest_telemetry_dict["controller"]["drive_mode"] = bool(enabled)
                self._latest_telemetry_bytes = json.dumps(self._latest_telemetry_dict).encode("utf-8")
            clients = list(self._clients)
            loop = self._loop
        if clients and loop:
            msg = json.dumps({"type": "teleop_state", "enabled": bool(enabled)})
            for ws in clients:
                try:
                    asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                except Exception:
                    pass

    def check_and_clear_estop(self) -> bool:
        with self._lock:
            if self._estop_triggered:
                self._estop_triggered = False
                return True
            return False

    def check_and_clear_home(self) -> bool:
        with self._lock:
            if self._home_requested:
                self._home_requested = False
                return True
            return False

    def check_and_clear_gripper_toggle(self) -> bool:
        with self._lock:
            if self._gripper_toggle_requested:
                self._gripper_toggle_requested = False
                return True
            return False

    def send_haptic(self, intensity: float = 1.0, duration: float = 100):
        with self._lock:
            clients = list(self._clients)
            loop = self._loop
        if clients and loop:
            msg = json.dumps({"type": "haptic", "intensity": intensity, "duration": duration})
            for ws in clients:
                try:
                    asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                except Exception:
                    pass

    def broadcast_robot_telemetry(self, pos, quat, status: str = "OK", gripper: bool = False, joints: list = None):
        """Phát tọa độ TCP thực tế, quaternion và trạng thái robot + VR controller tới WebXR Cockpit HUD."""
        with self._lock:
            clients = list(self._clients)
            loop = self._loop
            latest_frame = dict(self._latest_frame) if self._latest_frame else None
            now = time.time()
            is_vr_connected = bool(self._connected and (now - self._last_seen < 3.0))
            drive_mode = self._drive_mode

        hand_name = "Waiting..."
        trigger_pressed = False
        grip_pressed = False
        if is_vr_connected and latest_frame:
            raw_hand = latest_frame.get("hand", "right")
            hand_name = "Right Hand" if raw_hand == "right" else ("Left Hand" if raw_hand == "left" else str(raw_hand))
            trigger_pressed = bool(latest_frame.get("trigger", False))
            grip_pressed = bool(latest_frame.get("grip", False))

        controller_info = {
            "connected": is_vr_connected,
            "hand": hand_name,
            "trigger": trigger_pressed,
            "grip": grip_pressed,
            "gripper_btn": bool(gripper),
            "drive_mode": bool(drive_mode)
        }

        msg_dict = {
            "type": "robot_telemetry",
            "pos": [round(float(p), 4) for p in pos] if pos is not None else [0.0, 0.0, 0.0],
            "quat": [round(float(q), 4) for q in quat] if quat is not None else [1.0, 0.0, 0.0, 0.0],
            "status": str(status),
            "gripper": bool(gripper),
            "controller": controller_info
        }
        if joints is not None:
            msg_dict["joints"] = [round(float(j), 3) for j in joints]
            with self._lock:
                if len(joints) > 0:
                    self._latest_q0 = float(joints[0])

        msg_bytes = json.dumps(msg_dict).encode("utf-8")
        with self._lock:
            self._latest_telemetry_bytes = msg_bytes
            self._latest_telemetry_dict = msg_dict

        if clients and loop:
            msg = msg_bytes.decode("utf-8")
            for ws in clients:
                try:
                    asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                except Exception:
                    pass

    def send_camera_frame(self, frame_rgb, quality: int = 75):
        """Mã hóa frame ảnh RGB thành JPEG, lưu cache đĩa và gửi tới client qua WebSocket & MJPEG Stream."""
        try:
            import cv2
            import base64
            # Tự động điều chỉnh kích thước theo tỉ lệ khung hình (giữ đúng tỉ lệ khi xem side-by-side)
            h_f, w_f = frame_rgb.shape[:2]
            aspect = float(w_f) / float(max(1, h_f))
            if aspect > 2.0:
                small = cv2.resize(frame_rgb, (480, int(480 / aspect)))
            else:
                small = cv2.resize(frame_rgb, (320, 240))
            bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encimg = cv2.imencode('.jpg', bgr, encode_param)
            if success:
                jpeg_bytes = encimg.tobytes()
                with self._lock:
                    self._latest_jpeg = jpeg_bytes
                    clients = list(self._clients)
                    loop = self._loop
                    self._frame_cond.notify_all()

                # Lưu ảnh trực tiếp ra static/wrist_camera.jpg phục vụ chế độ xem file:/// cục bộ
                try:
                    tmp_file = self._cam_file + ".tmp"
                    with open(tmp_file, "wb") as f:
                        f.write(jpeg_bytes)
                    os.replace(tmp_file, self._cam_file)
                except Exception:
                    pass

                # Phát luồng trực tiếp qua WebSocket tới các client đang kết nối
                if clients and loop:
                    b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
                    msg = json.dumps({"type": "camera", "image": b64_str})
                    for ws in clients:
                        try:
                            asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                        except Exception:
                            pass
        except Exception:
            pass

    def _run_http(self):
        # Cấu hình HTTPServer đa luồng (ThreadingHTTPServer)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.cert, self.key)

        static_dir = self.static_dir
        server_self = self

        class RequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=static_dir, **kwargs)

            def end_headers(self):
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def do_GET(self):
                req_path = self.path.split("?")[0]
                client_ip = self.client_address[0] if self.client_address else "unknown"
                if req_path in ("/", "/index.html"):
                    print(f"[WebXR HTTPS] Thiết bị ({client_ip}) ĐÃ KẾT NỐI thành công vào trang WebXR Cockpit!")
                # 1. Luồng video MJPEG liên tục thời gian thực (Native Realtime Stream)
                if req_path in ("/video_feed", "/stream.mjpg", "/wrist_camera_stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    last_sent = None
                    try:
                        while server_self._running:
                            with server_self._frame_cond:
                                server_self._frame_cond.wait(timeout=0.1)
                                jpeg_bytes = server_self._latest_jpeg
                            if jpeg_bytes and jpeg_bytes != last_sent:
                                last_sent = jpeg_bytes
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode())
                                self.wfile.write(jpeg_bytes)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, Exception):
                        pass
                    return

                # 2. Ảnh đơn chụp nhanh (Snapshot)
                if req_path == "/wrist_camera.jpg" or req_path.endswith("wrist_camera.jpg"):
                    with server_self._lock:
                        jpeg_bytes = server_self._latest_jpeg
                    if jpeg_bytes:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Pragma", "no-cache")
                        self.send_header("Expires", "0")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        return
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return

                # 3. Telemetry JSON thời gian thực (hỗ trợ HTTP polling 10Hz-20Hz cho màn hình PC)
                if req_path in ("/telemetry.json", "/telemetry"):
                    with server_self._lock:
                        t_bytes = server_self._latest_telemetry_bytes
                    if not t_bytes:
                        default_dict = {
                            "type": "robot_telemetry",
                            "pos": [0.0, 0.0, 0.0],
                            "quat": [1.0, 0.0, 0.0, 0.0],
                            "status": "Waiting...",
                            "gripper": False,
                            "controller": {
                                "connected": False,
                                "hand": "Waiting...",
                                "trigger": False,
                                "grip": False,
                                "gripper_btn": False,
                                "drive_mode": False
                            }
                        }
                        t_bytes = json.dumps(default_dict).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(t_bytes)))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(t_bytes)
                    return

                # 4. REST Endpoints hỗ trợ điều khiển khi WebSocket bị giới hạn
                if req_path == "/api/go_home":
                    with server_self._lock:
                        server_self._home_requested = True
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                    return

                if req_path == "/api/toggle_gripper":
                    with server_self._lock:
                        server_self._gripper_toggle_requested = True
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                    return

                if req_path == "/api/drive_mode":
                    query_str = self.path.split("?")[1] if "?" in self.path else ""
                    is_drive = "state=true" in query_str.lower()
                    server_self.set_drive_mode(is_drive)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                    return

                super().do_GET()

            def log_message(self, format, *args):
                pass  # Tắt log mặc định để tránh spam terminal

        try:
            httpd = http.server.ThreadingHTTPServer((self.host, self.http_port), RequestHandler)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            print(f"[Quest] HTTPS Server running at https://{self.host}:{self.http_port}")
            httpd.serve_forever()
        except OSError as e:
            if e.errno == 98:
                print(f"[Quest] Lỗi: Cổng {self.http_port} đã bị chiếm dụng. Vui lòng tắt các tiến trình cũ.")
            else:
                print(f"[Quest] HTTPS Error: {e}")

    def _run_ws(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_main())

    async def _ws_main(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.cert, self.key)

        async def handler(websocket):
            remote_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
            print(f"[WebXR WebSocket] KÍNH QUEST ĐÃ KẾT NỐI ĐIỀU KHIỂN THÀNH CÔNG từ {remote_ip}!")
            with self._lock:
                self._connected = True
                self._clients.add(websocket)
                current_mode = self._drive_mode
            try:
                # Gửi trạng thái Teleop hiện tại ngay khi client vừa kết nối
                await websocket.send(json.dumps({"type": "teleop_state", "enabled": current_mode}))
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        with self._lock:
                            if data.get("type") == "estop":
                                self._estop_triggered = True
                            elif data.get("type") == "ui_state":
                                self._drive_mode = bool(data.get("drive_mode", False))
                            elif data.get("type") == "go_home":
                                self._home_requested = True
                                if self._latest_frame is not None:
                                    self._latest_frame["home"] = True
                                    self._latest_frame["thumbstick_click"] = True
                            elif data.get("type") == "toggle_gripper":
                                self._gripper_toggle_requested = True
                            else:
                                self._latest_frame = data
                                self._last_seen = time.time()
                                self._connected = True
                                if data.get("home_btn") or data.get("home") or data.get("thumbstick_click"):
                                    self._home_requested = True
                                if data.get("gripper_toggle_btn"):
                                    self._gripper_toggle_requested = True
                                # Đồng bộ tức thời trạng thái tay cầm vào telemetry cache để HTTP polling cập nhật nhạy
                                if self._latest_telemetry_dict is not None:
                                    raw_h = data.get("hand", "right")
                                    h_str = "Right Hand" if raw_h == "right" else ("Left Hand" if raw_h == "left" else str(raw_h))
                                    self._latest_telemetry_dict["controller"] = {
                                        "connected": True,
                                        "hand": h_str,
                                        "trigger": bool(data.get("trigger", False)),
                                        "grip": bool(data.get("grip", False)),
                                        "gripper_btn": bool(data.get("gripper_btn", False)),
                                        "drive_mode": bool(self._drive_mode)
                                    }
                                    self._latest_telemetry_bytes = json.dumps(self._latest_telemetry_dict).encode("utf-8")
                        if self._record_queue is not None:
                            self._record_frame(data)
                    except json.JSONDecodeError:
                        pass
            except (websockets.exceptions.ConnectionClosed, Exception):
                pass
            finally:
                print(f"[WebXR WebSocket] Kính Quest đã ngắt kết nối điều khiển từ {remote_ip}")
                with self._lock:
                    self._clients.discard(websocket)
                    if not self._clients:
                        self._connected = False
                        self._latest_frame = None
                        self._drive_mode = False

        try:
            # Chạy đồng thời WSS (cổng 8444 cho Quest WebXR) và WS (cổng 8445 cho PC/local preview)
            async with websockets.serve(handler, self.host, self.ws_port, ssl=context):
                async with websockets.serve(handler, self.host, self.ws_port_plain):
                    print(f"[Quest] WSS Server running at wss://{self.host}:{self.ws_port}")
                    print(f"[Quest] WS Server (Plain) running at ws://{self.host}:{self.ws_port_plain}")
                    await asyncio.Future()  # run forever
        except OSError as e:
            if e.errno == 98:
                print(f"[Quest] Lỗi: Cổng WebSocket ({self.ws_port} hoặc {self.ws_port_plain}) đã bị chiếm dụng.")
            else:
                print(f"[Quest] WSS Error: {e}")

    def set_q0_provider(self, func):
        """Đăng ký callback hàm trả về góc quay q0 hiện tại của robot theo thời gian thực."""
        with self._lock:
            self._q0_provider = func

    def _record_frame(self, data: dict):
        if not isinstance(data, dict):
            return
        if data.get("type") in ("estop", "ui_state", "go_home", "toggle_gripper"):
            return

        q0 = None
        with self._lock:
            provider = self._q0_provider
            latest_q0 = self._latest_q0
        if provider is not None:
            try:
                val = provider()
                if val is not None:
                    q0 = float(val)
            except Exception:
                pass
        if q0 is None:
            q0 = latest_q0

        trigger_cur = bool(data.get("trigger", False))
        event = None
        if trigger_cur and not self._prev_trigger:
            event = "trigger_start"
        elif not trigger_cur and self._prev_trigger:
            event = "trigger_end"
        self._prev_trigger = trigger_cur

        record_entry = {
            "timestamp": data.get("t", time.time()),
            "pos": data.get("pos"),
            "quat": data.get("quat"),
            "head_quat": data.get("head_quat"),
            "head_pos": data.get("head_pos"),
            "buttons": {
                "trigger": trigger_cur,
                "grip": bool(data.get("grip", False)),
                "gripper_raw": bool(data.get("gripper_raw", False)),
                "gripper_btn": bool(data.get("gripper_btn", False)),
                "button_a": bool(data.get("button_a", False)),
                "button_b": bool(data.get("button_b", False)),
                "thumbstick": bool(data.get("thumbstick", False)),
                "thumbstick_click": bool(data.get("thumbstick_click", False)),
                "drive_mode": bool(data.get("drive_mode", self._drive_mode))
            },
            "trigger": trigger_cur,
            "grip": bool(data.get("grip", False)),
            "q0": round(float(q0), 5) if q0 is not None else None,
            "event": event
        }
        try:
            self._record_queue.put_nowait(record_entry)
        except Exception:
            pass

    def _record_worker(self):
        if not self.record_file:
            return
        try:
            parent_dir = os.path.dirname(self.record_file)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(self.record_file, "a", buffering=64*1024, encoding="utf-8") as f:
                while self._running:
                    try:
                        item = self._record_queue.get(timeout=0.2)
                        f.write(json.dumps(item) + "\n")
                        while not self._record_queue.empty():
                            try:
                                it = self._record_queue.get_nowait()
                                f.write(json.dumps(it) + "\n")
                            except queue.Empty:
                                break
                        f.flush()
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"[Recorder Error] Không thể ghi file {self.record_file}: {e}")

if __name__ == "__main__":
    # Test server standalone
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Quest Server standalone test")
    parser.add_argument("--record", type=str, default=None, help="Ghi dữ liệu thô ra file JSONL")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8443)
    parser.add_argument("--ws-port", type=int, default=8444)
    parser.add_argument("--ws-port-plain", type=int, default=8445)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    cert_path = os.path.join(base_dir, "certs", "cert.pem")
    key_path = os.path.join(base_dir, "certs", "key.pem")
    
    if not os.path.exists(cert_path):
        print(f"[Error] Missing cert file at {cert_path}")
        sys.exit(1)
        
    server = QuestServer(cert_path, key_path, host=args.host, http_port=args.http_port, ws_port=args.ws_port, ws_port_plain=args.ws_port_plain, record_file=args.record)
    server.start()
    print("Press Ctrl+C to stop.")
    try:
        while True:
            frame = server.get_frame()
            if frame:
                # Xóa màn hình terminal và in đè lên
                print(f"\r[Tọa độ VR] X:{frame['pos'][0]:.3f} | Y:{frame['pos'][1]:.3f} | Z:{frame['pos'][2]:.3f} | Trigger: {frame.get('trigger')} | Grip: {frame.get('grip')}     ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Đã thoát]")
