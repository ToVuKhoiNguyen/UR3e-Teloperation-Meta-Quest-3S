#!/usr/bin/env python3
"""
Mô-đun khởi tạo môi trường và hệ thống (Bootstrap):
- Tải cấu hình an toàn từ config/singularity.yaml
- Tự động kiểm tra và khởi chạy Docker URSim nếu chạy mô phỏng
- Giám sát tiến trình ADB nền để tự động reverse cổng khi cắm cáp USB Quest
- Hiển thị thông tin kết nối và banner khởi động
"""

import os
import time
import socket
import shutil
import subprocess
import threading
import yaml

def load_singularity_config(base_dir=None):
    """Tải cấu hình giám sát kỳ dị từ file YAML"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, "config", "singularity.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_ursim_container(robot_ip: str):
    """Kiểm tra và tự động khởi động URSim nếu chạy mô phỏng cục bộ (127.0.0.1)"""
    if robot_ip not in ("127.0.0.1", "localhost"):
        return

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((robot_ip, 29999))
        s.close()
    except Exception:
        docker_bin = shutil.which("docker")
        if docker_bin:
            try:
                print("[Hệ thống] Phát hiện URSim chưa mở cổng. Đang thử kích hoạt Docker URSim...")
                res = subprocess.run([docker_bin, "start", "ursim"], capture_output=True, text=True, timeout=5)
                if res.returncode != 0 and shutil.which("ros2"):
                    subprocess.Popen(
                        ["ros2", "run", "ur_client_library", "start_ursim.sh", "-m", "ur3e"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                time.sleep(3.5)
            except Exception as dock_err:
                print(f"[Hệ thống] Lưu ý khởi chạy URSim: {dock_err}")
        else:
            print("[Hệ thống] Lưu ý: URSim chưa phản hồi tại 127.0.0.1:29999 (nếu dùng URSim, hãy chắc chắn container URSim đã bật).")

_adb_watcher_thread = None

def start_adb_watcher(ports=(8443, 8444, 8445, 8080)):
    """
    Tự động kiểm tra và chuyển tiếp cổng ADB nền liên tục (daemon=True) lặp lại mỗi 2.5 giây.
    Tự động hồi phục kết nối (Auto-Heal) khi kính cắm cáp muộn, bấm Allow muộn hoặc tuột dây cắm lại.
    """
    global _adb_watcher_thread
    if _adb_watcher_thread is not None and _adb_watcher_thread.is_alive():
        return _adb_watcher_thread

    def _adb_watcher():
        adb_bin = shutil.which("adb")
        if not adb_bin:
            return

        is_reversed = False
        last_warn_t = 0.0

        # Khởi động adb daemon nếu chưa chạy
        try:
            subprocess.run([adb_bin, "start-server"], capture_output=True, timeout=3)
        except Exception:
            pass

        while True:
            try:
                # 1. Chạy adb start-server (ẩn output lỗi nếu có)
                subprocess.run([adb_bin, "start-server"], capture_output=True, timeout=3)

                # 2. Đọc kết quả lệnh adb devices
                res = subprocess.run([adb_bin, "devices"], capture_output=True, text=True, timeout=4)
                lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
                device_lines = [l for l in lines if not l.startswith("List of devices") and not l.startswith("*")]

                has_unauth = any("unauthorized" in dl for dl in device_lines)
                authorized_devices = []
                for dl in device_lines:
                    parts = dl.split()
                    if len(parts) >= 2 and parts[1] == "device":
                        authorized_devices.append(parts[0])

                # Trường hợp 1 (Chưa cắm cáp): Nếu danh sách trống -> đặt cờ is_reversed = False, im lặng đợi vòng lặp tiếp theo
                if not device_lines or (not has_unauth and not authorized_devices):
                    is_reversed = False

                # Trường hợp 2 (Chưa cấp quyền trên máy mới - unauthorized):
                elif has_unauth:
                    is_reversed = False
                    now = time.time()
                    if now - last_warn_t >= 5.0:
                        print("\n\033[93;1m⚠️  [QUEST USB] Phát hiện kính Quest nhưng CHƯA CẤP QUYỀN! Hãy đeo kính lên và bấm 'Allow USB Debugging' (Cho phép gỡ lỗi USB)!\033[0m")
                        print("\033[93m   -> Mẹo: Tích vào 'Always allow from this computer' và bấm 'Allow' trên màn hình kính!\033[0m\n")
                        last_warn_t = now

                # Trường hợp 3 (Đã kết nối - <serial>\tdevice):
                elif authorized_devices:
                    target_serial = authorized_devices[0]
                    # Kiểm tra lệnh adb reverse --list
                    res_rev = subprocess.run([adb_bin, "-s", target_serial, "reverse", "--list"], capture_output=True, text=True, timeout=4)
                    rev_out = res_rev.stdout

                    need_reverse = (not is_reversed) or any(f"tcp:{p}" not in rev_out for p in ports)

                    if need_reverse:
                        success = True
                        for p in ports:
                            r = subprocess.run([adb_bin, "-s", target_serial, "reverse", f"tcp:{p}", f"tcp:{p}"], capture_output=True, text=True, timeout=4)
                            if r.returncode != 0:
                                success = False

                        if success:
                            is_reversed = True
                            try:
                                from teleop_quest.network_utils import get_local_ip
                                local_ip = get_local_ip()
                            except Exception:
                                local_ip = "127.0.0.1"

                            print("\n\033[92;1m" + "=" * 70 + "\033[0m")
                            print(f"\033[92;1m✅ [QUEST USB] Đã kết nối cáp USB thành công! Mở trình duyệt trên Quest và vào ngay: https://localhost:8443 (hoặc qua Wi-Fi: https://{local_ip}:8443)\033[0m")
                            print("   👉 3D Viser Dashboard: http://localhost:8080")
                            print("\033[92;1m" + "=" * 70 + "\033[0m\n")

            except Exception:
                pass

            time.sleep(2.5)

    _adb_watcher_thread = threading.Thread(target=_adb_watcher, daemon=True, name="AutoHealADBWatcher")
    _adb_watcher_thread.start()
    return _adb_watcher_thread

def get_ursim_ip(default: str = "192.168.56.101") -> str:
    """Lấy địa chỉ IP mạng nội bộ của container Docker URSim"""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return default
    try:
        res = subprocess.run(
            [docker_bin, "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", "ursim"],
            capture_output=True, text=True, timeout=1.0
        )
        ip = res.stdout.strip()
        if ip:
            return ip
    except Exception:
        pass
    return default

def print_startup_banner(local_ip: str, quest_port: int = 8443, viser_port: int = 8080, ursim_port: int = 6080):
    """In thông tin địa chỉ truy cập hệ thống trực quan và có thể click trực tiếp"""
    ursim_ip = get_ursim_ip()
    print("\n" + "=" * 68)
    print("  🚀 HỆ THỐNG TELEOP QUEST ĐÃ KHỞI CHẠY THÀNH CÔNG!")
    print("=" * 68)
    print(f"  [1] Giao diện 3D Viser Dashboard (Máy tính):")
    print(f"      👉 http://localhost:{viser_port}")
    print()
    print(f"  [2] Giao diện WebXR Cockpit (Trên máy tính hoặc Quest cắm cáp USB):")
    print(f"      👉 https://localhost:{quest_port}")
    if local_ip not in ("127.0.0.1", "localhost"):
        print()
        print(f"  [3] Giao diện WebXR Cockpit (Kính Quest qua Wi-Fi mạng LAN):")
        print(f"      👉 https://{local_ip}:{quest_port}")
    print()
    print(f"  [4] PolyScope URSim Robot Simulator (noVNC):")
    print(f"      👉 http://localhost:{ursim_port}/vnc.html (hoặc http://{ursim_ip}:{ursim_port}/vnc.html)")
    print()
    print("  🔒 Cảnh báo SSL: Khi vào lần đầu, bấm 'Nâng cao / Advanced' -> 'Tiếp tục truy cập / Proceed'")
    print("  🔥 Tường lửa (Firewall): Nếu máy mới chặn cổng 8443/8080, hãy mở port:")
    print("      sudo ufw allow 8443/tcp 8080/tcp 8444/tcp 8445/tcp")
    print("=" * 68 + "\n")

