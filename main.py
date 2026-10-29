#!/usr/bin/env python3
import sys
import os
import shutil
import subprocess

import threading
import time

_adb_watcher_thread = None

def setup_adb_reverse(ports=(8443, 8444, 8445, 8080)):
    """
    Khởi động luồng 'Auto-Heal ADB USB' chạy ngầm liên tục (daemon=True) lặp lại mỗi 2.5 giây.
    Khắc phục triệt để lỗi 'This site can't be reached' kể cả khi người dùng cấp quyền muộn hoặc tuột cáp cắm lại.
    """
    global _adb_watcher_thread
    if _adb_watcher_thread is not None and _adb_watcher_thread.is_alive():
        return _adb_watcher_thread

    def _adb_worker():
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

    _adb_watcher_thread = threading.Thread(target=_adb_worker, daemon=True, name="AutoHealADBWatcher")
    _adb_watcher_thread.start()
    return _adb_watcher_thread

if __name__ == "__main__":
    # Ensure the workspace root is in PYTHONPATH
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Tự động cấu hình ADB Reverse Port Forwarding cho cáp USB Quest
    setup_adb_reverse()

    # Truyền cờ --record <file> xuống quest_server
    if "--record" in sys.argv:
        try:
            idx = sys.argv.index("--record")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                record_file = sys.argv[idx + 1]
                os.environ["QUEST_RECORD_FILE"] = record_file
                print(f"[Recorder] Bật chế độ ghi log JSONL: {record_file}")
        except Exception:
            pass

    # Hỗ trợ cờ --disable-self-collision để bypass kiểm tra tự va chạm khi test
    if "--disable-self-collision" in sys.argv:
        os.environ["DISABLE_SELF_COLLISION"] = "1"
        print("[An Toàn] Cờ --disable-self-collision được kích hoạt: TẮT kiểm tra tự va chạm!")

    from teleop_quest.main_quest import main
    main()

