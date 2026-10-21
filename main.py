#!/usr/bin/env python3
import sys
import os
import shutil
import subprocess

def setup_adb_reverse(ports=(8443, 8444, 8445, 8080)):
    """
    Tự động kiểm tra kính Meta Quest qua ADB và cấu hình reverse port forwarding.
    Khắc phục triệt để lỗi 'This site can't be reached' khi truy cập https://localhost:8443 trên kính.
    """
    adb_bin = shutil.which("adb")
    if not adb_bin:
        print("[ADB] Lưu ý: 'adb' chưa được cài đặt trên hệ thống (sudo apt install -y adb). Bỏ qua ADB reverse.")
        return False

    try:
        # Khởi động adb server nếu chưa chạy
        subprocess.run([adb_bin, "start-server"], capture_output=True, timeout=3)

        # Kiểm tra danh sách thiết bị
        res = subprocess.run([adb_bin, "devices"], capture_output=True, text=True, timeout=3)
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]

        # Bỏ qua dòng tiêu đề "List of devices attached" và các dòng thông báo daemon
        device_lines = [l for l in lines if not l.startswith("List of devices") and not l.startswith("*")]

        authorized_devices = []
        unauthorized_devices = []

        for dl in device_lines:
            parts = dl.split()
            if len(parts) >= 2:
                serial, state = parts[0], parts[1]
                if state == "device":
                    authorized_devices.append(serial)
                elif state == "unauthorized":
                    unauthorized_devices.append(serial)

        if unauthorized_devices:
            print("\n" + "!" * 65)
            print(f"[ADB CẢNH BÁO] Phát hiện kính Quest ({', '.join(unauthorized_devices)}) nhưng CHƯA CẤP QUYỀN (unauthorized)!")
            print("-> Hãy đeo kính Meta Quest lên mắt.")
            print("-> Chọn 'Always allow from this computer' và bấm 'Allow' (Cho phép) trên kính!")
            print("!" * 65 + "\n")
            return False

        if not authorized_devices:
            print("[ADB] Không phát hiện kính Quest cắm cáp USB (chế độ Wi-Fi hoặc chưa cắm cáp).")
            return False

        # Thực thi reverse port forwarding cho từng cổng
        success_ports = []
        failed_ports = []
        for port in ports:
            rev_res = subprocess.run(
                [adb_bin, "reverse", f"tcp:{port}", f"tcp:{port}"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if rev_res.returncode == 0:
                success_ports.append(port)
            else:
                failed_ports.append(port)

        if success_ports:
            print("\n" + "=" * 65)
            print("[ADB USB] ĐÃ TỰ ĐỘNG THIẾT LẬP REVERSE PORT FORWARDING THÀNH CÔNG!")
            print(f"-> Thiết bị: {', '.join(authorized_devices)}")
            print(f"-> Các cổng đã forward: {', '.join(str(p) for p in success_ports)}")
            print("-> Hãy mở Meta Quest Browser trên kính và truy cập: https://localhost:8443")
            print("=" * 65 + "\n")
            return True
        else:
            print(f"[ADB] Lỗi khi forward các cổng: {failed_ports}")
            return False

    except Exception as e:
        print(f"[ADB] Lỗi trong quá trình kiểm tra/cấu hình ADB: {e}")
        return False

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

    from teleop_quest.main_quest import main
    main()

