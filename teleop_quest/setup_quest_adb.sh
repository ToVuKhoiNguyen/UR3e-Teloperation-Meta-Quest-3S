#!/bin/bash
# ==============================================================================
# Script tự động kết nối Meta Quest qua ADB Developer Mode (Cáp USB hoặc Wi-Fi)
# Hỗ trợ chuyển tiếp toàn bộ cổng WebXR, WebSocket và Viser vào localhost
# ==============================================================================

echo "=========================================================="
echo "    THIET LAP KET NOI META QUEST DEVELOPER MODE (ADB)"
echo "=========================================================="

# 1. Kiểm tra adb đã cài đặt chưa
if ! command -v adb &> /dev/null; then
    echo "[!] adb chua duoc cai dat tren he thong."
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update -qq && apt-get install -y --no-install-recommends adb
    elif command -v sudo &> /dev/null; then
        sudo apt update && sudo apt install -y adb
    else
        echo "[!] Khong the cai dat adb do khong co quyen root/sudo. Bo qua ADB."
    fi
fi


# 2. Khoi dong adb server
adb start-server

# 3. Kiem tra thiet bi Quest
echo "[*] Dang tim kiem kinh Meta Quest ket noi qua USB..."
AUTH_DEVICES=$(adb devices | grep -w "device" | awk '{print $1}')
UNAUTH_DEVICES=$(adb devices | grep -w "unauthorized" | awk '{print $1}')

if [ -n "$UNAUTH_DEVICES" ]; then
    echo "[!] PHAT HIEN KINH QUEST ($UNAUTH_DEVICES) NHUNG CHUA CAP QUYEN (unauthorized)!"
    echo "    >>> HANH DONG CAN LAM: <<<"
    echo "    1. Deo kinh Meta Quest len mat."
    echo "    2. Man hinh kinh se hien hop thoai: 'Allow USB Debugging?' (Cho phep go loi USB?)."
    echo "    3. Tich vao 'Always allow from this computer' va bam 'ALLOW' (CHO PHEP)."
    exit 2
fi

if [ -z "$AUTH_DEVICES" ]; then
    echo "[!] Khong tim thay kinh Meta Quest nao duoc ket noi qua USB!"
    echo "    Vui long kiem tra:"
    echo "    1. Cap USB-C da cam chac chan giua kinh va may tinh."
    echo "    2. Developer Mode da bat tren app Meta Horizon."
    exit 1
fi

echo "[+] Da tim thay thiet bi Quest da cap quyen: $AUTH_DEVICES"

# 4. Thiet lap ADB Reverse Port Forwarding
echo "[*] Dang thiet lap ADB Reverse Port Forwarding..."
REV_ERR=0
adb reverse tcp:8443 tcp:8443 || REV_ERR=1
adb reverse tcp:8444 tcp:8444 || REV_ERR=1
adb reverse tcp:8445 tcp:8445 || REV_ERR=1
adb reverse tcp:8080 tcp:8080 || REV_ERR=1

if [ $REV_ERR -ne 0 ]; then
    echo "[!] Loi khi thiet lap ADB Reverse Port Forwarding."
    exit 1
fi

echo ""
echo "=========================================================="
echo "    KET NOI THANH CONG! HUONG DAN SU DUNG:"
echo "=========================================================="
echo "1. Tren kinh Meta Quest, mo Meta Quest Browser."
echo "2. Truy cap vao dia chi:"
echo "   https://localhost:8443"
echo "   (Khong can quan tam IP LAN, khong lo song Wi-Fi chap chon!)"
echo ""
echo "3. De xem man hinh va Debug truc tiep tu PC:"
echo "   Mo Chrome tren Ubuntu -> vao chrome://inspect/#devices -> bam 'inspect'"
echo "=========================================================="
