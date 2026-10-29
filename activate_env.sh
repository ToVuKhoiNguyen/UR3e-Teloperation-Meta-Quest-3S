#!/bin/bash
# ==============================================================================
# activate_env.sh: Kích hoạt môi trường UR3e Teleoperation trên máy mới
# Cách dùng:
#   1. source activate_env.sh   (Khuyến nghị: để gõ trực tiếp lệnh python ngoài terminal)
#   2. ./activate_env.sh        (Vào thẳng giao diện dòng lệnh bên trong Docker)
# ==============================================================================

# Xóa sạch alias cũ ở mức top-level để tránh xung đột ngữ pháp bash khi khai báo hàm
unalias python python3 pytest 2>/dev/null || true

# Xác định thư mục dự án (nơi chứa script)
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    SCRIPT_PATH="$0"
fi
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
IMAGE_NAME="ur3e_teleoperation:latest"
FALLBACK_IMAGE="teleop_quest:offline"

# Tự động tìm file tar tương ứng trong thư mục
TAR_FILE="$PROJECT_DIR/ur3e_teleoperation.tar"
if [ ! -f "$TAR_FILE" ] && [ -f "$PROJECT_DIR/ur33_teleoperation.tar" ]; then
    TAR_FILE="$PROJECT_DIR/ur33_teleoperation.tar"
elif [ ! -f "$TAR_FILE" ] && [ -f "$PROJECT_DIR/teleop_quest_offline.tar" ]; then
    TAR_FILE="$PROJECT_DIR/teleop_quest_offline.tar"
elif [ ! -f "$TAR_FILE" ]; then
    for f in "$PROJECT_DIR"/*.tar; do
        if [ -f "$f" ]; then
            TAR_FILE="$f"
            break
        fi
    done
fi

echo "======================================================================"
echo "  🚀 KÍCH HOẠT MÔI TRƯỜNG DOCKER UR3e TELEOPERATION"
echo "======================================================================"

# 1. Kiểm tra Docker daemon
if ! command -v docker &> /dev/null; then
    echo "❌ LỖI: Máy tính chưa cài đặt Docker! Vui lòng cài đặt Docker trước:"
    echo "   sudo apt update && sudo apt install -y docker.io docker-compose-v2"
    echo "   sudo usermod -aG docker \$USER && newgrp docker"
    return 1 2>/dev/null || exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "⚠️  CẢNH BÁO: Docker daemon chưa chạy hoặc chưa cấp quyền cho user hiện tại."
    echo "   Thử chạy: sudo systemctl start docker"
fi

# 2. Kiểm tra và tự động nạp Docker Image nếu chưa có
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1 && ! docker image inspect "$FALLBACK_IMAGE" > /dev/null 2>&1; then
    if [ -f "$TAR_FILE" ]; then
        echo "📦 Phát hiện lần đầu chạy trên máy mới. Đang nạp Docker Image từ file:"
        echo "   -> $TAR_FILE"
        echo "   (Quá trình này mất khoảng 5-10 giây)..."
        docker load -i "$TAR_FILE"
        # Đảm bảo gán cả 2 tag để tương thích tuyệt đối
        LOADED_IMG=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(teleop.?quest|ur3[3e]_teleoperation)" | head -n 1)
        if [ -n "$LOADED_IMG" ]; then
            docker tag "$LOADED_IMG" "$IMAGE_NAME" 2>/dev/null || true
            docker tag "$LOADED_IMG" "$FALLBACK_IMAGE" 2>/dev/null || true
        fi
        echo "✅ Nạp Docker Image thành công!"
    else
        echo "⚠️  Không tìm thấy file ảnh $TAR_FILE."
        echo "   Đang kiểm tra Docker image cục bộ..."
        if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
            echo "🔨 Đang tiến hành build image từ Dockerfile..."
            docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
            docker tag "$IMAGE_NAME" "$FALLBACK_IMAGE" 2>/dev/null || true
        fi
    fi
else
    # Đồng bộ cả 2 tag nếu 1 trong 2 đã có
    if docker image inspect "$FALLBACK_IMAGE" > /dev/null 2>&1 && ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        docker tag "$FALLBACK_IMAGE" "$IMAGE_NAME" 2>/dev/null || true
    fi
    echo "✅ Docker Image '$IMAGE_NAME' đã sẵn sàng!"
fi

# 1. Dọn dẹp môi trường, giải phóng cổng và chuẩn bị firewall trên máy Host
echo "🧹 Đang dọn dẹp các tiến trình cũ và giải phóng cổng kết nối..."
docker stop $(docker ps -q --filter ancestor=ur3e_teleoperation:latest --filter ancestor=ur33_teleoperation:latest --filter ancestor=teleop_quest:offline) 2>/dev/null || true
fuser -k 8080/tcp 8081/tcp 8443/tcp 8444/tcp 8445/tcp 2>/dev/null || true

if command -v ufw >/dev/null 2>&1; then
    sudo ufw allow 8443/tcp 8080/tcp 8444/tcp 8445/tcp 2>/dev/null || true
fi

# Tắt adb server trên máy Host để nhường quyền truy cập USB độc quyền cho Docker container
if command -v adb >/dev/null 2>&1; then
    adb kill-server 2>/dev/null || true
fi

# 3. Tạo các thư mục cần thiết và file cấu hình .env
mkdir -p "$PROJECT_DIR/certs"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/.adb_keys"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        echo "📝 Chưa có file .env, tự động tạo từ .env.example..."
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    else
        touch "$PROJECT_DIR/.env"
    fi
fi

# 4. Chuẩn bị cờ mount USB (nếu hệ thống có)
USB_MOUNT=""
if [ -d "/dev/bus/usb" ]; then
    USB_MOUNT="-v /dev/bus/usb:/dev/bus/usb"
fi

ENV_FILE_FLAG=""
if [ -f "$PROJECT_DIR/.env" ]; then
    ENV_FILE_FLAG="--env-file $PROJECT_DIR/.env"
fi

# Kiểm tra TTY để tránh lỗi 'input device is not a TTY' khi chạy trong script
INTERACTIVE_FLAG=""
if [ -t 0 ] && [ -t 1 ]; then
    INTERACTIVE_FLAG="-it"
fi

# Tự động mount mã nguồn nếu có thư mục code bên ngoài để phản ánh chỉnh sửa trực tiếp
CODE_MOUNTS=""
if [ -d "$PROJECT_DIR/teleop_quest" ]; then
    CODE_MOUNTS="-v $PROJECT_DIR/teleop_quest:/app/teleop_quest -v $PROJECT_DIR/core:/app/core -v $PROJECT_DIR/config:/app/config"
    [ -d "$PROJECT_DIR/teleop" ] && CODE_MOUNTS="$CODE_MOUNTS -v $PROJECT_DIR/teleop:/app/teleop"
    [ -d "$PROJECT_DIR/safety" ] && CODE_MOUNTS="$CODE_MOUNTS -v $PROJECT_DIR/safety:/app/safety"
    [ -d "$PROJECT_DIR/assets" ] && CODE_MOUNTS="$CODE_MOUNTS -v $PROJECT_DIR/assets:/app/assets"
    [ -f "$PROJECT_DIR/main.py" ] && CODE_MOUNTS="$CODE_MOUNTS -v $PROJECT_DIR/main.py:/app/main.py"
fi

# Lệnh cơ sở chạy docker với network host và đầy đủ volumes (dùng .adb_keys cục bộ để tránh lỗi quyền sở hữu)
DOCKER_CMD="docker run $INTERACTIVE_FLAG --rm --network host --privileged $ENV_FILE_FLAG $USB_MOUNT $CODE_MOUNTS -v $PROJECT_DIR/.adb_keys:/root/.android -v $PROJECT_DIR/certs:/app/certs -v $PROJECT_DIR/logs:/app/logs -v $PROJECT_DIR/.env:/app/.env $IMAGE_NAME"

# 5. Phân tách hành vi khi 'source activate_env.sh' và khi chạy trực tiếp './activate_env.sh'
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    # Khai báo hàm shell (hoạt động 100% tin cậy trên mọi subshell, script và interactive terminal)
    python() {
        $DOCKER_CMD python "$@"
    }
    python3() {
        $DOCKER_CMD python3 "$@"
    }
    pytest() {
        $DOCKER_CMD pytest "$@"
    }

    # Sử dụng function shell chuẩn (hỗ trợ hoàn hảo tham số "$@" và không bị xung đột syntax khi source lại)


    echo "======================================================================"
    echo "🎉 MÔI TRƯỜNG ĐÃ ĐƯỢC KÍCH HOẠT THÀNH CÔNG!"
    echo "👉 BƯỚC TIẾP THEO: Hãy gõ lệnh khởi chạy hệ thống (ví dụ):"
    echo "   python main.py --robot sim"
    echo ""
    echo "⚠️  LƯU Ý QUAN TRỌNG: Bạn PHẢI gõ lệnh 'python main.py ...' ở trên trước."
    echo "   Khi màn hình hiện thông báo 'HỆ THỐNG ĐÃ KHỞI CHẠY THÀNH CÔNG',"
    echo "   lúc đó bạn mới mở trình duyệt vào các đường link sau:"
    echo "   - 3D Viser Dashboard (Máy tính):    http://localhost:8080"
    echo "   - WebXR Cockpit (Máy tính / Cáp USB): https://localhost:8443"
    echo "   - WebXR Cockpit (Quest qua Wi-Fi):  https://<IP_LAN_MAY>:8443"
    echo "======================================================================"
else
    # Trường hợp chạy trực tiếp: ./activate_env.sh
    echo "💡 Bạn đang chạy trực tiếp './activate_env.sh'."
    echo "👉 Đang mở shell tương tác bên trong container (gõ 'exit' để thoát)..."
    echo "   (Mẹo: Dùng 'source activate_env.sh' để gõ trực tiếp lệnh ngoài host)."
    echo ""
    exec $DOCKER_CMD bash
fi
