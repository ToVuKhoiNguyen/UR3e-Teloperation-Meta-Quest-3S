#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$PROJECT_DIR/dist_package"
ZIP_NAME="ur3e_teleoperation.zip"
IMAGE_TAR="ur3e_teleoperation.tar"
IMAGE_NAME="ur3e_teleoperation:latest"
FALLBACK_IMAGE="teleop_quest:offline"

echo "=========================================================="
echo "🚀 BẮT ĐẦU ĐÓNG GÓI DỰ ÁN UR3e TELEOPERATION (OFFLINE)"
echo "=========================================================="

# 1. Kiểm tra Docker Image, tự động build nếu chưa có
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "🔨 Chưa tìm thấy Docker Image $IMAGE_NAME. Đang tiến hành build..."
    docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
    docker tag "$IMAGE_NAME" "$FALLBACK_IMAGE" 2>/dev/null || true
else
    echo "✅ Đã tìm thấy Docker Image: $IMAGE_NAME"
    docker tag "$IMAGE_NAME" "$FALLBACK_IMAGE" 2>/dev/null || true
fi

# 2. Đồng bộ các file mã nguồn và tài liệu mới nhất vào dist_package
echo "📂 Đồng bộ mã nguồn và tài liệu sang dist_package..."
mkdir -p "$DIST_DIR"
cp -v "$PROJECT_DIR/activate_env.sh" "$DIST_DIR/"
cp -v "$PROJECT_DIR/Dockerfile" "$DIST_DIR/"
cp -v "$PROJECT_DIR/main.py" "$DIST_DIR/"
cp -v "$PROJECT_DIR/README.md" "$DIST_DIR/"
cp -v "$PROJECT_DIR/HUONG_DAN_CHAY.md" "$DIST_DIR/"
cp -v "$PROJECT_DIR/.env.example" "$DIST_DIR/"
[ -f "$PROJECT_DIR/.env" ] && cp -v "$PROJECT_DIR/.env" "$DIST_DIR/"
cp -v "$PROJECT_DIR/.dockerignore" "$DIST_DIR/"
cp -v "$PROJECT_DIR/docker-compose.yml" "$DIST_DIR/"
cp -v "$PROJECT_DIR/requirements.lock.txt" "$DIST_DIR/"
cp -v "$PROJECT_DIR/requirements.txt" "$DIST_DIR/"

cp -r "$PROJECT_DIR/assets" "$DIST_DIR/" 2>/dev/null || true
cp -r "$PROJECT_DIR/certs" "$DIST_DIR/" 2>/dev/null || true
cp -r "$PROJECT_DIR/config" "$DIST_DIR/" 2>/dev/null || true
cp -r "$PROJECT_DIR/core" "$DIST_DIR/" 2>/dev/null || true
cp -r "$PROJECT_DIR/safety" "$DIST_DIR/" 2>/dev/null || true
cp -r "$PROJECT_DIR/teleop" "$DIST_DIR/" 2>/dev/null || true
cp -r "$PROJECT_DIR/teleop_quest" "$DIST_DIR/" 2>/dev/null || true
mkdir -p "$DIST_DIR/logs"
mkdir -p "$DIST_DIR/.adb_keys"

# Dọn dẹp cache rác
find "$DIST_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DIST_DIR" -name "*.pyc" -delete 2>/dev/null || true

# 3. Xuất Docker Image thành file .tar để người nhận không cần tải mạng
echo "📦 Đang xuất Docker Image thành file $IMAGE_TAR (sẵn sàng chạy offline)..."
docker save "$IMAGE_NAME" -o "$DIST_DIR/$IMAGE_TAR"
echo "✅ Đã xuất xong $IMAGE_TAR ($(du -h "$DIST_DIR/$IMAGE_TAR" | cut -f1))"

# 4. Xóa file zip cũ nếu tồn tại
if [ -f "$PROJECT_DIR/$ZIP_NAME" ]; then
    echo "🗑️  Đang xóa bản build zip cũ..."
    rm -f "$PROJECT_DIR/$ZIP_NAME"
fi

# 5. Thực thi nén toàn bộ gói dist_package
echo "🗜️  Đang nén toàn bộ gói phân phối thành $ZIP_NAME..."
(cd "$DIST_DIR" && zip -r "$PROJECT_DIR/$ZIP_NAME" . \
    -x "*.venv*" "*venv*" "*.git*" "*__pycache__*" "*.log" "*.pyc" ".adb_keys/*")

echo ""
echo "=========================================================="
echo "🎉 HOÀN TẤT ĐÓNG GÓI THÀNH CÔNG!"
echo "📁 Thư mục đóng gói: dist_package/ ($(du -sh "$DIST_DIR" | cut -f1))"
echo "📁 File nén đầu ra: $ZIP_NAME ($(du -h "$PROJECT_DIR/$ZIP_NAME" | cut -f1))"
echo "=========================================================="
