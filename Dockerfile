# ==============================================================================
# Dockerfile: Hệ thống Teleop Quest cho Robot UR3e (Offline Zero-Setup Deployment)
# Tối ưu hóa CPU (PyRoki IK Solver + JAX + Torch CPU)
# ==============================================================================

FROM python:3.11-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt các gói hệ thống bắt buộc:
# - build-essential, git: Compile C++ extensions và cài repo git (pyroki, jaxls)
# - libgl1, libglib2.0-0: Thư viện runtime bắt buộc cho OpenCV
# - libusb-1.0-0: Thư viện cho Intel RealSense SDK
# - openssl: Tạo chứng chỉ SSL tự động cho WebXR HTTPS (cổng 8443)
# - iproute2, iputils-ping: Tiện ích mạng cho RTDE và Camera ZeroMQ
# - psmisc: Lệnh fuser để giải phóng port mạng khi khởi động lại server
# - adb: Kết nối USB reverse port forwarding cho kính Meta Quest
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    openssl \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    iproute2 \
    iputils-ping \
    psmisc \
    adb \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Tận dụng Docker layer caching: cài đặt dependencies từ requirements.lock.txt
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock.txt

# Copy toàn bộ mã nguồn của dự án (đã lọc qua .dockerignore)
COPY . .

# Khai báo các port giao tiếp phần cứng và giao diện mạng:
# - 8080: Viser 3D UI & Digital Twin
# - 8443: WebXR Cockpit HTTPS Web Server
# - 8444: WebXR Secure WebSocket (WSS Telemetry/Control)
# - 8445: WebXR Plain WebSocket (WS Telemetry)
# - 8765: WebSocket Camera Server
# - 30004: Universal Robots RTDE Real-time Control (125Hz)
# - 30002: Universal Robots Primary Interface
# - 5555: RealSense Camera ZeroMQ PUB/SUB Video Stream
EXPOSE 8080 8443 8444 8445 8765 30004 30002 5555

# Lệnh khởi chạy mặc định (được ghi đè linh hoạt)
CMD ["python", "main.py", "--robot=sim", "--camera=synthetic"]
