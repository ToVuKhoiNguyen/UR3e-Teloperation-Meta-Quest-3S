# HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG UR3e TELEOPERATION (DOCKER)

Hệ thống đã được đóng gói sẵn 100% môi trường và thư viện trong file ảnh Docker. Người dùng trên máy tính mới **KHÔNG CẦN CÀI THÊM BẤT KỲ THƯ VIỆN NÀO** (không cần pip, conda hay venv).

---

## BƯỚC 1: KÍCH HOẠT MÔI TRƯỜNG (CHỈ GÕ 1 LỆNH DUY NHẤT)

Mở Terminal tại thư mục dự án và chạy:
```bash
source activate_env.sh
```
*(Nếu là lần đầu chạy trên máy mới, script sẽ tự động nạp image `ur3e_teleoperation.tar` trong vài giây và thiết lập alias lệnh `python`).*

> **Mẹo:** Nếu bạn muốn vào thẳng giao diện dòng lệnh bên trong container, chỉ cần gõ `./activate_env.sh`.

---

## BƯỚC 2: CHẠY CHƯƠNG TRÌNH (GÕ LỆNH `python` Y HỆT MÁY THẬT)

Sau khi chạy lệnh ở Bước 1, bạn có thể copy-paste trực tiếp bất kỳ câu lệnh nào trong `README.md`:

```bash
# 1. Chạy mô phỏng 3D (URSim + Camera ảo - Khuyên dùng để test thử ban đầu):
python main.py --robot sim

# 2. Chạy với Robot UR3e thật (Tự động quét tìm IP qua mạng LAN):
python main.py --auto

# 3. Chạy với Robot UR3e thật (IP cố định):
python main.py --robot real --robot-ip 10.42.0.242

# 4. Chạy CẢ HAI: Robot UR3e thật + Camera Intel RealSense cắm cáp USB:
python main.py --robot real --robot-ip 10.42.0.242 --cam-source usb --camera realsense

# 5. Chạy CẢ HAI: Robot UR3e thật + Camera qua luồng WebSocket:
python main.py --robot real --robot-ip 10.42.0.242 --cam-source ws --cam-url ws://127.0.0.1:8765

# 6. Chạy các script công cụ xử lý dữ liệu HDF5:
python core/visualize_data.py --file logs/ai_training_data/episode_0.hdf5
python core/extract_hdf5.py --file logs/ai_training_data/episode_0.hdf5
```

---

## BƯỚC 3: MỞ GIAO DIỆN TRỰC QUAN (LOCALHOST & VR)

Ngay sau khi chương trình chạy, Terminal sẽ in sẵn các đường link để bạn bấm mở trực tiếp (Ctrl + Click):

1. **Trên trình duyệt máy tính:**
   - **Giao diện 3D Viser Dashboard:** [http://localhost:8080](http://localhost:8080) (Điều khiển bằng chuột qua Transform Gizmo, bật/tắt Teleop, ghi dữ liệu HDF5).
   - **Giao diện WebXR Cockpit:** [https://localhost:8443](https://localhost:8443)

2. **Trên kính thực tế ảo Meta Quest (Khuyên dùng):**
   - **Cách 1 - Cắm cáp USB (Tốc độ cao & Ổn định nhất):**
     + Cắm cáp USB Type-C giữa kính Meta Quest và máy tính.
     + Nếu Terminal xuất hiện dòng cảnh báo màu vàng:
       `⚠️  [QUEST USB] Phát hiện kính Quest nhưng CHƯA CẤP QUYỀN! Hãy đeo kính lên và bấm 'Allow USB Debugging'!`
       -> Hãy đeo kính lên mắt, tích chọn **"Always allow from this computer"** rồi bấm **"Allow" (Cho phép)**.
     + Chờ dòng thông báo màu xanh lá hiện ra trên Terminal:
       `✅ [QUEST USB] Đã kết nối cáp USB thành công! Mở trình duyệt trên Quest và vào ngay: https://localhost:8443`
     + Mở Meta Quest Browser trong kính và truy cập: **`https://localhost:8443`** (vào được ngay lập tức).
   - **Cách 2 - Dùng Wi-Fi cùng mạng LAN với máy tính:**
     + Mở Meta Quest Browser trong kính và truy cập: **`https://<IP_LAN_in_trên_terminal>:8443`** (ví dụ: `https://192.168.1.15:8443`).

3. **Cách vượt qua cảnh báo SSL tự ký trên trình duyệt (Chỉ cần làm 1 lần):**
   - Do chứng chỉ SSL được hệ thống tự ký an toàn để bảo mật WebXR:
   - Khi trình duyệt hiện cảnh báo *"Your connection is not private"* (Kết nối không riêng tư):
   - Bạn chỉ cần bấm vào **Advanced (Nâng cao)** -> Chọn **Proceed to localhost (Tiếp tục truy cập)**.
   - Khi vào giao diện Cockpit, bấm nút **"Enter VR"** màu xanh để hòa mình vào không gian điều khiển robot!

---

## BƯỚC 4: TÙY CHỈNH CẤU HÌNH TRONG FILE `.env` (NẾU CẦN)

Nếu môi trường mạng của bạn khác với mặc định, hãy mở file `.env` và sửa:
```bash
ROBOT_IP=10.42.0.242           # Địa chỉ IP của cánh tay robot UR3e
CAM_URL=tcp://10.42.0.100:5555  # URL nhận video camera qua mạng
```

---

## BƯỚC 5: TẮT CHƯƠNG TRÌNH

- Nhấn tổ hợp phím `Ctrl + C` trên terminal đang chạy để dừng chương trình an toàn.
- Nếu chạy qua Docker Compose nền: `docker compose down`.

---

## ❓ XỬ LÝ SỰ CỐ: TẠI SAO KHÔNG MỞ ĐƯỢC LOCALHOST?

Nếu bạn mở `http://localhost:8080` hoặc `https://localhost:8443` mà trình duyệt báo *"Không thể kết nối"* (Unable to connect / This site can’t be reached), hãy kiểm tra 3 nguyên nhân sau:

### 1. Bạn đã gõ lệnh chạy chương trình ở Bước 2 chưa?
- Lệnh `source activate_env.sh` (Bước 1) **chỉ kích hoạt môi trường**, chưa khởi động web server!
- Bạn **bắt buộc phải gõ tiếp** lệnh chạy ở Bước 2:
  ```bash
  python main.py --robot sim
  ```
- **Chờ khoảng 5-10 giây** cho PyRoki nạp xong mô hình và Terminal in ra bảng thông báo:
  `🚀 HỆ THỐNG TELEOP QUEST ĐÃ KHỞI CHẠY THÀNH CÔNG!`
  Lúc này các cổng web mới chính thức mở để bạn truy cập.

### 2. Cổng mạng bị tiến trình cũ / Docker ngầm chiếm dụng:
- Nếu trước đó bạn đã chạy một lần và tắt chưa sạch, hoặc có container Docker chạy ngầm, hãy chạy 1 dòng lệnh sau để dọn dẹp toàn bộ cổng:
  ```bash
  # Tắt mọi container ngầm và giải phóng các cổng 8080, 8443
  docker stop $(docker ps -q --filter ancestor=ur3e_teleoperation:latest --filter ancestor=ur33_teleoperation:latest --filter ancestor=teleop_quest:offline) 2>/dev/null || true
  fuser -k 8080/tcp 8081/tcp 8443/tcp 8444/tcp 8445/tcp 2>/dev/null || true
  ```
- Sau đó chạy lại: `python main.py --robot sim`.

### 3. Trình duyệt chặn cảnh báo SSL khi vào `https://localhost:8443`:
- Giao diện WebXR bắt buộc chạy qua giao thức bảo mật `HTTPS`.
- Khi Chrome/Firefox/Edge hiện trang cảnh báo màu đỏ *"Kết nối của bạn không phải là kết nối riêng tư"* (Your connection is not private):
  + Bấm vào nút **Nâng cao (Advanced)** ở góc dưới.
  + Bấm vào dòng link **Tiếp tục truy cập localhost (không an toàn)** / *Proceed to localhost*.
  + Trình duyệt sẽ mở ngay giao diện buồng lái Cockpit 3D.

### 4. Tường lửa Ubuntu chặn cổng (Connection Timeout trên 8443 / 8080):
- Nếu máy tính mới bật tường lửa `ufw`, các kết nối vào cổng 8443 sẽ bị timeout. Mở cổng bằng lệnh:
  ```bash
  sudo ufw allow 8443/tcp 8080/tcp 8444/tcp 8445/tcp
  ```

### 5. Kính Meta Quest báo "This site can't be reached" qua cáp USB:
- Hệ thống đã tích hợp sẵn cơ chế **Auto-Heal ADB USB Daemon** chạy ngầm liên tục mỗi 2.5 giây.
- Bạn chỉ cần nhìn vào màn hình Terminal:
  + Nếu thấy thông báo màu vàng:
    `⚠️  [QUEST USB] Phát hiện kính Quest nhưng CHƯA CẤP QUYỀN! Hãy đeo kính lên và bấm 'Allow USB Debugging'!`
    -> Đeo kính lên mắt, chọn *"Always allow from this computer"* rồi bấm **"Allow" (Cho phép)**.
  + Ngay khi bấm Allow (hoặc khi cắm lại dây nếu bị tuột), sau khoảng 2 giây màn hình sẽ hiện dòng chữ màu xanh lá:
    `✅ [QUEST USB] Đã kết nối cáp USB thành công! Mở trình duyệt trên Quest và vào ngay: https://localhost:8443`
  + Lúc này bạn chỉ cần bấm Tải lại (Reload) trang `https://localhost:8443` trong Meta Quest Browser là kết nối thành công ngay lập tức mà không cần tắt hay chạy lại code!

### 6. Mở được Viser 8080 nhưng kéo Gizmo robot thật (10.42.0.242) không di chuyển:
- **Nguyên nhân 1:** Máy tính chưa được gán IP tĩnh cùng dải `10.42.0.x` với robot.
  Kiểm tra bằng lệnh:
  ```bash
  ping -c 2 10.42.0.242
  ```
  Nếu không ping được, gán IP cho card mạng nối với robot:
  ```bash
  # Tìm tên card mạng nối với robot (ví dụ enp3s0 hoặc eth0)
  ip link
  # Gán IP 10.42.0.1/24 cho card mạng đó:
  sudo ip addr add 10.42.0.1/24 dev <tên_card_mạng>
  ```
- **Nguyên nhân 2:** Robot UR3e chưa bật nguồn hoặc chưa chuyển sang Remote Control.
  + Bật nguồn và nhả phanh trên Teach Pendant (trạng thái RUNNING màu xanh).
  + Chọn biểu tượng Remote ở góc trên bên phải màn hình PolyScope của robot.

