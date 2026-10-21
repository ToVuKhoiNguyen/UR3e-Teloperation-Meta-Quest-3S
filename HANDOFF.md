# HANDOFF: BÁO CÁO BÀN GIAO VÀ ĐÁNH GIÁ HỆ THỐNG TELEOP UR3e

> **Ngày thực hiện:** 21/10/2026  
> **Nhánh:** `master` (HEAD: `d9a7288`)  
> **Môi trường:** Ubuntu Linux, Python 3.11 (`.venv`), Universal Robots UR3e (Docker URSim `127.0.0.1`), PyRoki IK (CPU JAX JIT).

---

## 1. Tổng quan hệ thống

- **Robot & Cổng giao tiếp:**
  - Robot UR3e mô phỏng chạy qua Docker URSim (`127.0.0.1`).
  - Cổng 29999: Dashboard Server (mở khóa Protective Stop, đóng safety popup, bật Play).
  - Cổng 30002: URScript Secondary Port (nhận lệnh `movej` tốc độ thấp khi Go Home).
  - Cổng 30003: RTDE Control Interface (điều khiển ServoJ thời gian thực 20Hz/500Hz).
  - Cổng 30004: RTDE Receive Interface (đọc góc khớp và trạng thái robot).
  - Cổng 6080: Giao diện PolyScope Teach Pendant (noVNC).
- **Bộ giải Động học (IK):**
  - PyRoki trên CPU với JAX JIT, `rest_cost` trọng số 0.02, khởi tạo seed chuẩn `jaxls.VarValues.make([q_var.with_value(seed_arr)])`.
  - Giới hạn khớp an toàn trong URDF: `wrist_1` [-2.0, 2.0], `wrist_2` [-2.0, 2.0], `wrist_3` [-12.56637, 12.56637].
- **Giao diện Viễn thao (Teleop):**
  - Meta Quest WebXR (cổng 8443-8445), cơ chế Mặt Đồng Hồ (Phải/Trái quay trục đế q0 + bù góc Z ngàm; Tiến/Lùi duỗi thu theo hướng q0; Cao/Thấp theo Z).
  - Hiệu chuẩn hướng Tiến: Nút B ghi nhận yaw của `head_quat` chiếu theo hướng bàn máy (`_desk_u_fwd`).
- **Camera & Ngàm:**
  - Intel RealSense D435/D455 gắn trên cổ tay (`d455_link`), hỗ trợ synthetic stream khi không có phần cứng thực.

---

## 2. Quy tắc làm việc bắt buộc

1. **Quy trình 2 bước nghiêm ngặt:** Bước 1 (Đọc, khảo sát, lập kế hoạch chi tiết, chờ cố vấn/người dùng phê duyệt) $\rightarrow$ Bước 2 (Chỉ thực thi khi đã được đồng ý 100%).
2. **Commit nguyên tử:** Mỗi thay đổi logic tương ứng một commit độc lập kèm thông điệp rõ ràng.
3. **Tuyệt đối KHÔNG git push** lên bất kỳ remote server nào.
4. **Tuyệt đối KHÔNG xóa/sửa các file dữ liệu thực:** `quest_run2.jsonl`, `quest_run3.jsonl`.
5. **Quy ước trục điều khiển tiếng Việt:**
   - **Tiến / Lùi:** Trục $+X$ / $-X$ (duỗi ra / thu về theo mặt phẳng ngang bàn máy).
   - **Phải / Trái:** Trục $-Y$ / $+Y$ (quay theo chiều kim đồng hồ / ngược chiều quanh đế $q_0$).
   - **Cao / Thấp:** Trục $+Z$ / $-Z$ (nâng lên / hạ xuống theo phương thẳng đứng).

---

## 3. Các hạng mục ĐÃ GIẢI QUYẾT (Đã xác minh bằng lệnh thực tế)

> *Ghi chú: Toàn bộ mục dưới đây đã được kiểm chứng bằng chạy lệnh và đo lường trực tiếp trên hệ thống thật / URSim localhost, không suy đoán.*

- [x] **Dọn sạch tàn dư CuRobo:** Đã loại bỏ hoàn toàn các phụ thuộc GPU CuRobo và file cấu hình cũ (`ur3e_custom.yml`), `python -m compileall` đạt 100% không có cảnh báo cú pháp, import tất cả module thành công.
- [x] **Đường ống điều khiển ServoJ (15cm):** Đã kiểm chứng đường ống đầy đủ (`WidgetPoseSource` $\rightarrow$ `PyrokiIKSolver` $\rightarrow$ `SafetyGate` $\rightarrow$ `ServoJExecutor`) trên cả 3 trục Tiến, Phải, Cao từ 1cm đến 15cm. Toàn bộ 15/15 bước đạt sai số IK 0.00mm, không vấp Singularities, không bị ngắt quãng.
- [x] **Go Home an toàn 2 pha:** Đã kiểm thử từ tư thế vươn xa ($q_1 \approx -1.5\text{ rad}$): thời gian hoàn thành 5.50s, chuyển động mượt mà (bước nhảy lớn nhất giữa 2 mẫu liên tiếp ở 20Hz là $\Delta q_{max} = 0.0250\text{ rad} = 1.43^\circ < 0.15\text{ rad}$), Viser 3D bám sát góc khớp thực (79 lần cập nhật), và cờ `cb_teleop` tự động bật lại sau khi về đích.
- [x] **Xác minh đáp ứng dải nhỏ & deadband:** Tại commit `a8ed756`, bước di chuyển tay 3mm bị nuốt hoàn toàn do deadband 3mm; tại HEAD, deadband 1.5mm phản hồi được 0.32mm ở bước 3mm. Đáp ứng Đẩy/Kéo đối xứng hoàn hảo.
- [x] **Độ ổn định khởi động (`main.py`):** Tự động phát hiện và bật Docker URSim khi cổng 29999 chưa mở; kiểm tra ADB reverse xử lý ngoại lệ trong 0.008s không gây treo sập khi không cắm kính Quest; fallback an toàn sang synthetic camera khi không có RealSense.
- [x] **Bảo toàn cấu hình an toàn & Giới hạn khớp:** File `config/singularity.yaml` duy trì nguyên vẹn ngưỡng cảnh báo 0.06 và nguy hiểm 0.035. Giới hạn các khớp `wrist_1`, `wrist_2`, `wrist_3` trong `assets/ur3e_custom.urdf` còn nguyên.

---

## 4. Các vấn đề CÒN MỞ (RỦI RO KỸ THUẬT CẦN XỬ LÝ)

### ⚠️ VẤN ĐỀ 1 (ƯU TIÊN CAO NHẤT): Lỗi hồi quy hướng Tiến do "Auto Head Yaw" (Commit `42c8bf9`)
- **Hiện tượng:** Tại mỗi lần bóp cò, đoạn code dòng 301-315 trong `teleop_quest/delta_pose_source.py` tự động lấy quaternion đầu kính hiện tại (`head_quat`) chiếu xuống mặt phẳng ngang và **GHI ĐÈ** lên hướng tham chiếu `_anchor_u_fwd`.
- **Hậu quả:** Nút B hiệu chuẩn bàn máy bị vô hiệu hóa hoàn toàn khi có kính. Khi người vận hành quay đầu sang bên $-43^\circ$ nhưng tay vẫn cố ý đẩy thẳng về hướng bàn máy (dữ liệu kiểm chứng thực tế Quest run #21-23), hệ thống đọc nhầm thành lệnh rẽ Phải tới **+4.0cm** (lớn hơn cả thành phần Tiến +2.4cm, vượt xa ngưỡng cho phép $|Phải| < 1.5\text{cm}$).
- **Hướng khắc phục đề xuất:** Ưu tiên giữ nguyên hướng bàn máy `_desk_u_fwd` đã được hiệu chuẩn bằng nút B; chỉ cập nhật theo đầu khi người dùng chủ động chọn chế độ bám đầu hoặc bổ sung biến môi trường cấu hình `HEADING_MODE=desk|head`.

### ⚠️ VẤN ĐỀ 2 (ƯU TIÊN CAO): Rào cản sàn `z_floor=0.0` chặn ngàm gắp trên mặt bàn
- **Hiện tượng:** Trong `teleop/pyroki_solver.py` (`table_barrier_residual`), biến `z_floor` được gán cứng bằng `0.0` (mặt phẳng đế robot), trong khi mặt bàn máy nằm ở $Z = -0.04\text{m}$.
- **Hậu quả:** Cụm camera trên cổ tay có khoảng đệm an toàn 4cm + bán kính 3.5cm = 7.5cm. Khi hạ thấp tay, rào cản sàn chặn cứng camera tại $Z = +0.075\text{m}$, kéo theo đầu TCP dừng lại ở $Z = +0.086\text{m}$ (cách mặt bàn thực tế tới **12.6cm**). Mọi lệnh hạ ngàm xuống sát bàn máy để gắp vật đều bị IK từ chối (`pos_err_mm > 5mm`, trả về `None`).
- **Hướng khắc phục đề xuất:** Điều chỉnh lại `z_floor` xuống dưới mặt đáy bàn (ví dụ $Z = -0.5\text{m}$) hoặc chỉ áp dụng rào cản mặt bàn `z_table_top` trong phạm vi hình chiếu XY của bàn.

### ⚠️ VẤN ĐỀ 3: 4 file sửa đổi UI/VR chưa commit
- Các file `teleop_quest/quest_server.py`, `teleop_quest/static/index.html`, `teleop_quest/static/main.js`, `teleop_quest/static/style.css` đang chứa các chỉnh sửa bỏ nút VR Go Home và chuyển đổi giao diện cock-pit chưa được commit. Cần rà soát và tạo commit quản lý riêng.

---

## 5. Lệnh vận hành hay dùng

```bash
# 1. Kích hoạt môi trường và khởi động hệ thống teleop với URSim
cd ~/ur_ws/pyroki_ursim
.venv/bin/python main.py --sim-only

# 2. Đưa robot về vị trí Home an toàn chuẩn
.venv/bin/python reset_home.py

# 3. Mở khóa nhanh Protective Stop từ terminal (cổng 29999)
python3 -c "import socket, time; s=socket.socket(); s.connect(('127.0.0.1', 29999)); s.sendall(b'unlock protective stop\nclose safety popup\nplay\n'); time.sleep(0.5); s.close()"

# 4. Kiểm tra giới hạn khớp UR3e trong URDF
python3 -c "import re; urdf=open('assets/ur3e_custom.urdf').read(); print(re.findall(r'<joint name=\"(wrist_[123]_joint)\"[^>]*>.*?<limit[^>]*lower=\"([^\"]+)\"[^>]*upper=\"([^\"]+)\"', urdf, re.DOTALL))"
```

---

## 6. Nhật ký kiểm tra (Verification Log)

- **21/10/2026 10:35:** Hoàn thành đối chiếu dữ liệu `quest_run2.jsonl` giữa commit `a8ed756` và `HEAD`. Phát hiện lỗi hồi quy nghiêm trọng của hướng Tiến tại nhóm #21-23.
- **21/10/2026 10:37:** Hoàn thành kiểm thử đường ống ServoJ 1-15cm và Go Home 2 pha trên URSim 127.0.0.1. Mọi tham số thời gian và độ mịn chuyển động đều đạt chuẩn.
- **21/10/2026 10:39:** Đo đạc chính xác rào cản bàn/sàn trên bộ giải PyRoki JIT solver. Xác định rõ nguyên nhân TCP bị chặn ở Z=8.6cm (cách bàn 12.6cm).
