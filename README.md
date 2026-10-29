# UR3e Điều Khiển Teleoperation Thời Gian Thực qua Meta Quest 3S, Viser 3D & PyRoki CPU IK

Hệ thống điều khiển viễn thao thời gian thực (Real-time Teleoperation), hiển thị không gian 3D và thu thập dữ liệu trình diễn đa phương thức cho cánh tay robot công nghiệp Universal Robots UR3e sử dụng kính thực tế ảo Meta Quest 3S (WebXR) và trình duyệt Web máy tính qua giao diện Viser 3D WebGL. Hệ thống tích hợp bộ giải bài toán động học nghịch (Inverse Kinematics - IK) siêu tốc thuần CPU dựa trên Google PyRoki, JAX và `jaxls` (độ trễ ~1ms sau JIT warm-up, hoàn toàn không cần card đồ họa rời NVIDIA/CUDA). Ngoài ra, hệ thống sở hữu module **Selectable Multimodal Data Recorder** (Ghi HDF5 dạng luồng phân mảnh Chunked Streaming với bộ đệm RingBuffer chống tràn RAM và đồng bộ thời gian phần hardware timestamp $< 1.5$ms) tối ưu cho huấn luyện Robot AI (Imitation Learning, Diffusion Policy, ACT), cùng bộ công cụ giải nén dữ liệu tự động **HDF5 Extraction Suite**.

> [!IMPORTANT]
> ### ⚡ KHỞI TẠO NHANH TRÊN MÁY MỚI (CHỈ LÀM 1 BƯỚC DUY NHẤT - 100% OFFLINE)
> Người dùng trên máy tính mới **KHÔNG CẦN CÀI THÊM BẤT KỲ THƯ VIỆN NÀO** (không cần pip, venv hay conda). Chỉ cần mở Terminal tại thư mục dự án và gõ:
> ```bash
> source activate_env.sh
> ```
> *(Lệnh trên sẽ tự động nạp file ảnh Docker Offline `teleop_quest_offline.tar` trong vài giây và thiết lập alias lệnh `python`. Sau đó bạn chỉ việc copy-paste ĐÚNG Y HỆT các câu lệnh `python ...` bên dưới để chạy ngay! Chi tiết xem tại [`HUONG_DAN_CHAY.md`](file:///home/nguyen/ur_ws/pyroki_ursim/HUONG_DAN_CHAY.md)).*

---


## 1. Tính Năng Nổi Bật

1. **Điều Khiển Viễn Thao Thời Gian Thực (20Hz RTDE) & Cơ Chế Mặt Đồng Hồ (Clock Face):**
   - Đồng bộ hóa mục tiêu 6-DoF từ tay cầm Meta Quest 3S hoặc chuột máy tính (Viser 3D Transform Controls Target Widget) sang robot UR3e ở tần số 20Hz với độ trễ phản hồi dưới 10ms.
   - Bộ thực thi `ServoJExecutor` sử dụng tham số điều khiển tối ưu chuẩn công nghiệp: `dt = 0.05`s, `lookahead_time = 0.12`s, `gain = 250`.
   - **Scaled Vector Rate Limiter:** Tự động co giãn tỷ lệ đồng bộ vận tốc cả 6 khớp khi tiếp cận ngưỡng giới hạn, bảo toàn 100% hướng vector quỹ đạo trong không gian Descartes và triệt tiêu hoàn toàn hiện tượng méo quỹ đạo/giật cơ học.
   - **Cơ chế Mặt Đồng Hồ (True Mechanical Clock Navigation):**
     - **Phải / Trái:** Quay kim đồng hồ (khớp đế $q_0$) quanh trục Base Z đồng thời tự động bù góc xoay trục Z ngàm kẹp giữ nguyên hướng đối với bàn máy.
     - **Tiến / Lùi:** Duỗi dài hoặc thu ngắn bắp tay theo đúng phương vươn cơ học $\vec{u}_{reach} = [\cos(q_0), \sin(q_0), 0.0]$.
     - **Cao / Thấp:** Nâng hạ độ cao Z thẳng đứng trên mặt bàn làm việc.
     - **Hiệu chuẩn hướng bàn máy bằng Nút B:** Lấy góc yaw của kính (`head_quat`) làm hệ quy chiếu bàn máy cố định (`_desk_u_fwd`, `_desk_u_right`).
     - **Vùng chết tĩnh 3mm (`POS_DEADBAND = 0.003`m):** Hấp thụ hoàn toàn vi rung tay sinh học tự nhiên khi giữ phím.
   - **Chế độ điều khiển VR phân tách chức năng độc lập (Functional Decoupled Teleop):**
     - **Cò trỏ (Index Trigger) — Dịch chuyển tịnh tiến (Translation Only):** Tịnh tiến theo cơ chế Mặt Đồng Hồ (khóa cứng góc xoay), chuyển động thẳng tắp và cách ly tuyệt đối khỏi các rung động xoay cổ tay vô ý.
     - **Cò giữa (Hand Grip) — Cơ chế xoay bất biến tư thế (Posture-Invariant Horizon-Stabilized Mapping):**
       - **Triệt tiêu góc nghiêng tĩnh (Horizon-Leveled Hand Frame):** Tự động khử góc nghiêng/chúc tự nhiên (30°–45°) và độ xoay ban đầu của bàn tay người vận hành khi bóp giữ nút Grip, đưa hệ tọa độ tay cầm về trạng thái cân bằng phẳng theo phương chân trời.
       - **Hệ 3 trục trực giao hoàn hảo (Orthogonal Triad):** Tách bạch 100% ba cử động cổ tay tương ứng với 3 hành vi vật lý độc lập của robot:
         1. **Gập cổ tay (Lắc trái / phải):** Điều khiển quay quanh trục thẳng đứng Z (Yaw) của đầu robot $\rightarrow$ Quét ngang song song mặt bàn.
         2. **Uống nước / Đặt cốc (Duỗi / gập cổ tay):** Điều khiển quay quanh trục ngang (Pitch) của đầu robot $\rightarrow$ Ngửa lên / chúc xuống.
         3. **Vặn chìa khóa (Xoay tròn cổ tay - Clock-face Roll):** Điều khiển quay tròn quanh trục dọc cánh tay (Roll).
       - **Bám động theo hướng đầu máy ($R_{\text{tcp\_origin}}$):** Hệ 3 trục quay được neo động theo hướng thực tế hiện tại của đầu tay máy, đảm bảo ở **mọi vị trí và tư thế** (Home, nghiêng, cúi sâu, vươn xa) thì cử động cổ tay người vẫn điều khiển đúng chiều vật lý tương ứng của robot.
       - **Chống trôi / giật vi mô (Deadzone & Micro-Jitter Filtering):**
         - Thiết lập vùng chết xoay tĩnh (`ROT_DEADBAND = 0.05` rad $\approx 2.86^\circ$) và vùng chết tịnh tiến tĩnh (`POS_DEADBAND = 0.003` m $= 3$mm), triệt tiêu hoàn toàn hiện tượng rung giật sinh học vi mô (Micro-Jitter) của bàn tay khi giữ phím.
         - **Khóa trục xoay ưu thế (Dominant Single-Axis Lock):** Tự động phát hiện và khóa cứng chuyển động theo trục có biên độ xoay lớn nhất, chống hiện tượng phân rã chéo (Cross-talk) sang 2 trục phụ khi cổ tay không thực sự trực giao hoàn hảo.
     - **Bóp đồng thời cả Trigger + Grip:** Cho phép chuyển động phối hợp 6-DoF tự do đồng thời (Full 6-DoF Teleop).
     - **Cơ chế ly hợp thông minh (Smart Clutching):** Tự động chốt mỏ neo tọa độ mới khi bóp lại phím, triệt tiêu 100% hiện tượng nhảy vọt tọa độ (Zero Teleport), cho phép người vận hành với tới mọi điểm trong không gian làm việc chỉ với không gian sải tay hữu hạn.

2. **Động Học Nghịch Siêu Tốc Thuần CPU (PyRoki + JAX/jaxls) & Tối Ưu Không Gian Khớp:**
   - Giải bài toán tối ưu phi tuyến bình phương tối thiểu trên CPU bằng PyRoki kết hợp `jaxls` (Levenberg-Marquardt) và JAX JIT compile (~1ms sau warm-up), loại bỏ 100% sự phụ thuộc vào card đồ họa NVIDIA hoặc CUDA runtime.
   - Đồ thị tối ưu hóa hàm chi phí toàn diện:
     - `pose_cost_analytic_jac`: Bám vị trí (weight 50.0) và định hướng mục tiêu (weight 20.0).
     - `limit_cost`: Ràng buộc rào chắn giới hạn góc khớp (weight 0.5).
     - `rest_cost`: Chi phí neo giữ dáng mồi (weight 0.02) kết hợp hạt giống chuẩn `q_var.with_value(seed)` đảm bảo liên tục nghiệm giữa các frame.
   - Chuẩn hóa giới hạn góc khớp an toàn: Khớp khuỷu (`elbow_joint`) kẹp trong $[+0.10, +2.40]$ rad ($[+5.7^\circ, +137.5^\circ]$) chống gập va chạm thân robot; khớp nâng vai kẹp trong $[-2.65, +0.20]$ rad chống chúi cắm mặt bàn; các khớp cổ tay được bảo vệ chống tự bẻ gập.
   - `normalize_to_current`: Tự động tìm nghiệm góc khớp tương đương gần nhất với cấu hình hiện tại trong dải an toàn, triệt tiêu hiện tượng vặn xoắn cổ tay hoặc đảo nhánh IK bất thường.

3. **Bộ Ghi Dữ Liệu Đa Phương Thức Tùy Chọn (Selectable Multimodal HDF5 Recorder):**
   - Thiết kế chuyên dụng cho thu thập dữ liệu huấn luyện Robot Learning (ACT, Diffusion Policy, LeRobot).
   - Module ghi luồng HDF5 hiệu năng cao (`core/selectable_recorder.py`) với cơ chế xả mẻ định kỳ (Chunked Streaming) và bộ đệm RingBuffer trong RAM, giải quyết triệt để lỗi tràn bộ nhớ (Out-Of-Memory - OOM) trong các phiên ghi kéo dài.
   - **Tùy biến luồng dữ liệu thời gian thực:** Người vận hành có thể tick chọn hoặc bỏ tick các luồng dữ liệu trực tiếp trên giao diện Viser UI trước khi ghi để tối ưu băng thông và dung lượng lưu trữ (Kinematics, Dynamics, Wrench, Temperatures, Payload, Safety Flags, RGB Camera, Depth Camera).
   - Thu thập toàn diện dữ liệu đầu vào viễn thao: Tọa độ tay cầm VR (`observations/vr_controller_pose` $[T, 7]$), tọa độ Descartes mục tiêu (`observations/target_tcp_pose` $[T, 6]$) và lệnh điều khiển ngàm kẹp (`actions/gripper_cmd` $[T, 1]$).

4. **Đồng Bộ Thời Gian Phần Cứng Cực Phân Giải (Hardware Timestamp Sync < 1.5ms):**
   - Kiến trúc đồng bộ thời gian dựa trên camera (`teleop_quest/camera_driver.py`) đọc trực tiếp hardware timestamp từ chip ASIC của camera Intel RealSense.
   - Tự động tính toán độ lệch clock hệ thống (Clock Offset) để đồng pha giữa luồng ảnh camera và luồng dữ liệu RTDE từ robot, giảm độ trễ pha (jitter) từ $\pm 35$ms xuống $< 1.5$ms.

5. **Điều Khiển Tối Ưu Toàn Diện Tay Cầm Phải (Right Controller Teleop & AI Recorder):**
   - **Tích hợp Timer phân tách Short / Long Press & Phản hồi rung Haptic:**
     - **Nút B:** Nhấn nhả ($< 3.0$s) $\rightarrow$ Bật/Tắt viễn thao (`Toggle Teleop`); Nhấn giữ ($\ge 3.0$s) $\rightarrow$ Đưa robot về vị trí an toàn (`Go Home`) kèm rung xác nhận.
     - **Nút A:** Nhấn nhả ($< 2.0$s) $\rightarrow$ Bật/Tắt ghi dữ liệu (`Start/Stop Recording`); Nhấn giữ ($\ge 2.0$s) $\rightarrow$ Hủy tập dữ liệu hỏng (`Discard Data`) kèm rung xác nhận.
   - **Đa năng hóa cần gạt Analog (Joystick):**
     - **Nhấn thẳng (Click):** Đóng/Mở ngàm gắp (`Toggle Gripper`) kèm báo rung haptic.
     - **Gạt Trái/Phải (Axes X):** Điều chỉnh độ mở ngàm mượt mà (Continuous Gripper Control - không rung).
     - **Gạt Lên/Xuống (Axes Y):** Tăng/Giảm độ nhạy điều khiển (`Scale / Sensitivity` $0.25\times - 3.0\times$ - không rung).
   - **Đồng bộ hai chiều thời gian thực với Viser UI:** Mọi thao tác từ tay cầm phản ánh tức thì lên Checkbox (`Teleop`, `Gripper`, `Start Recording`, `Discard`), thanh trượt `Scale`, nhãn trạng thái Web UI và ngược lại với độ trễ $< 5$ms.

6. **Giao Diện Viser 3D Cockpit Tinh Gọn (100% Tiếng Anh, Lưới Gọn Gàng, Zero Emoji):**
   - Toàn bộ giao diện được chuẩn hóa bằng tiếng Anh ngắn gọn, loại bỏ sạch biểu tượng emoji và các nút trùng lặp.
   - **Thư mục Teleoperation:** Bố trí 2 checkbox nằm ngang gồm `Teleop` và `Gripper` (chia đều 50% - 50%), thanh trượt chỉnh độ nhạy `Scale` và nút `Go to Home`.
   - **Thư mục Data Recorder Setup:** Bố trí 2 checkbox nằm ngang gồm `Start Recording` và `Discard` (50% - 50%), đường phân cách và lưới 2 cột cho 8 checkbox chọn luồng dữ liệu.

7. **Bộ Công Cụ Xử Lý & Trực Quan Hóa Dữ Liệu HDF5 (`core/extract_hdf5.py` & `core/visualize_data.py`):**
   - **Trực quan hóa & Sanity Check đa phương thức (`core/visualize_data.py`):** Bố cục 2x2 gồm động học 6 góc khớp (`qpos`), tín hiệu ngàm (`gripper_cmd`), quỹ đạo không gian 3D (`tcp_pose` Start/End) và khung hình camera RGB (`wrist_camera`), hỗ trợ hiển thị GUI tương tác hoặc tự động xuất ảnh tĩnh PNG độ phân giải cao cho máy chủ Headless/SSH.
   - **Giải nén dữ liệu tự động (`core/extract_hdf5.py`):** Quét đệ quy cây dữ liệu HDF5 để xuất video MP4 (chuẩn H.264/FastStart), chuỗi ảnh Depth PNG 16-bit nguyên bản, các file `.csv` có header ngữ nghĩa vật lý và tệp tóm tắt `summary.txt`.

8. **Luồng Stream Camera Cổ Tay Đa Phương Thức (USB, WebSocket & ZeroMQ):**
   - Kiến trúc đa luồng độc lập (`CameraStreamPipeline`) truyền stream 30 FPS mượt mà:
     - **Triệt tiêu hoàn toàn Disk I/O:** Frame JPEG được lưu trực tiếp trên bộ đệm RAM (`_latest_jpeg`), loại bỏ 100% hiện tượng giật lag / glitch hình ảnh do ghi đĩa từng frame.
     - **Camera qua luồng mạng WebSocket (Khuyến nghị chuẩn mới):** Sử dụng chuẩn WebSocket RFC 6455 (`--cam-source ws --cam-url ws://<IP>:8765`), hỗ trợ truyền nhận nhị phân siêu nhẹ, độ trễ $< 3$ms, tích hợp server độc lập `teleop_quest/websocket_camera_server.py`.
     - **Camera qua mạng Ethernet ZeroMQ (`EthernetWristCamera`):** Nhận stream qua socket ZMQ Zero-lag (`CONFLATE=1`) triệt tiêu tích lũy trễ mạng LAN (`--cam-source eth --cam-url tcp://<IP>:5555`), kèm server `teleop_quest/camera_server_node.py`.
     - **Camera USB vật lý thật (`RealSenseWristCamera`):** Giao tiếp trực tiếp Intel RealSense D435/D455 qua SDK chính thức `pyrealsense2` (`--cam-source usb --camera realsense`).
     - **Camera mô phỏng 3D (`SyntheticWristCamera`):** Kết xuất ảnh RGB 640x480 góc nhìn Eye-in-Hand từ đầu tay máy trong không gian ảo.

9. **Mô Phỏng Gắp/Thả 3D & Cơ Chế Chốt Trạng Thái (Latching):**
   - Mô phỏng cơ chế hút dính vật thể quả cà chua 3D khi ngàm tiếp cận đúng vị trí (khoảng cách 3D $< 9.5$cm hoặc lệch XY $< 5.5$cm kèm độ cao Z $< 13$cm).
   - Tự động nhận diện khi đưa vật thể vào giỏ và ghi nhận sự kiện hoàn thành (`PLACE_SUCCESS`) đánh dấu cờ thành công trong file HDF5.

10. **Khóa Trục Xoay Cổ Tay Thông Minh (Soft Rotation Snapping):**
    - Giải quyết triệt để hiện tượng xoay đa trục ký sinh khi gập hoặc chúc cổ tay (loại bỏ rung lắc Yaw hoặc Roll khi đang gập Pitch).
    - Bộ lọc đường cong mềm tỷ lệ ba vùng:
      - **Tỷ lệ xoay phụ $< 55\%$:** Khóa cứng chuyển động trên một trục đơn thuần.
      - **Tỷ lệ xoay phụ $55\% - 75\%$:** Chuyển tiếp mượt mà bằng đường cong Hermite Smoothstep.
      - **Tỷ lệ xoay phụ $> 75\%$:** Cho phép quay 3D tự do đa hướng.

11. **Tự Động Kích Hoạt Docker URSim & Tiến Trình Nền ADB Watcher:**
    - Tự động nhận diện trạng thái container Docker `ursim` và kích hoạt dịch vụ PolyScope nếu chưa khởi chạy khi chạy mô phỏng.
    - Tiến trình ngầm `start_adb_watcher` liên tục giám sát cáp USB kết nối kính Meta Quest 3S và tự động cấu hình `adb reverse` (cổng 8443, 8444, 8080) trong vòng 2 giây sau khi cắm cáp.

12. **Giám Sát Hiệu Năng Thời Gian Thực & WebXR Watchdog (`teleop_quest/profiler.py`):**
    - **System Performance Profiler:** Đo lường chính xác độ trễ (latency ms) của 4 phân đoạn trọng yếu: `IK_Solver` (< 2.0ms), `Camera_Read` (< 1.0ms), `UI_Update` (< 5.0ms) và `Total_Loop` (< 33.3ms), tự động in bảng thống kê định dạng chuyên nghiệp ra Terminal mỗi 100 frames.
    - **WebXR Watchdog:** Tự động phát hiện khi mất tracking $> 1.0$s, kích hoạt `VR_PAUSED`, dừng robot an toàn, reset mỏ neo anchor, hiển thị cảnh báo edge-triggered trên Viser UI và phục hồi mượt mà khi nhận lại tín hiệu.
    - **Quản lý Vòng đời WebXR & Tự động Dọn dẹp Zombie Session:** Lắng nghe sự kiện `visibilitychange` từ kính Quest và tín hiệu `reset_session` từ server; tự động hủy bỏ session treo cũ (`disposeSession()`), mở khóa nút `Exit VR` và khôi phục nút `Enter VR` luôn sẵn sàng khởi tạo session sạch mà không cần tải lại trang.
    - **Phân tách Chế độ `--robot {real, sim}` & Chống Timeout Dashboard:** Bắt ngoại lệ socket timeout 2.0s trên cổng Dashboard 29999, cảnh báo chuyển Remote Control trực quan không gây crash, không tự động đóng phanh/bật nguồn robot thật, và tự động fallback sang cổng RTDE 30004.

13. **Tự Động Thu Hồi Tọa Độ (Auto-Snap on Stuck) & Khôi Phục Hạt Giống IK:**
    - Tự động rung phản hồi Haptic và giật tọa độ Widget 3D về đúng vị trí đầu tay máy thật (`fk_res`) khi bị lệch bám tracking quá 1.2s (`snap_threshold = 1.2`).
    - Gán lại hạt giống giải IK `teleop._last_valid_q = list(q_real)`, giúp thuật toán lập tức thoát kẹt điểm kỳ dị.

14. **Quy Hoạch Về Home 2 Pha Không Giật (500Hz URScript) & Zero-Click Reconnect:**
    - Quy hoạch chuyển động an toàn 2 pha qua lệnh native URScript `movej` chạy ở tần số 500Hz qua cổng Secondary Port 30002 với góc bo cung mềm mại (`blend_radius = 0.03`m).
    - Tự động kết nối lại RTDE và reset mỏ neo tay cầm VR ngay khi robot về tới Home mà không cần click bật lại checkbox thủ công.

15. **Hệ Thống An Toàn Đa Lớp Thời Gian Thực 20Hz (Safety Gate, 10mm Barrier & Phanh Mềm Chống Trip):**
    - **An toàn rào chắn va chạm 10mm:** Bảo vệ ngàm kẹp và camera mount khỏi tự va chạm (Self-Collision) với độ dày áo giáp bảo vệ 10mm (hỗ trợ cờ `--disable-self-collision` để bypass trong các kịch bản test đặc biệt).
    - **Rào chắn mặt bàn bảo vệ vật lý:** Ràng buộc độ cao an toàn $Z \ge 0.035$m và góc nâng vai $\le +0.20$ rad triệt tiêu nguy cơ chúc va mặt bàn làm việc.
    - **Chống vượt giới hạn khớp (`JOINT LIMIT`) & Điểm kỳ dị (`SINGULARITY`):**
      - Tinh chỉnh ngưỡng Elbow Singularity và Manipulability Metric chống kích hoạt báo lỗi giả (Anti-False Trigger) khi xoay cổ tay ở tư thế nghiêng.
      - **Thuật toán phanh mềm 20Hz (Soft Braking & Scaled Rate Limiting):** Khi tiến sát giới hạn khớp hoặc gặp kỳ dị, hệ thống tự động co giãn tỷ lệ vận tốc và hãm tốc độ êm ái, **tuyệt đối không gọi lệnh phanh giật đột ngột `servoStop()`** trong vòng lặp teleop để tránh trip bảo vệ phần cứng (Protective Stop) của cánh tay UR3e.
    - **Phát hiện bước nhảy góc khớp:** Chặn các bước nhảy mục tiêu bất thường (`max_jump > 1.5` rad).
    - **Giám sát mất kết nối (Timeout Watchdog):** Tự động hãm tốc dừng an toàn nếu mất tín hiệu VR quá ngưỡng cho phép.

16. **Kiến Trúc Mô-đun Hóa Tinh Gọn (Single Responsibility Architecture):**
    - Tách biệt rõ ràng trách nhiệm từng thành phần: `teleop/pyroki_solver.py` (bộ giải CPU IK), `core/selectable_recorder.py` (lõi ghi HDF5), `core/extract_hdf5.py` (bộ giải nén), `teleop_quest/viser_ui.py` (giao diện web) và `teleop_quest/camera_driver.py` (driver camera).

---

## 2. Kiến Trúc & Luồng Hoạt Động Của Hệ Thống

Toàn bộ hệ thống điều khiển viễn thao và thu thập dữ liệu vận hành theo chu kỳ đồng bộ 20Hz (chu kỳ 50ms), giao tiếp giữa người vận hành, máy chủ tính toán PyRoki CPU và bộ điều khiển Universal Robots UR3e qua các cổng mạng chuyên dụng: 29999 (Dashboard Server), 30002 (URScript Secondary Interface), 30003/30004 (RTDE ServoJ & Receive Interface), 8443 (HTTPS WebXR), 8444 (WSS Quest WebXR) và 8080 (Viser WebGL UI).

### 2.1. Sơ Đồ Khối Luồng Dữ Liệu

```text
[Tay Cầm Meta Quest 3S]               [Chuột Máy Tính (Viser 3D)]
  (Phím A/B, Joystick, Trigger, Grip)  (Gizmo 3D Transform Controls)
           │                                          │
           ▼ (WebSocket WSS 8444)                     │
[QuestServer & DeltaPoseSource]                       │
  (Cơ chế Mặt Đồng Hồ, Deadband, Clutch)              │
           │                                          │
           └──────────────────┬───────────────────────┘
                              ▼
                  [Target Widget 3D (Viser)]
            (Tọa độ đích 6-DoF: X, Y, Z, Qw, Qx, Qy, Qz)
                              │
                              ▼
              [Bộ Giải PyRoki CPU IK (JAX)]
           (JIT warm-up ~1ms, jaxls, rest_cost)
                              │
                              ▼
              [SafetyGate & SingularityGuard]
          (Jacobian sigma_min >= 0.035, Rate Limiter)
                              │
                              ▼
              [ServoJExecutor (RTDE Cổng 30003/30004)]
           (servoJ lookahead_time = 0.07s, gain = 350, Adaptive EMA)
                              │
                              ▼
                [Cánh Tay Robot UR3e]
                 (URSim / Robot thật)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
[SelectableDataRecorder (HDF5)]      [CameraStreamPipeline]
(RingBuffer, Chunked Streaming,      (RealSense Hardware Sync,
 Teleop Inputs, Robot Telemetry)      20 FPS Live Stream, OSD)
              │
              ▼
[core/extract_hdf5.py]
(Video MP4, Ảnh Depth PNG 16-bit, File CSV Ngữ Nghĩa)
```

### 2.2. Cơ Chế Máy Trạng Thái (Drive Mode vs. Edit Mode)

1. **Chế độ Lái Robot (Drive Mode - Mặc định khi vào VR):**
   - **Mục đích:** Thao tác điều khiển cánh tay robot trực tiếp. Giao diện HUD hiển thị trạng thái robot, camera cổ tay và tọa độ thực mà không gây cản trở tia chỉ chuột ảo.
   - **Tương tác:** Giao diện WebXR HUD kích hoạt `pointer-events: none !important`, ẩn hoàn toàn tia laser 2D. Mọi thao tác bấm nút A, B, Joystick, bóp Trigger và Grip đều được chuyển 100% sang chuyển động robot, ngàm kẹp và ghi dữ liệu AI.
   - **Bật/Tắt Teleop:** Bấm nhanh phím **B** (< 3.0s) để chuyển đổi trạng thái điều khiển; Nhấn giữ **B** ($\ge 3.0$s) để đưa robot về Home an toàn.
   - **Cách kích hoạt:** Tự động kích hoạt khi vào VR hoặc bấm phím **B/Y**.

2. **Chế độ Chỉnh Sửa Giao Diện (Edit Mode):**
   - **Mục đích:** Bật lại tia laser chuột 2D để tương tác với menu giao diện web, chỉnh thông số hoặc xoay góc nhìn camera.
   - **Khóa an toàn:** Cánh tay robot được ngắt điều khiển tức thì (`teleop.enabled = False`). Mọi thao tác vung tay hay bóp phím trong chế độ này đều không làm chuyển động robot.
   - **Quay lại Drive Mode:** Bấm lại phím **B/Y**. Tia laser biến mất và hệ thống tiếp tục điều khiển robot ngay lập tức.

### 2.3. Quy Trình Vòng Lặp Điều Khiển Thời Gian Thực (Chu Kỳ 20Hz - 50ms)

Trong mỗi chu kỳ 50ms của `teleop_quest/main_quest.py`, hệ thống thực thi tuần tự 6 bước:

1. **Bước 1: Thu thập & Lọc tín hiệu điều khiển (`DeltaPoseSource` / `WidgetPoseSource`):**
   - Áp dụng cơ chế Mặt Đồng Hồ, bộ lọc Soft Rotation Snapping và vùng chết tĩnh 3mm.
   - Khống chế mục tiêu nằm trong hộp an toàn: $X, Y \in [-0.52, 0.52]$m, $Z \in [0.035, 0.60]$m, bán kính $R \in [0.10, 0.52]$m.

2. **Bước 2: Giải ngược động học siêu tốc trên CPU (`PyrokiIKSolver`):**
   - Giải bài toán bình phương tối thiểu phi tuyến trong ~1ms bằng PyRoki + `jaxls` trên CPU.
   - Áp dụng chi phí neo giữ `rest_cost` và `normalize_to_current` để bảo đảm tính liên tục của nghiệm góc khớp.

3. **Bước 3: Kiểm định an toàn & Điểm kỳ dị (`SafetyGate` & `SingularityGuard`):**
   - Đánh giá ma trận Jacobian của cấu hình mục tiêu. Chặn lệnh và phát rung cảnh báo nếu $\sigma_{min} < 0.035$.
   - Kẹp vận tốc góc khớp tối đa qua Scaled Vector Rate Limiter ($\le 1.0$ rad/s).

4. **Bước 4: Thực thi lệnh thời gian thực (`ServoJExecutor`):**
   - Truyền gói 6 góc khớp mục tiêu $q_{target}$ qua RTDE bằng lệnh `servoJ(q, t=0.05, lookahead_time=0.12, gain=250)`.

5. **Bước 5: Ghi dữ liệu HDF5 & Mô phỏng gắp nhặt (`SelectableDataRecorder`):**
   - Mô phỏng tương tác vật lý hút/gắp vật thể cà chua 3D.
   - Nếu cờ ghi dữ liệu đang bật, nạp dữ liệu vào RingBuffer và xả xuống đĩa theo mẻ chunked HDF5 định kỳ.

6. **Bước 6: Thu nhận hình ảnh & Đồng bộ thời gian phần cứng (`CameraStreamPipeline`):**
   - Thu nhận frame ảnh đồng bộ thời gian từ RealSense hardware clock (độ trễ jitter $< 1.5$ms).
   - Nén MJPEG tích hợp lớp phủ OSD hiển thị trực tiếp lên WebXR Cockpit và Viser sidebar.

---

## 3. Danh Sách Thư Viện & Môi Trường Cài Đặt

### Yêu Cầu Hệ Thống
- **Hệ điều hành:** Ubuntu 20.04 hoặc 22.04 LTS (x86_64).
- **Bộ vi xử lý:** CPU đa nhân (Intel Core i5/i7/i9 hoặc AMD Ryzen 5/7/9). Không yêu cầu card đồ họa rời NVIDIA hay GPU CUDA.
- **Bộ điều khiển Robot:** Trình giả lập Docker URSim hoặc Tủ điều khiển Robot UR3e thật (Cổng 29999, 30002, 30003/30004).

### Bảng Thư Viện Python

| Thư viện | Phiên bản | Vai trò trong hệ thống |
| :--- | :--- | :--- |
| `jax[cpu]`, `jaxlie` | 0.4.20+, 1.0.0+ | Tính toán đại số Lie và biên dịch JIT siêu tốc trên CPU. |
| `jaxls` | git (brentyi/jaxls) | Bộ giải bình phương tối thiểu Levenberg-Marquardt cho bài toán IK. |
| `pyroki` | git (chungmin99/pyroki) | Thư viện động học và tối ưu hóa robot trên nền tảng JAX. |
| `yourdfpy` | 0.0.53+ | Phân tích cú pháp mô hình URDF robot (`assets/ur3e_custom.urdf`). |
| `robot_descriptions` | 1.10.0+ | Nạp mô tả động học robot tiêu chuẩn. |
| `torch` | 2.0.0+ (CPU) | Tính toán ma trận Jacobian và giá trị suy biến singular value. |
| `viser` | 0.2.0+ | Máy chủ WebGL 3D Visualization, Gizmo tương tác và giao diện Web UI. |
| `ur_rtde` | 1.5.0+ | Giao tiếp thời gian thực RTDE với bộ điều khiển Universal Robots. |
| `opencv-python-headless` | 4.8.0+ | Xử lý ảnh, truyền stream MJPEG, xuất video MP4 và xuất ảnh Depth 16-bit. |
| `pyrealsense2` | 2.54.0+ | SDK giao tiếp trực tiếp camera độ sâu Intel RealSense với hardware timestamp. |
| `transforms3d` | 0.4.1+ | Chuyển đổi giữa góc Euler, Quaternion và ma trận quay SE(3). |
| `trimesh` | 4.0.0+ | Phân tích lưới 3D robot, ngàm gắn camera và rào chắn vật cản. |
| `h5py` | 3.9.0+ | Ghi luồng HDF5 phân mảnh (Chunked) hiệu năng cao và trích xuất dữ liệu. |
| `pyyaml` | 6.0+ | Đọc cấu hình rào cản (`obstacle_scene.yml`) và ngưỡng kỳ dị (`singularity.yaml`). |
| `aiohttp` | 3.8.0+ | Máy chủ bất đồng bộ phục vụ WebXR HTTPS và WebSocket độ trễ thấp. |
| `cryptography` | 41.0+ | Tự động sinh chứng chỉ SSL/TLS cục bộ phục vụ kết nối HTTPS WebXR. |
| `numpy`, `scipy` | 1.24+, 1.11+ | Tính toán ma trận số học và bộ lọc tín hiệu làm mượt EMA. |

### Lệnh Cài Đặt

```bash
# 1. Tạo và kích hoạt môi trường ảo Python 3.11
python3 -m venv /home/nguyen/ur_ws/pyroki_ursim/.venv
source /home/nguyen/ur_ws/pyroki_ursim/.venv/bin/activate

# 2. Cập nhật pip và cài đặt toàn bộ gói phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Cấu Trúc Thư Mục Dự Án

```text
ur_ws/pyroki_ursim/
├── main.py                          # Điểm khởi chạy CLI thống nhất của hệ thống
├── build_zip.sh                     # Script tự động đóng gói dự án thành Teleop_Quest_Project.zip
├── Dockerfile                       # Image Docker Python 3.11-slim (CPU PyRoki, OpenCV, ZMQ, RTDE)
├── docker-compose.yml               # Cấu hình triển khai Zero-Setup với network_mode: host
├── .env.example                     # File mẫu biến môi trường cấu hình IP Robot và Camera
├── HUONG_DAN_CHAY.md                # Hướng dẫn 3 bước khởi chạy nhanh bằng Docker
├── README.md                        # Tài liệu hướng dẫn sử dụng toàn diện
├── requirements.txt                 # Danh sách gói phụ thuộc Python
├── reset_home.py                    # Script hỗ trợ khẩn cấp đưa robot về Home
├── assets/
│   ├── ur3e_custom.urdf             # URDF UR3e có ngàm kẹp và camera offset
│   ├── obstacle_scene.yml           # Cấu hình ranh giới an toàn mặt bàn
│   ├── robot_camera_mount_1.stl     # Mô hình 3D ngàm gắn camera vào bích tool0
│   ├── d455.stl                     # Mô hình 3D camera Intel RealSense D455
│   ├── episode_0_visualization.png  # Ảnh tổng quan kiểm định dữ liệu đa phương thức (2x2 plots)
│   └── quest3_ctrl_*.glb            # Mô hình 3D tay cầm hiển thị phản hồi phím
├── config/
│   └── singularity.yaml             # Ngưỡng an toàn Jacobian singularity (sigma_min)
├── core/
│   ├── selectable_recorder.py       # Bộ ghi dữ liệu HDF5 dạng luồng phân mảnh
│   ├── extract_hdf5.py              # Công cụ giải nén HDF5 sang MP4, PNG 16-bit và CSV
│   ├── visualize_data.py            # Phân tích và trực quan hóa dữ liệu đa phương thức HDF5 (2x2 plots)
│   ├── pyroki_ursim_control.py      # Bộ điều khiển trung tâm (RTDE, PyRoki, URScript)
│   ├── verify_dataset.py            # Công cụ kiểm định chất lượng tệp HDF5
│   └── replay_episode.py            # Công cụ tái lập quỹ đạo thao tác
├── logs/
│   ├── ai_training_data/            # Tệp HDF5 lưu trữ các episode huấn luyện AI
│   ├── quest_runs/                  # Lưu vết dữ liệu viễn thao Meta Quest (.jsonl)
│   └── singularity_data/            # Dữ liệu ma trận suy biến Jacobian
├── safety/
│   ├── __init__.py                  # Khởi tạo package an toàn
│   └── singularity_guard/
│       ├── guard_core.py            # Giám sát điểm kỳ dị thời gian thực
│       ├── jacobian_metrics.py      # Tính toán giá trị suy biến nhỏ nhất sigma_min
│       └── calibration.py           # Hiệu chuẩn ngưỡng kỳ dị trong không gian làm việc
├── teleop/
│   ├── __init__.py                  # Khởi tạo package viễn thao
│   ├── pyroki_solver.py             # Bộ giải PyrokiIKSolver (JAX JIT CPU, ~1ms, rest_cost)
│   └── teleoperation.py             # TeleoperationController, ServoJExecutor, SafetyGate
├── teleop_quest/
│   ├── __init__.py                  # Khởi tạo package Quest teleop
│   ├── bootstrap.py                 # Điều phối khởi động (Docker URSim, ADB, SSL)
│   ├── camera_pipeline.py           # Quản lý luồng nền camera, preview OSD & streaming
│   ├── camera_driver.py             # Driver camera hợp nhất (USB, WebSocket RFC 6455 & ZeroMQ)
│   ├── camera_server_node.py        # Node phát stream video camera qua Ethernet ZeroMQ
│   ├── websocket_camera_server.py   # Node phát stream video camera qua mạng WebSocket (RFC 6455)
│   ├── synthetic_camera.py          # Render camera ảo cổ tay 640x480 (Eye-in-Hand)
│   ├── viser_ui.py                  # Giao diện Viser UI tinh gọn (Lưới 2 cột, 100% tiếng Anh)
│   ├── grasp_simulator.py           # Mô phỏng gắp/thả quả cà chua 3D vào giỏ
│   ├── quest_server.py              # Server HTTPS (8443) và WSS (8444)
│   ├── network_utils.py             # Dò tìm IP LAN và tự sinh chứng chỉ SSL
│   ├── pose_sources.py              # WidgetPoseSource đọc tọa độ chuột có lọc EMA
│   ├── delta_pose_source.py         # Cơ chế Mặt Đồng Hồ, Posture-Invariant Hand Frame & Deadzone
│   ├── timeout_watchdog.py          # Watchdog ngắt khẩn cấp khi mất tín hiệu
│   ├── home_planner.py              # Đưa robot về Home an toàn 2 pha (500Hz movej)
│   ├── profiler.py                  # Đo đạc vi độ trễ và xuất báo cáo hiệu năng mỗi 100 frames
│   ├── setup_quest_adb.sh           # Script chuyển tiếp cổng ADB sang localhost cho Quest
│   └── static/
│       ├── index.html               # Giao diện WebXR Cockpit trên kính Quest
│       ├── main.js                  # Xử lý gamepad WebXR, Toggle Gripper & phím VR
│       └── style.css                # Style hiển thị HUD hiện đại
└── tests/
    ├── test_system_profiler_and_watchdog.py # Kiểm thử Profiler, WebXR Watchdog & Dashboard Timeout
    ├── test_extract_hdf5.py         # Kiểm thử toàn diện công cụ giải nén HDF5
    ├── test_visualize_data.py       # Kiểm thử công cụ trực quan hóa đa phương thức HDF5
    ├── test_base_frame_rotation.py  # Kiểm thử xoay Base Frame và chiều xoay cơ học VR
    ├── test_vr_key_integration.py   # Kiểm thử phân tách phím vật lý VR A/X
    ├── test_vr_record_and_ui.py     # Kiểm thử Viser UI Checkbox và HDF5 streaming
    ├── test_realsense.py            # Kiểm thử SDK camera Intel RealSense
    ├── test_dual_camera.py          # Kiểm thử hệ thống camera đa phương thức (USB & Ethernet ZMQ)
    ├── test_websocket_camera.py     # Kiểm thử luồng camera qua mạng WebSocket (RFC 6455)
    ├── test_webxr_reconnect_and_occlusion.py # Kiểm thử khôi phục kết nối WebXR và góc khuất tay cầm
    ├── test_cockpit_joint_control.py# Kiểm thử điều khiển khớp 6-DoF trực tiếp trên Cockpit WebXR
    ├── test_collision_visualization.py # Kiểm thử hiển thị ranh giới va chạm 3D
    ├── test_joint_limits.py         # Kiểm thử giới hạn góc khớp và phanh mềm 20Hz
    ├── test_multi_layer_safety.py   # Kiểm thử hệ thống rào chắn an toàn đa lớp
    ├── test_simulated_pick_and_place.py # Kiểm thử chu trình gắp thả mô phỏng
    ├── test_singularity_guard.py    # Kiểm thử Jacobian singularity metric
    ├── test_camera.py               # Kiểm thử driver camera cơ sở
    └── test_viser_ui_layout.py      # Kiểm thử bố cục giao diện Viser UI
```

---

## 5. Hướng Dẫn Vận Hành

### Bước 1: Kích Hoạt Môi Trường Trên Máy Mới (Chỉ Gõ 1 Lệnh - 100% Offline)
Người dùng trên máy tính mới **không cần cài đặt bất kỳ thư viện nào** (không cần pip, venv hay conda). Chỉ cần mở Terminal tại thư mục dự án và chạy:

```bash
source activate_env.sh
```
*(Nếu là lần đầu chạy trên máy mới, script sẽ tự động nạp file ảnh Docker Offline `teleop_quest_offline.tar` trong vài giây và thiết lập alias lệnh `python`. Bạn cũng có thể vào thẳng shell bên trong Docker bằng cách gõ `./activate_env.sh`).*

---

### Bước 2: Chạy Chương Trình (Gõ Lệnh `python` Y Hệt Máy Thật)
Sau khi kích hoạt môi trường ở Bước 1, bạn có thể copy-paste trực tiếp toàn bộ các câu lệnh điều khiển bên dưới:

```bash
# 1. Chạy với URSim mô phỏng (Camera ảo 3D - Mặc định tự bật Docker):
python main.py --robot sim

# 2. Chạy với Robot UR3e thật (TỰ ĐỘNG DÒ TÌM IP QUA DHCP - Không cần nhập IP):
python main.py --auto

# Hoặc nếu đặt IP Tĩnh cố định:
python main.py --robot real --robot-ip 10.42.0.242

# 3. Chạy CẢ HAI: Robot UR3e thật + Camera Intel RealSense cắm cáp USB (Cấu hình chuẩn phòng Lab):
python main.py --robot real --robot-ip 10.42.0.242 --cam-source usb --camera realsense

# 4. Chạy CẢ HAI: Robot UR3e thật + Camera qua luồng mạng WebSocket (Khuyến nghị chuẩn mới):
python main.py --robot real --robot-ip 10.42.0.242 --cam-source ws --cam-url ws://127.0.0.1:8765
# (Trên máy phát chạy: python teleop_quest/websocket_camera_server.py --camera realsense --port 8765 --fps 30)

# 5. Chạy CẢ HAI: Robot UR3e thật + Camera qua cáp mạng Ethernet (ZeroMQ Zero-Lag):
python main.py --robot real --robot-ip 10.42.0.242 --cam-source eth --cam-url tcp://127.0.0.1:5555
# (Trên máy phát chạy: python teleop_quest/camera_server_node.py --camera realsense --port 5555 --fps 30)
```

*(Tùy chọn: Nếu muốn chạy ngầm toàn bộ dịch vụ qua Docker Compose, bạn chỉ cần gõ: `docker compose up -d` và tắt bằng: `docker compose down`).*

---

### Bước 3: Mở Các Đường Dẫn Giao Diện (Localhost & VR)
Sau khi lệnh `python main.py ...` khởi động, toàn bộ dịch vụ mạng nội bộ sẽ lắng nghe trực tiếp trên máy của bạn:

1. **Trên Trình Duyệt Máy Tính:**
   - **Giao diện 3D Viser Dashboard:** [http://localhost:8080](http://localhost:8080) (Điều khiển 3D Gizmo bằng chuột, bật/tắt Teleop, ghi dữ liệu HDF5).
   - **Giao diện WebXR Cockpit:** [https://localhost:8443](https://localhost:8443)
   - **Giao diện URSim PolyScope (noVNC):** [http://localhost:6080](http://localhost:6080)

2. **Trên Kính Thực Tế Ảo Meta Quest (Meta Quest Browser):**
   - **Cách 1 - Cắm Cáp USB Type-C (Khuyến nghị - Tốc độ cao & Zero Lag):**
     - Cắm cáp USB nối kính Quest với máy tính.
     - Nếu Terminal xuất hiện dòng cảnh báo màu vàng:
       `⚠️  [QUEST USB] Phát hiện kính Quest nhưng CHƯA CẤP QUYỀN! Hãy đeo kính lên và bấm 'Allow USB Debugging'!`
       -> Đeo kính lên mắt, chọn **"Always allow from this computer"** rồi bấm **"Allow" (Cho phép)**.
     - Chờ dòng thông báo màu xanh lá hiện ra trên Terminal:
       `✅ [QUEST USB] Đã kết nối cáp USB thành công! Mở trình duyệt trên Quest và vào ngay: https://localhost:8443`
     - Mở Meta Quest Browser trên kính và truy cập: **`https://localhost:8443`** *(vào được ngay lập tức)*.
   - **Cách 2 - Dùng Wi-Fi cùng mạng LAN với máy tính:**
     - Mở Meta Quest Browser trên kính và truy cập: **`https://<IP_LAN_in_trên_terminal>:8443`** (ví dụ: `https://192.168.1.15:8443`).

3. **Vượt Qua Cảnh Báo SSL Tự Ký Trên Trình Duyệt (Chỉ làm 1 lần đầu):**
   - WebXR yêu cầu bảo mật HTTPS. Vì chứng chỉ được hệ thống tự sinh nội bộ, trình duyệt sẽ hiện *"Your connection is not private"* (Kết nối không riêng tư).
   - Bạn chỉ cần bấm vào **Advanced (Nâng cao)** -> Chọn **Proceed to localhost (Tiếp tục truy cập)**.
   - Bấm nút **"Enter VR"** màu xanh để vào không gian điều khiển robot!

---

### Bước 4: Tùy Chỉnh Cấu Hình Trong File `.env` (Nếu Cần)
Nếu dải IP của Robot hoặc Camera trong phòng Lab của bạn thay đổi, chỉ cần mở file `.env` và sửa:
```bash
ROBOT_IP=10.42.0.242           # Địa chỉ IP của Robot UR3e
CAM_URL=tcp://10.42.0.100:5555  # URL nhận luồng video camera qua mạng
```


---

### Bước 2: Điều khiển bằng Viser Web UI (Máy tính)

Mở trình duyệt web trên máy tính và truy cập `http://localhost:8080`:

1. **Thư mục Teleoperation:**
   - **`Teleop` Checkbox:** Tích chọn để kích hoạt cánh tay robot bám theo mục tiêu (Status chuyển sang `Tracking`).
   - **`Gripper` Checkbox:** Tích chọn để đóng ngàm kẹp / hút vật thể (hoặc bỏ chọn để nhả vật thể).
   - *(Cả 2 checkbox trên được xếp gọn trên cùng 1 hàng ngang, mỗi ô chiếm 50% độ rộng).*
   - **`Scale` Slider:** Điều chỉnh độ nhạy chuyển động ($0.25\times - 3.0\times$) theo thời gian thực.
   - **`Go to Home` Button:** Kích hoạt quỹ đạo di chuyển 2 pha êm ái đưa robot về tư thế sẵn sàng an toàn.

2. **Thư mục Data Recorder Setup:**
   - **`Start Recording` Checkbox:** Bắt đầu phiên ghi HDF5 mới. Nhãn checkbox tự động đổi thành **`Recording (Ep X)`**. Bỏ tick để kết thúc phiên và lưu tệp xuống đĩa.
   - **`Discard` Checkbox:** Hủy bỏ phiên ghi hiện tại và xóa file HDF5 đang ghi dở (checkbox tự động nhả sau 200ms).
   - *(Cả 2 checkbox trên được xếp gọn trên cùng 1 hàng ngang, mỗi ô chiếm 50% độ rộng).*
   - **Lưới chọn luồng dữ liệu (2-Column Grid):**
     - Cột trái: `Kinematics`, `Wrench`, `Payload`, `RGB Camera`.
     - Cột phải: `Dynamics`, `Temperatures`, `Safety Flags`, `Depth Camera`.
     - Bỏ tick luồng nào sẽ tự động lược bỏ dataset tương ứng trong tệp HDF5.

3. **Thao tác Gizmo 3D:** Nhấp chuột và kéo các trục mũi tên hoặc vòng xoay của khối Target Widget 3D để điều khiển robot trực quan bằng chuột.

---

### Bước 3: Hướng Dẫn Chi Tiết Điều Khiển Bằng Kính Meta Quest 3S (WebXR)

#### 3.1. Bảng Chức Năng Các Phím Trên Tay Cầm Meta Quest 3S (Tay Cầm Phải - Right Controller)

| Phím bấm | Thao tác / Thời gian | Chức năng chính | Phản hồi rung (Haptic) | Đồng bộ Viser UI | Mô tả chi tiết hành vi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cò Trỏ (Index Trigger)** | Bóp giữ ($> 0.25$) | **Dịch Chuyển Tịnh Tiến (Translation)** | Click nhẹ ($35\%$, $40$ms) khi bắt đầu bóp | Status: `Tracking` | Tịnh tiến theo cơ chế Mặt Đồng Hồ (khóa góc xoay). Nhả cò dừng robot ngay lập tức và reset mỏ neo tọa độ. |
| **Cò Giữa (Hand Grip)** | Bóp giữ ($> 0.25$) | **Xoay Góc Cổ Tay (Rotation)** | Click nhẹ ($35\%$, $40$ms) khi bắt đầu bóp | Status: `Tracking` | Xoay 3 trục độc lập cố định theo hệ tọa độ Gốc (Base Frame Left-Multiply): Pitch ngửa/cụp theo phương cánh tay hướng về thân Base, Roll xoay quanh cánh tay, Yaw xoay quanh Base Z. Nhả grip dừng robot ngay lập tức. |
| **Nút B** | **Nhấn nhả ($< 3.0$s)** | **Bật / Tắt Viễn Thao (Toggle Teleop)** | Rung vừa ($70\%$, $150$ms) | Checkbox `Teleop` (Tick/Untick) | Chuyển đổi trạng thái điều khiển robot. Bật: `Tracking`; Tắt: Ngắt điều khiển tức thì (Safe Pause). |
| **Nút B** | **Nhấn giữ ($\ge 3.0$s)** | **Về Vị Trí An Toàn (Go Home)** | Rung mạnh ($100\%$, $300$ms) | Status: `Home`, Checkbox `Teleop` $\rightarrow$ Untick | Kích hoạt thuật toán quy hoạch quỹ đạo URScript 500Hz đưa robot về cấu hình khớp an toàn chuẩn. |
| **Nút A** | **Nhấn nhả ($< 2.0$s)** | **Bật / Tắt Ghi Dữ Liệu (Record AI)** | Bắt đầu: Rung ($90\%$, $200$ms)<br>Lưu: Rung ($50\%$, $100$ms) | Checkbox `Start Recording` (Tick/Untick)<br>Nhãn: `Recording (Ep X)` | Bấm lần 1: Bắt đầu episode mới. Bấm lần 2: Kết thúc và lưu tệp HDF5 hoàn chỉnh vào thư mục `logs/`. |
| **Nút A** | **Nhấn giữ ($\ge 2.0$s)** | **Hủy Tập Dữ Liệu (Discard Data)** | Rung cảnh báo ($70\%$, $300$ms) | Checkbox `Start Recording` $\rightarrow$ Untick<br>Status: `Discarded Episode` | Hủy bỏ phiên ghi hiện tại nếu thao tác lỗi/thất bại, xóa tệp HDF5 dở dang và giải phóng tài nguyên. |
| **Joystick Click** | **Nhấn thẳng (Click R3)** | **Đóng / Mở Ngàm Kẹp (Toggle Gripper)** | Kẹp: Rung ($90\%$, $150$ms)<br>Nhả: Rung ($40\%$, $150$ms) | Checkbox `Gripper` (Tick/Untick) | Chuyển đổi trạng thái hút/kẹp hoặc nhả phôi của ngàm gắp điện / khí nén. |
| **Joystick Axes X** | **Gạt Trái / Phải** | **Chỉnh Độ Mở Ngàm Mượt Mà (Continuous Gripper)** | **Không rung** | Checkbox `Gripper` cập nhật khi vượt ngưỡng $50\%$ | Gạt Phải ($X > +0.35$): Khép dần ngàm gắp; Gạt Trái ($X < -0.35$): Mở dần ngàm gắp mượt mà. |
| **Joystick Axes Y** | **Gạt Lên / Xuống** | **Điều Chỉnh Độ Nhạy (Scale / Sensitivity)** | **Không rung** | Thanh trượt `Scale` Slider ($0.25\times - 3.00\times$) | Đẩy Lên ($Y < -0.35$): Tăng độ nhạy; Kéo Xuống ($Y > +0.35$): Giảm độ nhạy theo bước $0.05\times$. |
| **Nút Meta (Oculus)** | Nhấn giữ ($2.0$s) | **Căn Chuẩn Trục Nhìn (Recenter)** | Theo mặc định của kính | Không ảnh hưởng robot | Đồng bộ lại tâm không gian phòng thực tế ảo của kính Meta Quest. |

#### 3.2. Cơ Chế Mặt Đồng Hồ (True Mechanical Clock Navigation)
- **Tâm đồng hồ:** Trục quay khớp đế Base Z $(0, 0)$.
- **Kim đồng hồ:** Góc vật lý của khớp đế $q_0$.
- **Thao tác Sang Phải / Trái:** Quay kim đồng hồ (khớp đế $q_0$) quanh Base Z, tự động bù góc xoay trục Z ngàm kẹp để giữ ngàm luôn hướng thẳng vào bàn máy.
- **Thao tác Tiến / Lùi:** Duỗi dài hoặc thu ngắn kim đồng hồ theo đúng phương vươn cơ học $\vec{u}_{reach} = [\cos(q_0), \sin(q_0), 0.0]$, loại bỏ hiện tượng bị bẻ ngoặt khi kéo tay thẳng về sau.
- **Thao tác Cao / Thấp:** Nâng hạ độ cao Z thẳng đứng trên mặt bàn, luôn nằm trong khoảng an toàn $Z \in [0.035, 0.60]$m.

#### 3.3. Kỹ Thuật Ly Hợp Điều Khiển Tầm Xa (Smart Clutching)
Khi cần điều khiển robot di chuyển quãng đường dài vượt quá sải tay:
1. Bấm giữ Cò trỏ và đưa tay về phía trước đến hết tầm với thoải mái.
2. **Nhả Cò trỏ:** Robot dừng lại và giữ nguyên vị trí hiện tại.
3. Rút tay cầm về gần ngực hoặc vị trí thoải mái nhất.
4. **Bấm giữ lại Cò trỏ:** Hệ thống tự động neo mốc mới (Reset Anchor). Robot tiếp tục tiến tới mượt mà, hoàn toàn không bị nhảy vọt tọa độ (Zero Teleport).

#### 3.4. Kết Nối Kính Meta Quest Qua Cáp USB (Cơ Chế Auto-Heal ADB USB)
Để đạt độ trễ thấp nhất và triệt tiêu giật lag do sóng Wi-Fi:
1. Cắm cáp USB-C 3.0 giữa PC và kính Meta Quest.
2. Hệ thống chạy tiến trình ngầm **Auto-Heal ADB USB Daemon** liên tục mỗi 2.5 giây.
3. Nếu Terminal hiện cảnh báo màu vàng:
   ```text
   ⚠️  [QUEST USB] Phát hiện kính Quest nhưng CHƯA CẤP QUYỀN! Hãy đeo kính lên và bấm 'Allow USB Debugging'!
   ```
   $\rightarrow$ Đeo kính lên mắt, tích chọn **"Always allow from this computer"** rồi bấm **"Allow" (Cho phép)**.
4. Chờ dòng thông báo màu xanh lá xuất hiện trên Terminal:
   ```text
   ✅ [QUEST USB] Đã kết nối cáp USB thành công! Mở trình duyệt trên Quest và vào ngay: https://localhost:8443
   ```
5. Mở trình duyệt Meta Quest Browser trên kính và truy cập:
   ```text
   https://localhost:8443
   ```
6. Bấm **Advanced** $\rightarrow$ **Proceed to localhost** $\rightarrow$ Bấm **Enter VR** để bắt đầu điều khiển.

*(Lưu ý: Kể cả khi người dùng bấm Allow muộn, hay đang chạy mà bị tuột dây USB cắm lại, hệ thống sẽ tự động khôi phục kết nối trong vòng 2 giây mà không cần khởi động lại chương trình!)*

---

## 6. Hướng Dẫn Triển Khai Trên Cánh Tay Robot UR3e Thực Tế

### 6.1. Kết Nối Phần Cứng & Cấu Hình Mạng LAN

Cắm cáp mạng Ethernet trực tiếp từ cổng mạng của máy tính PC (hoặc củ chuyển USB-to-Ethernet 3.0) vào cổng mạng trên tủ điều khiển UR Control Box.

#### Lựa chọn 1: Cấu hình tự động qua DHCP (Khuyên dùng)
1. Trên màn hình PolyScope Teach Pendant: Vào **Settings** $\rightarrow$ **System** $\rightarrow$ **Network**.
2. Chọn **DHCP** $\rightarrow$ Bấm **Apply**.
3. Chờ trạng thái chuyển sang màu xanh lá: **`Network is connected`**.
4. Ghi lại địa chỉ IP hiển thị (ví dụ: `10.42.0.242` hoặc `192.168.1.xxx`).
5. Trên máy tính điều khiển PC, kiểm tra kết nối:
   ```bash
   ping -c 3 <ROBOT_IP>
   nc -zv <ROBOT_IP> 29999 30002 30004
   ```

#### Lựa chọn 2: Cài đặt IP Tĩnh (`10.42.0.242`)
1. **Trên Teach Pendant:** Đặt IP `10.42.0.242`, Subnet `255.255.255.0`, Gateway `10.42.0.1`.
2. **Trên máy tính điều khiển (PC):** Đặt IP tĩnh `10.42.0.1/24` cho card mạng Ethernet qua lệnh `nmcli`:
   ```bash
   sudo nmcli con add type ethernet con-name "UR3e-LAN" ifname <TEN_CARD_MANG> ip4 10.42.0.1/24
   sudo nmcli con up "UR3e-LAN"
   ```

---

### 6.2. Cấu Hình URCap & Tương Thích Thanh Ghi Fieldbus
Nếu bộ điều khiển báo lỗi `One of the RTDE input registers are already in use!`:
1. Trên PolyScope: Mở **Installation** $\rightarrow$ **Fieldbus** $\rightarrow$ Tắt (bỏ chọn) **Profinet** và **EtherNet/IP**.
2. Hệ thống `ServoJExecutor` tự động gửi lệnh `stop\n` tới Dashboard Server (Port 29999) khi kết nối và tự động chuyển sang dải thanh ghi trên 24-47 (`FLAG_UPPER_RANGE_REGISTERS`), tránh xung đột hoàn toàn với URCap của ngàm kẹp.

---

### 6.3. Quy Trình Chuẩn Hóa 10 Bước Trên Tay Dạy PolyScope

1. **Bước 1: Bật nguồn Control Box:** Bật công tắc cơ học trên mặt trước của tủ điều khiển UR.
2. **Bước 2: Thiết lập mạng LAN:** Xác nhận trong **Settings** $\rightarrow$ **System** $\rightarrow$ **Network** báo `Network is connected`.
3. **Bước 3: Kích hoạt Remote Control:** Vào **Settings** $\rightarrow$ **System** $\rightarrow$ **Remote Control** $\rightarrow$ Chọn **Enable**. Trên thanh tiêu đề header trên cùng, chuyển từ **Local** sang **Remote Control** (biểu tượng chuyển thành hình mạng máy tính).
4. **Bước 4: Khai báo TCP, Tải trọng (Payload) & Trọng tâm (CoG):**
   - Trong **Installation** $\rightarrow$ **General** $\rightarrow$ **TCP**: Khai báo offset (ví dụ: $Z = 0.135$m cho ngàm kẹp + mount camera).
   - Trong **Installation** $\rightarrow$ **General** $\rightarrow$ **Payload**: Khai báo khối lượng $\approx 1.1$ kg và CoG $Cx=0, Cy=0, Cz=0.045$m để chống lỗi dừng bảo vệ `Protective Stop: Position deviation`.
5. **Bước 5: Cấu hình mặt phẳng an toàn:** Trong **Installation** $\rightarrow$ **Safety** $\rightarrow$ **Planes**, thêm mặt phẳng hạn chế (*Restrict Plane*) cách mặt bàn vật lý $\ge 2$ cm.
6. **Bước 6: Tắt truyền thông Fieldbus:** Tắt Profinet và EtherNet/IP nếu không dùng PLC ngoại vi.
7. **Bước 7: Bật nguồn khớp & Nhả phanh:** Nhấn nút nguồn màu đỏ góc dưới bên trái màn hình, nhấn **ON**, sau đó nhấn **START / RELEASE BRAKES** đến khi trạng thái chuyển sang màu xanh lá (**RUNNING**).
8. **Bước 8: Đưa cánh tay về tư thế an toàn ban đầu (Ready Pose):** Đưa robot về góc khớp: Base $0^\circ$, Shoulder $-126^\circ$, Elbow $+109^\circ$, Wrist 1 $-90^\circ$, Wrist 2 $-90^\circ$, Wrist 3 $0^\circ$. Tránh tư thế duỗi thẳng cánh tay hoặc $q_5 = 0^\circ$ (kỳ dị cổ tay).
9. **Bước 9: Hạ thanh trượt tốc độ:** Kéo thanh trượt tốc độ (**Speed Slider**) xuống **10% – 20%** cho lần chạy thử nghiệm ban đầu.
10. **Bước 10: Giữ nút dừng khẩn cấp:** Người vận hành luôn cầm chắc Teach Pendant, ngón tay đặt sẵn trên nút Emergency Stop màu đỏ.

---

### 6.4. Bảng Checklist 10 Điểm Trước Khi Khởi Chạy Code Teleop

| STT | Hạng mục kiểm tra | Trạng thái đạt yêu cầu | Ghi chú |
| :---: | :--- | :---: | :--- |
| 1 | Cáp mạng LAN | Đã cắm chắc chắn | Cáp Cat5e/Cat6 cắm trực tiếp vào Control Box |
| 2 | Trạng thái mạng | `Network is connected` | Đèn xanh lá trong PolyScope Settings |
| 3 | Chế độ điều khiển | `Remote Control` | Icon mạng máy tính trên thanh header PolyScope |
| 4 | Cấu hình Payload | Đã nhập đúng kg & CoG | Tránh lỗi lệch vị trí Protective Stop |
| 5 | Ranh giới an toàn | Plane $\ge 2$ cm | Chống cánh tay va quẹt mặt bàn |
| 6 | Fieldbus | Đã tắt Profinet / EtherNet-IP | Tránh xung đột thanh ghi RTDE |
| 7 | Trạng thái nguồn | `RUNNING` (Xanh lá) | Đã bật nguồn và nhả phanh cơ học |
| 8 | Tư thế khởi đầu | Khuỷu gập tự nhiên | $q_1 \approx -126^\circ, q_2 \approx +109^\circ, q_4 \approx -90^\circ$ |
| 9 | Thanh tốc độ | Đặt mức 10% – 20% | Cho lần chạy thử nghiệm ban đầu |
| 10 | An toàn con người | Cầm sẵn nút Emergency Stop | Quan sát không gian xung quanh cánh tay |

---

### 6.5. Hướng Dẫn Cấu Hình & Kiểm Tra Camera Vật Lý Độc Lập

Trước khi chạy viễn thao robot, hãy kiểm tra độc lập camera RealSense để đảm bảo luồng hình ảnh và hardware timestamp hoạt động chính xác:

```bash
# Kiểm tra nhận diện thiết bị USB
lsusb | grep -i intel
rs-enumerate-devices --short

# Chạy bài benchmark kiểm định RealSense tự động (đánh giá latency, FPS và timestamp)
python tests/test_realsense.py --gui

# Kiểm tra hệ thống camera đa phương thức (USB & Ethernet)
python tests/test_dual_camera.py

# Kiểm tra luồng WebSocket Camera (Client/Server)
python tests/test_websocket_camera.py
```

---

## 7. Quản Lý Dữ Liệu, Kiểm Định & Trích Xuất Dữ Liệu HDF5

### 7.1. Bảng Đặc Tả Cấu Trúc Dữ Liệu HDF5 Đa Phương Thức

Tập tin ghi hình (`logs/ai_training_data/episode_X.hdf5`) tuân theo định dạng chuẩn tương thích với ACT, Diffusion Policy và LeRobot:

| Đường dẫn Dataset | Kích thước (Shape) | Kiểu dữ liệu | Mô tả & Đơn vị vật lý |
| :--- | :--- | :---: | :--- |
| `action` | `[T, 7]` | `float32` | Góc khớp mục tiêu $[q_1..q_6]$ (rad) + lệnh kẹp $[0..1]$ |
| `actions/gripper_cmd` | `[T, 1]` | `float32` | Lệnh điều khiển ngàm kẹp chuẩn hóa $[0.0, 1.0]$ |
| `observations/qpos` | `[T, 6]` | `float32` | Góc khớp thực tế (rad): `[base, shoulder, elbow, wrist_1..3]` |
| `observations/qvel` | `[T, 6]` | `float32` | Vận tốc góc khớp thực tế (rad/s) |
| `observations/tcp_pose` | `[T, 6]` | `float32` | Tọa độ TCP thực tế: `[x, y, z]` (m) + `[rx, ry, rz]` (rotation vector rad) |
| `observations/tcp_speed` | `[T, 6]` | `float32` | Vận tốc TCP thực tế: `[vx, vy, vz]` (m/s) + `[wx, wy, wz]` (rad/s) |
| `observations/tcp_force` | `[T, 6]` | `float32` | Lực tiếp xúc đầu công cụ: `[fx, fy, fz]` (N) + `[tx, ty, tz]` (Nm) |
| `observations/ft_raw_wrench` | `[T, 6]` | `float32` | Lực/mô-men thô cảm biến F/T: `[fx, fy, fz]` (N) + `[tx, ty, tz]` (Nm) |
| `observations/vr_controller_pose` | `[T, 7]` | `float32` | Tọa độ tay cầm VR: `[x, y, z]` (m) + `[qx, qy, qz, qw]` (quaternion) |
| `observations/target_tcp_pose` | `[T, 6]` | `float32` | Tọa độ TCP mục tiêu phát lệnh: `[x, y, z, rx, ry, rz]` |
| `observations/gripper` | `[T, 1]` | `float32` | Trạng thái đóng/mở thực tế của ngàm kẹp $[0.0, 1.0]$ |
| `observations/currents` | `[T, 6]` | `float32` | Dòng điện động cơ 6 khớp (Amperes) |
| `observations/joint_voltages` | `[T, 6]` | `float32` | Điện áp động cơ 6 khớp (Volts) |
| `observations/temperatures` | `[T, 6]` | `float32` | Nhiệt độ động cơ 6 khớp ($^\circ$C) |
| `observations/payload` | `[T, 4]` | `float32` | Tải trọng công cụ: `[mass_kg, cog_x, cog_y, cog_z]` |
| `observations/tool_accelerometer` | `[T, 3]` | `float32` | Gia tốc kế đầu công cụ: `[ax, ay, az]` (m/s$^2$) |
| `observations/safety_flags` | `[T, 2]` | `int32` | Cờ trạng thái dừng khẩn cấp và dừng bảo vệ |
| `observations/timestamps_ms` | `[T]` | `int64` | Timestamp đồng hồ hệ thống (milliseconds) |
| `observations/timestamps_ns` | `[T]` | `int64` | Timestamp đồng bộ phần cứng camera RealSense (nanoseconds) |
| `observations/images/wrist_camera` | `[T, H, W, 3]` | `uint8` | Khung hình video màu RGB (định dạng RGB) |
| `observations/images/wrist_camera_depth` | `[T, H, W]` | `uint16` | Khung hình độ sâu Depth (khoảng cách vật lý tính bằng milimet) |

---

### 7.2. Công Cụ Trích Xuất Dữ Liệu Tự Động (`core/extract_hdf5.py`)

Tập lệnh giải nén tự động quét đệ quy toàn bộ cây dữ liệu HDF5 và xuất ra các tệp tin dễ đọc mà không can thiệp sửa đổi tệp HDF5 gốc:

#### Cú pháp dòng lệnh CLI:
```bash
# 1. Trích xuất mặc định (Tự tạo thư mục <tên_file>_extracted/ cùng cấp):
python core/extract_hdf5.py --file logs/ai_training_data/episode_0.hdf5

# 2. Chỉ định thư mục lưu kết quả tùy ý:
python core/extract_hdf5.py --file logs/ai_training_data/episode_0.hdf5 --output-dir /duong_dan/thu_muc_xuat

# 3. Ghi đè tốc độ khung hình (FPS) cho Video MP4:
python core/extract_hdf5.py --file logs/ai_training_data/episode_0.hdf5 --fps 30.0
```

#### Cấu trúc thư mục kết quả sinh ra:
```text
episode_0_extracted/
├── episode_metadata.json          # Toàn bộ thuộc tính phiên ghi (hz, duration, streams...)
├── summary.txt                    # Báo cáo thống kê tệp tin và dung lượng
├── wrist_camera.mp4               # Video MP4 đồng bộ tốc độ khung hình
├── wrist_camera_depth/            # Thư mục chứa chuỗi ảnh Depth PNG 16-bit
│   ├── depth_info.json            # Thống kê min/max/mean khoảng cách mm
│   ├── frame_00000.png            # Ảnh uint16 bảo toàn độ chính xác mm
│   └── ...
├── action.csv                     # Lệnh điều khiển góc khớp và ngàm kẹp
├── qpos.csv                       # 6 góc khớp thực tế có header ngữ nghĩa
├── qvel.csv                       # Vận tốc 6 góc khớp
├── tcp_pose.csv                   # Tọa độ Descartes TCP thực tế (m, rad)
├── vr_controller_pose.csv         # Tọa độ tay cầm VR (pos_x..z, quat_x..w)
├── target_tcp_pose.csv            # Tọa độ TCP mục tiêu ra lệnh
├── tcp_force.csv                  # Lực và mô-men tiếp xúc đầu công cụ
├── ft_raw_wrench.csv              # Dữ liệu đo thô từ cảm biến F/T
├── gripper.csv, gripper_cmd.csv   # Dữ liệu ngàm kẹp
├── currents.csv, joint_voltages.csv
├── temperatures.csv, payload.csv  # Trạng thái nhiệt độ và tải trọng
└── timestamps_ms.csv              # Dấu thời gian từng bước
```

---

### 7.3. Trực Quan Hóa & Kiểm Định Nhanh Đa Phương Thức (Data Visualization & Sanity Check)

Công cụ độc lập [`core/visualize_data.py`](file:///home/nguyen/ur_ws/pyroki_ursim/core/visualize_data.py) hoạt động ở chế độ chỉ đọc (**Read-only**), giúp các kỹ sư AI nhanh chóng kiểm tra chất lượng dữ liệu (sanity check) trước khi nạp vào mạng nơ-ron học sâu (Deep Learning).

![Ảnh Kết Quả Trực Quan Hóa Dữ Liệu HDF5 2x2](assets/episode_10_visualization.png)

Bố cục tổng hợp 4 biểu đồ trực quan trên cùng một giao diện (Layout 2x2):
- **Top-Left (Kinematics):** Đồ thị 2D thể hiện sự biến thiên của 6 góc khớp (`qpos`).
- **Top-Right (Actions):** Đồ thị 2D thể hiện tín hiệu mở/đóng ngàm kẹp (`gripper_cmd` & phản hồi thực tế `gripper`).
- **Bottom-Left (Trajectory):** Đồ thị 3D không gian biểu diễn quỹ đạo di chuyển của đầu búp kẹp (`tcp_pose`), đánh dấu rõ điểm bắt đầu (🟢 Start) và điểm kết thúc (🔴 End).
- **Bottom-Right (Vision):** Khung hình ảnh màu RGB từ camera cổ tay (`wrist_camera`) tại thời điểm giữa phiên ghi.

#### Cài đặt thư viện bổ sung:
```bash
pip install matplotlib
```

#### Cú pháp dòng lệnh thực thi:
```bash
# 1. Hiển thị cửa sổ đồ thị 2x2 tương tác (đồng thời tự động xuất ảnh PNG cùng thư mục):
python core/visualize_data.py --file logs/ai_training_data/episode_0.hdf5

# 2. Chế độ Headless / Server (chỉ xuất và lưu file ảnh PNG tĩnh mà không mở cửa sổ GUI):
python core/visualize_data.py --file logs/ai_training_data/episode_0.hdf5 --no-gui

# 3. Chỉ định đường dẫn lưu ảnh PNG và chọn khung hình camera quan sát:
python core/visualize_data.py --file logs/ai_training_data/episode_0.hdf5 --save /duong_dan/anh_kiem_dinh.png --frame-idx 15
```

---

### 7.4. Kiểm Định & Tái Lập Quỹ Đạo (verify_dataset.py & replay_episode.py)

```bash
# 1. Kiểm định chất lượng tập dữ liệu (quét NaN, Inf, kiểm tra dt jitter, kích thước ảnh)
python core/verify_dataset.py

# 2. Tái hiện quỹ đạo Offline Dry-Run (Xem mô phỏng trên Viser, không gửi lệnh sang robot)
python core/replay_episode.py --file logs/ai_training_data/episode_0.hdf5

# 3. Tái hiện quỹ đạo Live Hardware Replay (Gửi lệnh thực thi trực tiếp trên URSim / Robot thật)
python core/replay_episode.py --file logs/ai_training_data/episode_0.hdf5 --execute --ip 10.42.0.242
```

---

### 7.5. Bộ Kiểm Thử Tự Động (pytest)

Chạy bộ kiểm thử để xác thực toàn diện các luồng viễn thao, trích xuất dữ liệu, trực quan hóa và giao diện UI:

```bash
# Chạy toàn bộ các bài kiểm thử tự động cốt lõi
pytest tests/test_extract_hdf5.py tests/test_visualize_data.py tests/test_vr_record_and_ui.py tests/test_vr_key_integration.py tests/test_base_frame_rotation.py -v
```

---

## 8. Bảng Thông Số Kỹ Thuật Điều Khiển Chính

| Thông số | Giá trị vận hành | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- |
| **Tần số vòng lặp chính** | 20 Hz (`dt = 0.05`s) | Cân bằng tối ưu giữa băng thông RTDE servoJ và thời gian tính toán IK trên CPU. |
| **Tần số Camera Stream** | 20 FPS | Luồng nền độc lập, triệt tiêu hoàn toàn giật lag cho vòng lặp RTDE 20Hz. |
| **Thời gian giải PyRoki IK (CPU)** | `~1.0` ms | Giải trên CPU bằng JAX JIT sau warm-up, Levenberg-Marquardt qua `jaxls`. |
| **Chi phí dáng mồi IK (`rest_cost`)** | `weight = 0.02` | Neo giữ nghiệm bám theo hạt giống `seed` của frame trước, đảm bảo tính liên tục. |
| **ServoJ Lookahead Time** | `0.12` s | Nội suy dự đoán đa chu kỳ chuẩn Universal Robots, triệt tiêu vi rung giật. |
| **ServoJ Proportional Gain** | `250` | Độ cứng vị trí tối ưu trên robot vật lý, giúp chuyển động lướt êm ái không bị gắt. |
| **Vận tốc khớp tối đa (Rate Limiter)** | `1.0` rad/s ($\approx 57.3^\circ$/s) | Scaled Vector Limiter co giãn đồng bộ cả 6 khớp, bảo toàn hướng vector Descartes. |
| **Bán kính làm việc tối đa ($R_{max}$)** | `0.52` m | Khống chế phạm vi vươn phù hợp tầm với 500mm của UR3e. |
| **Độ cao an toàn tối thiểu ($Z_{min}$)** | `0.035` m ($35\text{mm}$) | Chống cánh tay và ngàm kẹp va chạm mặt bàn ($Z \ge 35$mm). |
| **Ngưỡng Singularity Warning & Danger** | $\sigma_{warn} = 0.042$, $\sigma_{danger} = 0.028$ | Continuous Damping hãm mượt và chặn lệnh khi áp sát điểm kỳ dị Jacobian. |
| **Dung sai vị trí IK (`pos_err`)** | $< 0.018$ m ($18\text{mm}$) | Đảm bảo tỷ lệ hội tụ IK vùng biên vươn xa đạt $>98\%$. |
| **Ngưỡng phát hiện vặn xoắn khớp** | `2.2` rad | Chặn các bước nhảy target đột ngột do đảo nhánh IK gần điểm kỳ dị. |
| **Giới hạn khớp khuỷu (`elbow_joint`)** | `[+0.10, +2.40]` rad | Chống cùi chỏ tự gập sát đâm vào thân robot. |
| **Giới hạn khớp nâng vai (`shoulder_lift`)** | `[-2.65, +0.20]` rad | Ngăn ngừa rủi ro đâm bắp tay xuống mặt bàn. |
| **Thuật toán quy hoạch Home** | URScript 2 pha `movej` | Chạy trực tiếp ở tần số 500Hz trên URControl qua cổng Secondary 30002. |
| **Khôi phục Teleop sau khi về Home** | Tự động (Zero-Click) | Tự động kết nối lại RTDE và reset mỏ neo tay cầm VR ngay khi robot về đích. |
| **Vùng chết tĩnh tay cầm VR** | $3$ mm (`0.003` m) | Loại bỏ vi rung tay sinh học với cơ chế trừ mềm (`POS_DEADBAND`). |
| **Tỷ lệ truyền chuyển động tay (`Scale`)** | `0.75` (Mặc định) | Điều chỉnh linh hoạt qua thanh trượt Viser UI ($0.25\times - 3.0\times$). |
| **Độ lệch tay cầm tối đa (`max_delta_m`)** | `0.22` m | Khống chế bước dịch chuyển Descartes tối đa mỗi nhịp bóp cò. |
| **Ngưỡng thời gian Auto-Snap** | $1.2$ s | Tự động kéo tọa độ về đầu tay máy và reset seed khi bị kẹt tư thế. |
| **Khóa trục xoay cổ tay (Soft Snapping)** | $<55\%$ (cắt về 0), $55-75\%$ (làm mềm) | Triệt tiêu góc xoay phụ khi gập/chúc cổ tay, bảo vệ trục quay chính xác. |
| **Cơ chế kích hoạt ngàm kẹp** | Phím vật lý X (Toggle) | Bấm 1 lần kẹp, bấm lần nữa nhả kèm rung Haptic. |
| **Cơ chế kích hoạt ghi dữ liệu** | Phím vật lý A (Toggle) | Bấm 1 lần bắt đầu ghi, bấm lần nữa dừng và lưu file HDF5. |
| **Kiến trúc ghi dữ liệu HDF5** | Chunked HDF5 + RAM RingBuffer | Xả định kỳ xuống đĩa; triệt tiêu hoàn toàn rò rỉ bộ nhớ và lỗi OOM. |
| **Độ lệch thời gian phần cứng** | $< 1.5$ ms | RealSense ASIC chip hardware timestamp đồng pha với RTDE controller clock. |
| **Tốc độ trích xuất HDF5** | $> 500$ steps/s | Giải nén siêu tốc sang video MP4, ảnh PNG 16-bit và file CSV ngữ nghĩa. |
| **Tiến trình ADB Forwarding Watcher** | Chu kỳ quét 2.5 s | Tự động thiết lập đường hầm USB reverse cho WebXR khi cắm cáp. |

---

## 9. Tự Động Đóng Gói Dự Án (Build Script)

Hệ thống tích hợp sẵn script tự động hóa [`build_zip.sh`](file:///home/nguyen/ur_ws/pyroki_ursim/build_zip.sh) để đóng gói toàn bộ dự án thành file `Teleop_Quest_Project.zip` gửi cho đối tác hoặc triển khai sang máy tính mới:
- **Tự động xuất Docker Image sẵn (`teleop_quest_image.tar`):** Đóng gói toàn bộ Image Linux + Python 3.11 + PyRoki + Torch CPU để người nhận có thể chạy 100% Offline mà không phải chờ tải mạng 1-3 phút.
- **Giữ lại file `.env`:** Bảo lưu cấu hình địa chỉ IP Robot (`ROBOT_IP`) và Camera (`CAM_URL`) hiện tại để người nhận có thể chạy ngay lập tức mà không cần sửa cấu hình.
- **Loại trừ rác & môi trường ảo nặng:** Tự động loại bỏ `.venv/`, `venv/`, `.git/`, `__pycache__/`, `*.pyc`, `*.log`.
- **Tự động làm sạch:** Xóa file zip cũ trước khi nén để tránh xung đột dung lượng.

### Lệnh thực thi:
Bất cứ khi nào bạn cập nhật code và muốn xuất file zip đóng gói mới:

```bash
./build_zip.sh
```

Người nhận chỉ cần giải nén file `Teleop_Quest_Project.zip` và xem hướng dẫn tại [`HUONG_DAN_CHAY.md`](file:///home/nguyen/ur_ws/pyroki_ursim/HUONG_DAN_CHAY.md):
```bash
# 1. Nạp image (chỉ 1 lần trên máy mới - mất 2 giây):
docker load -i teleop_quest_image.tar

# 2. Khởi chạy (chạy tức thì trong 1 giây):
docker compose up -d
```
