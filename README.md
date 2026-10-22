# UR3e Teleoperation qua Meta Quest 3S, Viser 3D & PyRoki CPU IK

Hệ thống điều khiển viễn thao thời gian thực (Real-time Teleoperation) cho cánh tay robot công nghiệp Universal Robots UR3e sử dụng kính thực tế ảo Meta Quest 3S (WebXR) và chuột trên giao diện Web 3D Viser. Hệ thống giải bài toán động học nghịch (Inverse Kinematics) siêu tốc thuần CPU bằng Google PyRoki kết hợp JAX và `jaxls` (độ trễ ~1ms sau JIT warm-up, hoàn toàn không cần GPU NVIDIA/CUDA), đồng thời hỗ trợ thu thập dữ liệu trình diễn (Demonstrations) chuẩn HDF5 phục vụ huấn luyện mô hình Robot AI (Imitation Learning, Diffusion Policy, ACT).

---

## 1. Tính Năng Nổi Bật

1. **Điều khiển Viễn Thao Thời Gian Thực (20Hz RTDE servoJ & Tách Luồng Camera):**
   - Đồng bộ chuyển động 6DoF từ tay cầm Meta Quest 3S hoặc chuột máy tính (Viser 3D Transform Controls Target Widget) sang robot UR3e với độ trễ cực thấp.
   - Bộ thực thi `ServoJExecutor` sử dụng tham số tối ưu chuẩn công nghiệp: `dt = 0.05`s, `lookahead_time = 0.12`s, `gain = 250`.
   - Cơ chế **Scaled Vector Rate Limiter** co giãn đồng bộ cả 6 khớp khi vượt ngưỡng vận tốc, bảo toàn 100% hướng vector trong không gian Descartes và loại bỏ hoàn toàn hiện tượng méo quỹ đạo/giật cục.
   - Kiến trúc đa luồng (`CameraStreamPipeline`): Tách riêng luồng render và truyền stream camera cổ tay, triệt tiêu hoàn toàn jitter timing cho vòng lặp điều khiển RTDE 20Hz.
   - **Cơ chế Mặt Đồng Hồ (True Mechanical Clock Navigation):**
     - **Phải/Trái:** Chỉ quay kim đồng hồ (khớp đế $q_0$) quanh trục Base Z đồng thời tự động bù góc xoay trục Z ngàm kẹp giữ nguyên hướng đối với bàn máy.
     - **Tiến/Lùi:** Duỗi dài hoặc thu ngắn bắp tay theo đúng phương vươn cơ học $\vec{u}_{reach}$ của hướng $q_0$.
     - **Cao/Thấp:** Nâng hạ độ cao Z thẳng đứng trên mặt bàn.
     - **Hiệu chuẩn hướng Tiến theo bàn máy bằng nút B:** Bắt cạnh lên nút B lấy góc yaw của kính (`head_quat`) làm hệ quy chiếu bàn máy cố định (`_desk_u_fwd` và `_desk_u_right`).
     - **Vùng chết tĩnh 3mm (`POS_DEADBAND = 0.003`m):** Triệt tiêu hoàn toàn vi rung sinh học khi bóp giữ phím.
   - Chế độ điều khiển VR phân tách chức năng:
     - **Chỉ bóp Cò (Trigger):** Tịnh tiến theo cơ chế Mặt Đồng Hồ (khóa góc xoay).
     - **Chỉ bóp Grip (Nút hông):** Xoay đầu kẹp theo hệ tọa độ đầu công cụ (Tool-Centric mapping, khóa tịnh tiến).
     - **Bóp cả Trigger + Grip:** Chuyển động phối hợp 6DoF đồng thời.
     - **Cơ chế Smart Clutch:** Tự động chốt mỏ neo mới (Reset Anchor) ngay khi thay đổi nút bấm, triệt tiêu hoàn toàn lỗi nhảy vọt tọa độ (teleport).
   - Chế độ điều khiển Chuột trên PC: Tương tác trực tiếp bằng cách kéo các trục tịnh tiến hoặc vòng xoay góc của khối Target Widget trên trình duyệt Viser.

2. **Động Học Nghịch Siêu Tốc Thuần CPU (PyRoki + JAX/jaxls) & Tối Ưu Không Gian Khớp:**
   - Giải bài toán IK thời gian thực trên CPU bằng PyRoki kết hợp `jaxls` LeastSquaresProblem và JAX JIT compile (độ trễ siêu tốc ~1ms sau warm-up), loại bỏ 100% sự phụ thuộc vào card đồ họa NVIDIA hoặc CUDA runtime.
   - Đồ thị tối ưu hóa chi phí hoàn chỉnh:
     - `pose_cost_analytic_jac`: Bám vị trí (weight 50.0) và định hướng mục tiêu (weight 20.0).
     - `limit_cost`: Ràng buộc giới hạn khớp (weight 0.5).
     - `rest_cost`: Chi phí neo giữ dáng mồi (weight 0.02) kết hợp hạt giống chuẩn `q_var.with_value(seed)` đảm bảo liên tục nghiệm giữa các frame.
   - Chuẩn hóa giới hạn góc khớp: Khớp khuỷu (`elbow_joint`) trong dải an toàn $[+0.10, +2.40]$ rad ($[+5.7^\circ, +137.5^\circ]$) loại bỏ nguy cơ gập sát tự đâm vào thân robot đồng thời cho phép duỗi vươn xa mép bàn, khớp nâng vai `shoulder_lift_joint` trong $[-2.65, +0.20]$ rad chống chúi đập bàn, khớp cổ tay `wrist_1_joint` trong $[-3.20, +0.10]$ rad và `wrist_2_joint` trong $[-2.95, -0.20]$ rad chống bẻ ngược vào bắp tay.
   - Thuật toán `normalize_to_current`: Tự động tìm nghiệm góc khớp tương đương gần nhất với cấu hình hiện tại trong dải an toàn, triệt tiêu hoàn toàn hiện tượng vặn xoắn cổ tay hoặc đảo nhánh IK bất thường.

3. **Luồng Stream Camera Cổ Tay Trực Tiếp (Real-time 20 FPS) & Driver Đa Phương Thức (USB / Ethernet):**
   - Khung "CAMERA STREAM" trên giao diện WebXR Cockpit hiển thị trực tiếp góc nhìn từ camera cổ tay robot ở tần số thời gian thực 20 FPS.
   - Kiến trúc Driver Đa Phương Thức (`camera_driver.py`) hỗ trợ chuyển đổi linh hoạt qua cờ dòng lệnh CLI:
     - **Camera mô phỏng 3D (`SyntheticWristCamera`):** Kết xuất đồ họa màu RGB 640x480, góc nhìn Eye-in-Hand từ đầu tay máy quan sát quả cà chua và khay chứa.
     - **Camera USB vật lý thật (`RealWristCamera`):** Đọc trực tiếp luồng hình ảnh từ camera Intel RealSense (D435/D455) hoặc USB webcam cắm trực tiếp vào máy chủ (`--cam-source usb`).
     - **Camera qua cáp mạng Ethernet (`EthernetWristCamera`):** Nhận stream qua socket ZMQ Zero-lag (chế độ `CONFLATE=1`) hoặc HTTP stream từ máy tính nhúng gắn trên robot (`--cam-source eth --cam-url tcp://192.168.1.100:5555`).
   - Tích hợp mô hình ngàm kẹp + camera thực tế (`robot_camera_mount_1.stl`, `d455.stl`).
   - Tự động tích hợp thông tin trạng thái OSD (trạng thái ngàm kẹp Gripper) và truyền đồng thời qua kênh WebSocket và MJPEG `/video_feed`.

4. **Mô Phỏng Gắp/Thả 3D & Cơ Chế Toggle Gripper:**
   - Mô phỏng cơ chế hút dính vật thể quả cà chua 3D khi ngàm tiếp cận đúng vị trí (khoảng cách 3D $< 9.5$cm hoặc lệch XY $< 5.5$cm kèm độ cao Z $< 13$cm).
   - Tự động nhận diện khi đưa vật thể vào khay chứa và ghi nhận sự kiện hoàn thành nhiệm vụ (`PLACE_SUCCESS`).
   - **Cơ chế Toggle Gripper (Nhấn Nhả Chuyển Trạng Thái):** Người dùng chỉ cần nhấn 1 lần nút A (tay phải) hoặc X (tay trái) để kẹp/hút vật thể, và nhấn lại lần nữa để thả vật thể. Không cần phải giữ chặt ngón tay liên tục, giảm mỏi tay tối đa và có rung haptic xác nhận mỗi lần chuyển trạng thái.

5. **Khóa Trục Xoay Cổ Tay Thông Minh (Soft Rotation Snapping):**
   - Giải quyết triệt để hiện tượng quay đa trục ký sinh khi gập hoặc chúc cổ tay (lỗi vô thức lắc ngang Yaw hoặc xoắn Roll khi đang muốn gập Pitch).
   - Áp dụng bộ lọc đường cong mềm tỷ lệ ba vùng theo độ lớn vector xoay:
     - **Tỷ lệ xoay phụ $< 55\%$ (Soft Snapping Low):** Triệt tiêu hoàn toàn góc xoay phụ về 0, khóa chuyển động quay chuẩn xác trên một trục duy nhất.
     - **Tỷ lệ xoay phụ $55\% - 75\%$:** Làm mềm chuyển tiếp bằng đường cong Hermite Smoothstep, không gây giật góc.
     - **Tỷ lệ xoay phụ $> 75\%$ (Soft Snapping High):** Cho phép quay tự do 3D khi người dùng chủ động vặn xoay tổ hợp nhiều hướng.

6. **Tự Động Kích Hoạt Docker URSim & Tiến Trình Nền ADB Watcher:**
   - Khi khởi động chế độ mô phỏng (`127.0.0.1`), hệ thống tự động kiểm tra cổng 29999. Nếu URSim chưa bật, chương trình tự động chạy `docker start ursim` mà người dùng không cần mở thêm terminal phụ.
   - Tích hợp tiến trình ngầm (`start_adb_watcher`) tự động kiểm tra cáp USB nối với kính Meta Quest mỗi 2.5 giây. Khi phát hiện kính đã cắm, hệ thống tự động chạy lệnh reverse toàn bộ cổng (`8443`, `8444`, `8445`, `8080`) vào `localhost`.

7. **Cơ Chế Tự Động Thu Hồi Tọa Độ XYZ (Auto-Snap on Stuck) & Reset Seed Thoát Kẹt:**
   - Khi robot gặp tình trạng kẹt tư thế (vặn xoắn khớp) hoặc vượt quá tầm với kéo dài quá 1.2 giây (`snap_threshold = 1.2`), hệ thống tự động rung phản hồi Haptic tới tay cầm và giật tọa độ Widget 3D về đúng đầu tay máy robot thật (`fk_res`).
   - Tự động gán lại hạt giống giải IK `teleop._last_valid_q = list(q_real)`, giúp thuật toán lập tức thoát kẹt và sẵn sàng nhận chuyển động mới từ người vận hành.

8. **Tự Động Dọn Dẹp Kịch Bản Ngầm & Giải Phóng Thanh Ghi RTDE:**
   - Bộ thực thi `ServoJExecutor` và controller `PyrokiURSim` tích hợp cơ chế tự động gửi lệnh `stop\n` tới Dashboard Server (Port 29999) trước khi kết nối RTDE.
   - Loại bỏ hoàn toàn lỗi xung đột thanh ghi (`One of the RTDE input registers are already in use!`) hoặc treo tiến trình khi khởi động lại sau các lần thoát đột ngột.
   - Tự động fallback sang dải thanh ghi trên (`FLAG_UPPER_RANGE_REGISTERS`) nếu kết nối trên robot thật có cấu hình fieldbus (Profinet/EtherNet-IP).

9. **Quy Hoạch Trở Về Home An Toàn Không Giật & Chống Rung Lắc Hình Ảnh:**
   - Quy hoạch 2 pha an toàn thuần CPU qua URScript `movej` chạy ở tần số 500Hz trực tiếp trong bộ điều khiển Universal Robots qua cổng Secondary Port 30002: Pha 1 nâng khuỷu và vai lên cao tránh quẹt mặt bàn, Pha 2 di chuyển đồng bộ về vị trí Home an toàn.
   - Tự động hòa trộn góc bo cung (`blend radius = 0.03`m giữa các waypoint), triệt tiêu 100% hiện tượng dừng giật cơ học.
   - Vòng lặp giám sát cập nhật dáng robot trên Viser 3D ở tần số 25Hz mượt mà theo góc khớp thực của robot, loại bỏ hoàn toàn glitch hình ảnh chập chờn.

10. **Tự Động Tái Kích Hoạt Teleop Sau Khi Về Home (Zero-Click Auto Reconnect):**
    - Tích hợp cờ giám sát `_was_navigating_home`, tự động tái kết nối RTDE `executor.connect()`, đặt lại mỏ neo tay cầm VR `pose_source.reset_anchor()` và đưa hệ thống vào trạng thái sẵn sàng điều khiển ngay khi robot về tới Home mà người dùng không cần phải click tắt/bật lại checkbox thủ công.

11. **Giao Diện Viser 3D Cockpit Tối Giản (Bố Cục Lưới 2x2):**
    - Gom cụm 4 thao tác điều khiển cốt lõi thành ma trận 2x2 tiện dụng ngay trên đầu bảng điều khiển:
      - Hàng 1: `Bật Teleop` (Checkbox) & `Gripper (Bóp Kẹp)` (Checkbox).
      - Hàng 2: `Record AI Episode` (Checkbox) & `Hủy Episode (Discard)` (Checkbox dạng tự nhả sau 200ms).
    - Tối ưu không gian hiển thị, giúp tập trung tối đa vào không gian làm việc 3D và khung camera stream.

12. **Thu Thập & Quản Lý Dữ Liệu AI Chuẩn HDF5:**
    - Ghi lại chuỗi thao tác đầy đủ: góc khớp (`qpos`), vận tốc (`qvel`), lệnh điều khiển (`action`), trạng thái ngàm kẹp (`gripper`), ảnh camera cổ tay (`images/wrist_camera`), timestamp ở tần số 20Hz.
    - Tự động đánh dấu kết quả thành công/thất bại (`success=True/False`).

13. **Bộ Công Cụ Kiểm Định & Tái Lập Quỹ Đạo (Data Sanity & Replay):**
    - `verify_dataset.py`: Quét toàn bộ file HDF5 kiểm tra giá trị NaN/Inf, độ lệch chu kỳ lấy mẫu (dt jitter), kích thước tensor hình ảnh.
    - `replay_episode.py`: Tái hiện lại quỹ đạo trình diễn ở chế độ Offline Dry-Run hoặc Live Replay gửi lệnh trực tiếp sang robot.

14. **Các Lớp Bảo Vệ An Toàn (Safety Gate & Singularity Guard):**
    - **Giám sát điểm kỳ dị (Singularity Guard):** Đánh giá ma trận Jacobian thời gian thực, tự động chặn lệnh nếu giá trị suy biến nhỏ nhất $\sigma_{min} < 0.035$.
    - **Chống chúi cắm bắp tay:** Giới hạn góc nâng vai `shoulder_lift_joint` $\le +0.25$ rad để ngăn ngừa rủi ro đâm bắp tay xuống mặt bàn.
    - **Giới hạn tốc độ và gia tốc (Scaled Vector Rate Limiter):** Kẹp vận tốc góc khớp tối đa $\le 0.6$ rad/s ($\approx 34.4^\circ$/s) để tái hiện chuyển động êm ái, an toàn.
    - **Phát hiện lật nhánh IK:** Chặn các bước nhảy góc khớp đột ngột (`max_jump > 1.5` rad).
    - **Giám sát mất kết nối (Timeout Watchdog):** Tự động dừng khẩn cấp robot qua `servoStop` nếu mất tín hiệu VR quá thời gian timeout.

15. **Kiến Trúc Mô-đun Hóa Tinh Gọn (Single Responsibility Architecture):**
    - Phân tách độc lập các khối chức năng: `teleop/pyroki_solver.py` (bộ giải PyRoki IK/FK CPU), `bootstrap.py` (khởi tạo hệ sinh thái phần mềm và kết nối phần cứng), `camera_pipeline.py` (luồng đa nhiệm camera & OSD), `camera_driver.py` (driver camera ảo và RealSense), `main_quest.py` (bộ điều phối trung tâm) và `teleop_quest/__init__.py` xuất khẩu các interface tiêu chuẩn.

---

## 2. Kiến Trúc & Luồng Hoạt Động Của Hệ Thống

Toàn bộ hệ thống điều khiển viễn thao vận hành theo luồng khép kín thời gian thực ở tần số 20Hz (chu kỳ 50ms), kết nối đồng bộ giữa giao diện điều khiển (Chuột trên PC / Kính Meta Quest 3S), máy chủ giải thuật PyRoki CPU và bộ điều khiển robot Universal Robots UR3e qua các cổng mạng chuyên dụng: cổng 29999 (Dashboard Server), cổng 30002 (URScript Secondary Interface), cổng 30003/30004 (RTDE ServoJ & Receive Interface), cổng 8443 (HTTPS WebXR), cổng 8444 (WSS Quest WebXR) và cổng 8445 (WS PC Local Preview).

### 2.1. Sơ Đồ Khối Luồng Dữ Liệu

```text
[Tay cầm Meta Quest 3S]               [Chuột PC trên Viser 3D]
  (Trigger/Grip/Nút A/X/B/Y)             (Transform Controls Widget)
          │                                        │
          ▼ (WebSocket WSS 8444)                   │
[QuestServer & DeltaPoseSource]                    │
  (Cơ chế Mặt Đồng Hồ, Nút B Desk-Aligned, Clutch) │
          │                                        │
          └───────────────┬────────────────────────┘
                          ▼
              [Target Widget 3D (Viser)]
           (Tọa độ đích 6DoF: X, Y, Z, Qw, Qx, Qy, Qz)
                          │
                          ▼
              [PyRoki IK Solver (JAX CPU)]
           (JIT warm-up ~1ms, jaxls, rest_cost -> q_target)
                          │
                          ▼
              [SafetyGate & SingularityGuard]
           (Jacobian sigma_min >= 0.035, Rate Limiter)
                          │
                          ▼
              [ServoJExecutor (RTDE Cổng 30003/30004)]
           (servoJ với lookahead_time = 0.12s, gain = 250)
                          │
                          ▼
              [Robot UR3e (URSim / Robot thật)]
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
[DataLogger HDF5]             [CameraStreamPipeline]
(q, qd, tcp, action, nhãn)    (20 FPS độc lập, OSD Watermark, Viser & Quest)
```

### 2.2. Cơ Chế Máy Trạng Thái (Edit Mode vs. Drive Mode)

Hệ thống thiết lập 2 chế độ vận hành độc lập, hỗ trợ đồng bộ hai chiều tức thì giữa kính thực tế ảo và màn hình máy tính:

1. **Chế độ Cầm lái (Drive Mode - Mặc định khi vào VR):**
   - **Mục đích:** Vừa quan sát toàn bộ giao diện Cockpit HUD (mô hình 3D, camera cổ tay, tọa độ thời gian thực) nổi trước mắt, vừa tự do điều khiển cánh tay robot bằng tay cầm mà không bị vướng chuột ảo.
   - **Tương tác:** Giao diện HUD tự động kích hoạt chế độ xuyên thấu triệt để (`pointer-events: none !important`). Hệ điều hành Meta Horizon OS sẽ tắt hoàn toàn tia laser 2D. Lúc này mọi thao tác bóp cò Trigger, bóp Grip và bấm nút A/X được chuyển 100% sang điều khiển robot 6DoF, không còn hiện tượng "di chuột trên màn hình".
   - **Hiệu chuẩn hướng bàn máy:** Nhấn nút **B** (tay phải) hoặc **Y** (tay trái) khi nhìn về phía trước mép bàn để lập tức khóa góc yaw đầu làm trục hướng Tiến chuẩn cho cơ chế Mặt Đồng Hồ.
   - **Phản hồi thời gian thực:** Thẻ `ROBOT POSE` cập nhật liên tục tọa độ Position (X, Y, Z tính bằng mét) và Orientation (Qx, Qy, Qz, Qw) từ vòng lặp RTDE 20Hz của cánh tay robot UR3e thực tế. Thẻ `VR CONTROLLER` hiển thị trạng thái các nút bấm.
   - **Cách kích hoạt:** Mặc định tự động bật khi người dùng bấm *START IMMERSIVE VR*, hoặc bấm phím cứng **B** (tay phải) / **Y** (tay trái) trên tay cầm, hoặc click nút *Lock UI* trên menu HUD.

2. **Chế độ Chỉnh sửa Giao diện (Edit Mode):**
   - **Mục đích:** Thao tác cài đặt thông số, dùng tia laser chuột để xoay camera xem mô hình 3D trong khung Viser hoặc click các nút bấm trên trang web.
   - **Cách chuyển sang Edit Mode:** Khi đang ở trong VR, chỉ cần **bấm nhả phím B (hoặc Y)** trên tay cầm. Nút menu góc trên bên trái chuyển sang màu cam (`Lock UI (Drive Mode)`), thẻ Mode chuyển sang màu cam (`Edit Mode`), và tia laser chuột 2D xuất hiện.
   - **An toàn:** Cánh tay robot được khóa trạng thái (`teleop.enabled = False`). Mọi thao tác bóp cò hoặc vung tay trong chế độ này đều không phát lệnh di chuyển robot, tránh va chạm ngoài ý muốn.
   - **Quay lại lái robot:** Sau khi thao tác giao diện xong, chỉ cần **bấm nhả lại phím B (hoặc Y)**. Tia laser lập tức biến mất và hệ thống quay lại Drive Mode để tiếp tục lái robot mượt mà mà không cần thoát khỏi VR.

### 2.3. Quy Trình Vòng Lặp Điều Khiển Thời Gian Thực (Chu Kỳ 20Hz - 50ms)

Trong mỗi chu kỳ 50ms của [main_quest.py](file:///home/nguyen/ur_ws/pyroki_ursim/teleop_quest/main_quest.py), hệ thống thực thi tuần tự 6 công đoạn:

1. **Bước 1: Thu thập & Lọc tín hiệu điều khiển (`DeltaPoseSource` / `WidgetPoseSource`):**
   - Khi dùng kính Quest: Áp dụng cơ chế Mặt Đồng Hồ (Cò trỏ Trigger phụ trách di chuyển Tiến/Lùi dọc theo phương vươn $q_0$, Phải/Trái xoay góc đế $q_0$ kèm bù góc Z ngàm kẹp, Cao/Thấp nâng hạ Z; Cò báng Grip phụ trách xoay hướng ngàm; nút A/X chuyển đổi trạng thái ngàm kẹp Toggle; nút B hiệu chuẩn hướng bàn máy).
   - Áp dụng bộ lọc Soft Rotation Snapping triệt tiêu dao động góc quay phụ khi gập cổ tay và deadband 3mm chống rung tĩnh.
   - Khi dùng chuột: Tọa độ đích được cập nhật trực tiếp từ khối Target Widget trên không gian 3D của Viser.
   - Giới hạn Workspace an toàn: $X, Y \in [-0.49, 0.49]$m, $Z \in [0.00, 0.58]$m (an toàn tuyệt đối không va chạm mặt bàn), bán kính vươn XY kẹp trong $[0.10, 0.485]$m bảo vệ tầm với 500mm của UR3e.

2. **Bước 2: Giải ngược động học siêu tốc trên CPU (`PyrokiIKSolver`):**
   - Nhận tọa độ mục tiêu 6DoF từ Widget 3D, giải bài toán IK phi tuyến bằng PyRoki + `jaxls` trên CPU trong khoảng ~1ms sau JIT warm-up.
   - Áp dụng hàm chi phí `rest_cost` (weight 0.02) kết hợp hạt giống `q_seed` từ frame trước qua `q_var.with_value(seed)`, đảm bảo nghiệm mượt mà không nhảy nhánh.
   - Chuẩn hóa góc khớp `normalize_to_current` và xác thực giới hạn góc an toàn `validate_joints`.

3. **Bước 3: Kiểm định an toàn và điểm kỳ dị (`SafetyGate` & `SingularityGuard`):**
   - Đánh giá ma trận Jacobian của cấu hình góc khớp mục tiêu, đo lường giá trị suy biến nhỏ nhất $\sigma_{min}$.
   - Nếu $\sigma_{min} < 0.035$ (tư thế tiến sát điểm kỳ dị), cổng an toàn lập tức chặn lệnh, phát xung rung cảnh báo về tay cầm VR và giữ robot đứng yên.
   - Khối Scaled Vector Rate Limiter kiểm tra và kẹp vận tốc góc khớp tối đa $\le 0.6$ rad/s để tạo chuyển động êm dịu, không giật cục.

4. **Bước 4: Thực thi lệnh chuyển động thời gian thực (`ServoJExecutor`):**
   - Đảm bảo giải phóng thanh ghi RTDE thông qua lệnh Dashboard `stop\n`.
   - Truyền gói dữ liệu 6 góc khớp mục tiêu $q_{target}$ qua giao thức RTDE (Cổng 30003/30004) bằng lệnh `servoJ` với `lookahead_time = 0.12`s và `gain = 250`.
   - Robot phản hồi bám sát tức thì với độ trễ chuyển động thực tế dưới 10ms.

5. **Bước 5: Mô phỏng gắp nhặt & Ghi tập dữ liệu AI (`GraspSimulator` & `DataLogger`):**
   - Kiểm tra tương tác hình học giữa đầu ngàm kẹp và quả cà chua mục tiêu. Khi tiếp cận đúng phạm vi và ngàm kẹp đóng, vật thể được gắn dính vào đầu kẹp và gửi phản hồi Haptic thành công.
   - Khi cờ ghi hình kích hoạt (`ui.cb_record`), module `DataLogger` lưu trữ đồng bộ trạng thái góc khớp thực tế, vận tốc, tọa độ TCP, lệnh điều khiển, trạng thái kẹp và ảnh camera thô vào tệp HDF5 theo chu kỳ cố định 20Hz.

6. **Bước 6: Thu nhận hình ảnh camera cổ tay độc lập (`CameraStreamPipeline`):**
   - Luồng nền camera thu nhận ảnh ở tần số 20 FPS từ camera ảo 3D hoặc camera vật lý thật RealSense D435/D455 gắn trên cổ tay (RGB 640x480).
   - Tích hợp lớp phủ OSD hiển thị trạng thái ngàm kẹp, nén định dạng JPEG chất lượng cao và phát trực tiếp tới cả sidebar Viser và WebXR Cockpit qua WebSocket và MJPEG.

---

## 3. Danh Sách Thư Viện & Môi Trường Cài Đặt

### Yêu Cầu Hệ Thống
- **Hệ điều hành:** Ubuntu 20.04 hoặc 22.04 LTS (x86_64).
- **Phần cứng:** CPU đa nhân (Intel Core i5/i7/i9 hoặc AMD Ryzen 5/7/9). Không yêu cầu card đồ họa rời NVIDIA hay GPU CUDA.
- **Bộ điều khiển Robot:** Trình giả lập URSim trên Docker hoặc Cánh tay UR3e thật (Cổng 29999 Dashboard, 30002 Secondary URScript, 30003/30004 RTDE).

### Bảng Thư Viện Python

| Thư viện | Phiên bản | Vai trò trong hệ thống |
| :--- | :--- | :--- |
| `jax[cpu]`, `jaxlie` | 0.4.20+, 1.0.0+ | Tính toán đại số vi phân Lie và biên dịch JIT siêu tốc trên CPU. |
| `jaxls` | git (brentyi/jaxls) | Bộ giải bình phương tối thiểu phi tuyến (Levenberg-Marquardt) cho IK. |
| `pyroki` | git (chungmin99/pyroki) | Thư viện động học học máy và tối ưu hóa robot trên JAX. |
| `yourdfpy` | 0.0.53+ | Tải và phân tích mô hình robot URDF (`assets/ur3e_custom.urdf`). |
| `robot_descriptions` | 1.10.0+ | Nạp mô tả động học robot và các tài nguyên mô hình tiêu chuẩn. |
| `torch` | 2.0.0+ (CPU) | Tính toán ma trận Jacobian cho SingularityGuard metric. |
| `viser` | 0.2.0+ | Máy chủ WebGL 3D Visualization, Gizmo tương tác và Web UI. |
| `ur_rtde` | 1.5.0+ | Giao tiếp thời gian thực RTDE với bộ điều khiển UR qua cổng 30003/30004. |
| `opencv-python-headless` | 4.8.0+ | Render và thu nhận ảnh camera cổ tay (640x480), nén MJPEG và overlay OSD. |
| `pyrealsense2` | 2.54.0+ | Driver giao tiếp trực tiếp camera độ sâu Intel RealSense D435/D455. |
| `transforms3d` | 0.4.1+ | Chuyển đổi giữa Axis-Angle, Quaternion và ma trận quay. |
| `trimesh` | 4.0.0+ | Tải và phân tích mô hình lưới 3D robot, ngàm camera và vật cản. |
| `h5py` | 3.9.0+ | Ghi và đọc tập dữ liệu AI Demonstration chuẩn HDF5. |
| `pyyaml` | 6.0+ | Đọc file cấu hình cảnh vật cản (`obstacle_scene.yml`) và ngưỡng kỳ dị (`singularity.yaml`). |
| `aiohttp` | 3.8.0+ | Máy chủ truyền nhận dữ liệu WebXR và WebSocket với độ trễ thấp. |
| `cryptography` | 41.0+ | Tự động sinh chứng chỉ SSL/TLS cho máy chủ HTTPS WebXR. |
| `numpy`, `scipy` | 1.24+, 1.11+ | Tính toán đại số tuyến tính, phép quay SE(3) và bộ lọc làm mượt tín hiệu. |

### Lệnh Cài Đặt

```bash
# Tạo và kích hoạt môi trường ảo Python 3.11
python3 -m venv /home/nguyen/ur_ws/pyroki_ursim/.venv
source /home/nguyen/ur_ws/pyroki_ursim/.venv/bin/activate

# Nâng cấp pip và cài đặt toàn bộ gói phụ thuộc từ requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Cấu Trúc Thư Mục Dự Án

```text
ur_ws/pyroki_ursim/
├── main.py                          # Điểm khởi chạy CLI duy nhất của hệ thống
├── README.md                        # Tài liệu hướng dẫn sử dụng hệ thống
├── requirements.txt                 # Danh sách gói phụ thuộc Python (PyRoki, JAX, ur_rtde, viser)
├── reset_home.py                    # Script hỗ trợ khẩn cấp đưa robot về Home qua URScript
├── assets/
│   ├── ur3e_custom.urdf             # URDF UR3e mô tả link, khớp và tool0
│   ├── obstacle_scene.yml           # Cấu hình kích thước bàn máy làm việc tham chiếu
│   ├── robot_camera_mount_1.stl     # Mô hình 3D ngàm gắn camera RealSense vào bích tool0
│   ├── d455.stl                     # Mô hình 3D camera Intel RealSense D455
│   └── quest3_ctrl_*.glb            # Mô hình 3D tay cầm Meta Quest 3S hiển thị trực quan
├── config/
│   └── singularity.yaml             # Ngưỡng an toàn Jacobian singularity (sigma_min)
├── core/
│   ├── pyroki_ursim_control.py      # Bộ điều khiển PyrokiURSim (JAX CPU IK, URScript, RTDE)
│   ├── data_logger.py               # Module ghi Demonstration HDF5 ở tần số 20Hz
│   ├── verify_dataset.py            # Công cụ kiểm tra chất lượng tập dữ liệu HDF5
│   └── replay_episode.py            # Công cụ tái hiện quỹ đạo thao tác
├── logs/
│   ├── quest_runs/                  # Lưu vết dữ liệu viễn thao Meta Quest (.jsonl)
│   ├── ai_training_data/            # Dữ liệu thao tác HDF5 thu thập phục vụ huấn luyện AI
│   └── singularity_data/            # Dữ liệu ma trận suy biến Jacobian
├── safety/
│   ├── __init__.py                  # Export SingularityGuard
│   └── singularity_guard/
│       ├── guard_core.py            # Giám sát điểm kỳ dị thời gian thực
│       ├── jacobian_metrics.py      # Tính toán sigma_min từ ma trận Jacobian
│       └── calibration.py           # Cân chỉnh ngưỡng kỳ dị theo không gian làm việc
├── teleop/
│   ├── __init__.py                  # Khởi tạo module teleop
│   ├── pyroki_solver.py             # Bộ giải PyrokiIKSolver (JAX JIT CPU, ~1ms, rest_cost, UR3E_LIMITS)
│   └── teleoperation.py             # TeleoperationController, ServoJExecutor, SafetyGate
├── teleop_quest/
│   ├── __init__.py                  # Xuất khẩu các lớp và hàm chính của package
│   ├── bootstrap.py                 # Khởi tạo Docker URSim, ADB watcher, config & SSL
│   ├── camera_pipeline.py           # Quản lý luồng nền camera, preview OSD & streaming
│   ├── camera_driver.py             # Driver camera cổ tay hợp nhất: Ảo 3D, RealSense USB & Ethernet
│   ├── camera_server_node.py        # Node phát stream video camera qua mạng Ethernet (ZMQ / HTTP)
│   ├── synthetic_camera.py          # Render camera ảo cổ tay 640x480 (Eye-in-Hand)
│   ├── viser_ui.py                  # Giao diện Viser 3D Cockpit và bảng điều khiển 2x2
│   ├── grasp_simulator.py           # Mô phỏng gắp/thả quả cà chua 3D vào giỏ
│   ├── quest_server.py              # Server HTTPS (8443), WSS (8444) và WS plain (8445)
│   ├── network_utils.py             # Dò tìm IP LAN và tự động sinh chứng chỉ SSL
│   ├── pose_sources.py              # WidgetPoseSource (đọc tọa độ Widget kèm bộ lọc EMA)
│   ├── delta_pose_source.py         # Cơ chế Mặt Đồng Hồ, nút B hiệu chuẩn bàn máy, deadband 3mm
│   ├── timeout_watchdog.py          # E-Stop và dừng khẩn cấp robot khi mất tín hiệu
│   ├── home_planner.py              # Đưa robot về Home an toàn 2 pha (movej 500Hz, cổng 30002)
│   ├── setup_quest_adb.sh           # Script chuyển tiếp cổng ADB sang localhost cho Quest
│   └── static/
│       ├── index.html               # Giao diện WebXR Cockpit trên kính Quest
│       ├── main.js                  # Xử lý gamepad WebXR, Toggle Gripper và rung Haptic
│       └── style.css                # Style hiển thị phong cách HUD hiện đại
└── tests/
    ├── test_camera.py               # Kiểm thử render camera ảo ở các góc nhìn & giao diện GUI
    ├── test_realsense.py            # Kiểm thử kết nối và đọc luồng ảnh camera Intel RealSense
    ├── test_dual_camera.py          # Kiểm thử hệ thống camera đa phương thức (USB & Ethernet)
    ├── test_joint_limits.py         # Kiểm thử các rào chắn giới hạn góc khớp và chống tự va chạm
    ├── test_multi_layer_safety.py   # Kiểm thử chuỗi bảo vệ an toàn đa lớp
    ├── test_simulated_pick_and_place.py # Kiểm thử tự động pick-place cà chua và xuất HDF5
    ├── test_singularity_guard.py    # Kiểm thử tính toán Jacobian singularity metric (sigma_min)
    ├── test_3d_hud.py               # Kiểm thử giao diện tương tác HUD tay cầm 3D
    ├── test_viser_ui_layout.py      # Kiểm thử bố cục điều khiển Viser UI
    ├── test_wrist_cam_removal.py    # Kiểm tra giải phóng tài nguyên luồng camera cổ tay Viser
    └── test_cockpit_launcher_bypass.py # Kiểm tra cơ chế bypass màn hình khởi chạy Cockpit
```

---

## 5. Hướng Dẫn Vận Hành

### Bước 1: Khởi động hệ thống
Bạn chỉ cần thực thi một lệnh duy nhất. Hệ thống sẽ tự động kiểm tra và khởi động container Docker URSim nếu đang chạy mô phỏng cục bộ:

```bash
cd /home/nguyen/ur_ws/pyroki_ursim
source .venv/bin/activate

# 1. Chạy với URSim mô phỏng (Camera ảo 3D - Mặc định, tự động kích hoạt Docker URSim):
python main.py

# 2. Chạy với Robot UR3e thật (Sử dụng Camera ảo 3D):
python main.py --robot-ip 10.42.0.242

# 3. Chạy kiểm tra độc lập Camera Intel RealSense D455/D435 (không cần bật Robot):
python tests/test_realsense.py --gui     # Cửa sổ trực quan live stream

# 4. Chạy CẢ HAI: Robot UR3e thật + Camera Intel RealSense cắm USB thật (Chuẩn thực tế):
python main.py --robot-ip 10.42.0.242 --cam-source usb --camera realsense

# 5. Chạy CẢ HAI: Robot UR3e thật + Camera qua cáp mạng Ethernet (ZMQ Zero-lag):
# (Phía máy nhúng gắn camera chạy: python teleop_quest/camera_server_node.py --port 5555)
python main.py --robot-ip 10.42.0.242 --cam-source eth --cam-url tcp://192.168.1.100:5555

# 6. Chạy CẢ HAI kèm Ghi dữ liệu Huấn luyện AI (JSONL Record):
python main.py --robot-ip 10.42.0.242 --camera realsense --record logs/quest_runs/episode_real_01.jsonl
```

Các tham số dòng lệnh được hỗ trợ:
- `--robot-ip`: Địa chỉ IP của Robot thật hoặc URSim (ví dụ: `10.42.0.242`, mặc định: `127.0.0.1`).
- `--cam-source`: Phương thức kết nối camera: `usb` (cắm cáp USB trực tiếp vào máy chủ) hoặc `eth` (truyền qua cáp mạng Ethernet ZMQ Zero-lag / HTTP) (mặc định: `usb`).
- `--cam-url`: Địa chỉ ZMQ endpoint hoặc HTTP URL khi sử dụng camera qua Ethernet (mặc định: `tcp://192.168.1.100:5555`).
- `--camera`: Nguồn camera cổ tay khi chạy USB: `realsense` (Intel RealSense SDK D455/D435 - Khuyến nghị), `real` (OpenCV V4L2), hoặc `synthetic` (mô phỏng 3D) (mặc định: `synthetic`).
- `--camera-device`: Chỉ số Device ID của camera thật V4L2 (mặc định: `0` tương ứng `/dev/video0`).
- `--camera-serial`: Serial number chỉ định khi máy cắm nhiều camera Intel RealSense.
- `--record <tên_file.jsonl>`: Tự động ghi log toàn bộ gói tin thô VR Teleop và telemetry robot ra file JSONL (khuyến nghị lưu vào thư mục `logs/quest_runs/`).

Hệ thống sẽ hiển thị các cổng truy cập:
- Viser Dashboard: `http://localhost:8080`
- WebXR Quest (Wi-Fi): `https://<IP_LAN>:8443`
- WebXR Quest (Cáp USB ADB): `https://localhost:8443`
- Cổng WebSocket WebXR Quest: `wss://<IP_LAN>:8444`
- Cổng WebSocket PC / Local Preview: `ws://localhost:8445`
- Giao diện URSim PolyScope (noVNC): `http://192.168.56.101:6080/vnc.html` (hoặc `http://localhost:6080`)

### Bước 2: Điều khiển bằng Viser Web UI (Máy tính)
1. Mở trình duyệt web trên máy tính và truy cập `http://localhost:8080`.
2. Kiểm tra mô hình robot 3D hiển thị khớp đúng với trạng thái thực tế của robot.
3. **Thao tác trên Bảng điều khiển Lưới 2x2:**
   - **Bật Điều Khiển (Teleop):** Tích chọn để kích hoạt cánh tay robot bám theo mục tiêu (Nhãn `Status` chuyển sang `Tracking`).
   - **Gripper (Bóp Kẹp):** Tích chọn để đóng ngàm kẹp gắp quả cà chua (hoặc bỏ chọn để nhả vật thể).
   - **Record AI Episode:** Tích chọn để bắt đầu ghi chuỗi thao tác phục vụ huấn luyện mô hình Robot AI.
   - **Hủy Episode (Discard):** Tích chọn khi thực hiện thao tác hỏng; hệ thống sẽ hủy bỏ dữ liệu tập hiện tại và tự động nhả checkbox sau 200ms.
4. **Điều khiển chuyển động:** Dùng chuột nhấp và kéo các trục mũi tên hoặc vòng xoay của khối Target Widget 3D: Cánh tay robot sẽ chuyển động bám theo tọa độ đích của chuột trong thời gian thực.
5. **Trở về vị trí an toàn:** Bấm nút **Go to Home** khi cần đưa robot về vị trí mặc định an toàn. Quỹ đạo chuyển động 2 pha được điều phối mượt mà không rung giật (MoveJ 500Hz qua cổng 30002), cập nhật Viser 25Hz, không glitch hình ảnh và tự động kích hoạt lại chế độ Teleop sau khi robot về đích.

### Bước 3: Hướng Dẫn Chi Tiết Điều Khiển Bằng Kính Meta Quest 3S (WebXR)

#### 3.1. Bảng Chức Năng Các Phím Trên Tay Cầm Meta Quest 3S

| Phím bấm | Ngón tay thao tác | Cơ chế điều khiển | Mô tả chi tiết hành vi |
| :--- | :--- | :--- | :--- |
| **Nhấn cần gạt Thumbstick (R3/L3)** | Ngón cái | **Về Vị Trí An Toàn (Go Home)** | **Nhấn 1 lần vào cần gạt Analog** (hoặc nhấn giữ nút B/Y trong 3 giây) để tự động đưa cánh tay robot về vị trí Home an toàn. Tay cầm rung xác nhận 2 nhịp, tự động mở khóa bảo vệ và tái kích hoạt Teleop khi về đích. |
| **Nút B (Tay phải) / Y (Tay trái)** | Ngón cái | **Hiệu Chuẩn Bàn Máy / Đổi Chế Độ** | - **Bấm nhanh (< 3s):** Hiệu chuẩn hướng Tiến chuẩn theo bàn máy (lấy góc yaw đầu) đồng thời chuyển đổi giữa **Edit Mode** và **Drive Mode**.<br>- **Nhấn giữ (> 3s):** Kích hoạt lệnh **Go Home** đưa robot về vị trí an toàn. |
| **Cò Trỏ (Index Trigger)** | Ngón trỏ | **Tịnh Tiến Mặt Đồng Hồ (Clock Face Translation)** | Bấm giữ cò và di chuyển tay: **Tiến/Lùi** duỗi/thu bắp tay theo phương vươn $q_0$; **Phải/Trái** quay đế $q_0$ quanh Base Z kèm bù góc Z ngàm kẹp; **Cao/Thấp** nâng hạ Z thẳng đứng. Deadband tĩnh 3mm. |
| **Nút Hông (Hand Grip)** | Ngón giữa / ngón áp út | **Xoay 3 Trục Độc Lập (Decoupled 3-Axis Rotation)** | Bấm giữ nút hông để xoay ngàm gắp trên cả 3 trục: Gập/chúc (Pitch), Lắc ngang (Yaw), và Vặn ngàm (Roll). Hệ thống tích hợp bộ lọc trực giao World Frame và Khóa trục thông minh (Soft Rotation Snapping), triệt tiêu hoàn toàn hiện tượng rò rỉ chéo trục khi gập mở cổ tay. |
| **Nút A (Tay phải) / X (Tay trái)** | Ngón cái | **Bật / Tắt Ngàm Kẹp (Toggle Gripper)** | **Nhấn 1 lần** để kẹp/hút quả cà chua, **nhấn lần nữa** để nhả vật thể vào giỏ. Tay cầm rung phản hồi haptic mỗi lần bấm. |
| **Nút Meta (Oculus)** | Ngón cái (Tay phải) | **Căn Chuẩn Trục Nhìn (Recenter)** | Nhấn giữ 2 giây để đồng bộ lại hệ trục phòng của kính Meta Quest. |

#### 3.2. Cơ Chế Khóa Hướng Tiến Theo Đế Robot ($q_0$)
Khi người vận hành đứng ở các góc khác nhau quanh robot hoặc khi đế robot đã xoay một góc $q_0$:
- **Tâm đồng hồ:** Trục quay khớp đế Base Z $(0, 0)$.
- **Kim đồng hồ:** Góc vật lý của khớp đế $q_0$.
- **Thao tác Sang Phải / Sang Trái:** Quay kim đồng hồ (khớp đế $q_0$) quanh Base Z (sang Phải quay theo chiều kim đồng hồ, sang Trái quay ngược chiều kim đồng hồ), đồng thời tự động bù góc xoay trục Z ngàm kẹp để giữ ngàm luôn hướng thẳng vào bàn máy.
- **Thao tác Tiến / Lùi:** Duỗi dài hoặc thu ngắn kim đồng hồ theo đúng phương vươn cơ học $\vec{u}_{reach} = [\cos(q_0), \sin(q_0), 0.0]$, triệt tiêu hoàn toàn hiện tượng robot bị bẻ ngoặt hoặc quay vòng khi kéo tay thẳng về sau.
- **Thao tác Cao / Thấp:** Nâng hạ độ cao Z thẳng đứng trên mặt bàn làm việc, luôn nằm trong khoảng an toàn $Z \in [0.00, 0.58]$m.
- **Hiệu chuẩn hướng Tiến theo bàn máy bằng nút B:** Khi đứng trước bàn máy, người dùng chỉ cần nhìn thẳng mép bàn và bấm phím **B**. Hệ thống lập tức trích xuất góc yaw của đầu kính (`head_quat`) làm vector hướng Tiến chuẩn (`_desk_u_fwd`) và hướng Phải chuẩn (`_desk_u_right`), đảm bảo các lần vươn tay sau đó luôn trùng khớp 100% với cạnh bàn.

#### 3.3. Hệ Thống Lọc Nhiễu Đa Tầng & Xoay 3 Trục Độc Lập (Anti-Noise & Arm-Plane Decoupled 3-Axis)
Nhằm mang lại chuyển động siêu êm, loại bỏ triệt để rung lắc sinh trắc học và tracking jitter:
- **Tách biệt hoàn toàn 3 trục (World Frame Decoupling & Intuitive Mapping):** Tính toán biến thiên góc xoay trực tiếp trong Không gian Thực ($\Delta R_{world} = R_{hand} \cdot R_{anchor}^{-1}$), triệt tiêu 100% hiện tượng góc nghiêng tay cầm Quest làm biến dạng trục xoay:
  - **Pitch (Gập/ngửa cổ tay - Uống nước):** Khi dốc mũi tay về phía mình, đầu ngàm kẹp ngửa ra xa thân máy theo đúng chiều trực quan; khi chúi tay xuống, đầu kẹp gập vào thân máy.
  - **Roll (Lắc cổ tay trái/phải):** Khi lắc nhẹ bàn tay sang hai bên, hai ngón kẹp của robot xoay tròn tại chỗ quanh trục kẹp (Tool Z) để căn chỉnh góc đón quả cà chua trên bàn một cách tự nhiên và đỡ mỏi cơ cẳng tay.
  - **Tilt (Vặn chìa khóa quanh cẳng tay):** Nghiêng đầu ngàm kẹp sang trái hoặc sang phải (Lateral Tilt quanh trục vươn của cánh tay $\vec{u}_{reach}$), phản hồi trực quan theo hướng vặn chìa khóa.
- **Đồng bộ hóa theo Mặt Phẳng Cánh Tay (Arm-Plane Rotation Mapping):** Trục gập (Pitch quanh $\vec{u}_{lateral}$) và trục nghiêng hai bên (Tilt quanh $\vec{u}_{reach}$) tự động bám theo góc vươn của cánh tay robot ($q_0$). Khi bấm giữ cả 2 nút (vừa kéo tay thu về vừa ngửa cổ tay), chuyển động tịnh tiến và gập ngửa hòa làm một chuyển động liền mạch dọc theo đúng hướng cánh tay thu về, không bị lệch pha hay ngoẹo góc.
- **Khóa rãnh đơn trục triệt để (Soft Snapping < 55%):** Nâng ngưỡng khóa trục phụ lên $55\%$ để triệt tiêu hoàn toàn góc kéo xéo sinh học tự nhiên khi thao tác. Khi gập ngửa hoặc lắc cổ tay, robot chỉ chuyển động thẳng tắp trên trục mong muốn, triệt tiêu $100\%$ hiện tượng lắc chao đảo phụ. Chuyển tiếp mượt sang xoay 3D đa trục khi chủ ý bẻ góc lớn ($> 75\%$).
- **Vùng chết mềm chống vi rung (Soft Deadband $2.6^\circ$ & $3\text{mm}$):** Khi người vận hành bấm giữ phím Grip hoặc Trigger mà vẫn giữ yên tay, mọi dao động run cơ học tự nhiên được hấp thụ hoàn toàn về $0.00$ (`POS_DEADBAND = 0.003`m). Chuyển động bắt đầu từ $0$ liên tục, không bị nhảy cóc khi vượt ngưỡng.
- **Chốt giữ trục chính (Hysteresis Latch):** Ngăn chặn hiện tượng lật trục chập chờn (Chatter) khi người dùng thao tác ở ranh giới giữa 2 trục, ưu tiên giữ ổn định trục ban đầu trừ khi trục mới vượt trội > 15%.
- **Bộ lọc thích ứng động (Dynamic Adaptive EMA & Slew Rate Limiter):**
  - **Trên không gian trắc địa $SO(3)$ và $\mathbb{R}^3$:** Khi tay đứng yên hoặc vi rung chậm, hệ số $\alpha$ tự động giảm thấp ($\alpha \approx 0.18$) để lọc phẳng hoàn toàn nhiễu tracking IMU. Khi di chuyển dứt khoát, $\alpha$ tự động tăng cao ($\alpha \approx 0.28 - 0.40$) để phản hồi tức thì với độ trễ cực thấp.
  - **Giới hạn biến thiên tối đa (Slew Rate Limiter):** Khống chế bước nhảy tối đa $5.5\text{cm}/50\text{ms}$ (tương đương vận tốc $1.1\text{m/s}$) và $11.5^\circ/50\text{ms}$, triệt tiêu nguy cơ robot bị giật nảy đột biến khi WebXR rớt frame hoặc tracking bị vấp.
- **Tùy biến linh hoạt qua biến môi trường:**
  - `ROTATION_MODE=decoupled_3axis` (Mặc định): Điều khiển 3 trục độc lập theo Arm-Plane với hệ thống lọc nhiễu đa tầng.
  - `ROTATION_MODE=table_roll`: Khóa chúc ngàm cố định vuông góc bàn, chỉ Roll quanh trục kẹp.
  - `ROTATION_MODE=free_3d`: Xoay 3D tự do.
  - `ROT_DEADBAND` (Mặc định `0.045` rad ~ $2.6^\circ$), `POS_DEADBAND` (Mặc định `0.003` m ~ $3\text{mm}$).
  - `ROT_SNAP_LOW` (Mặc định `0.55`), `ROT_SNAP_HIGH` (Mặc định `0.75`).

#### 3.4. Bảng Quy Chuẩn Điều Khiển & Kịch Bản Kiểm Thử (Đã Xác Thực Đúng Logic & Sẵn Sàng Vận Hành)
Hệ thống điều khiển 6-DOF đã được chuẩn hóa và kiểm nghiệm thực tế, hoàn toàn sẵn sàng cho vận hành trên robot thật:

| Thao tác tay người cầm Quest | Trục toán học | Phản hồi chuyển động trên Robot UR3e | Trạng thái kiểm chứng |
| :--- | :--- | :--- | :--- |
| **Gập/ngửa cổ tay (Uống nước)** | Pitch ($\vec{u}_{lateral}$) | Khi dốc tay về phía mình: ngàm ngửa ra xa thân máy. Khi chúi tay: ngàm gập vào thân máy. Khóa cứng lắc phụ. | Đã xác thực đúng logic |
| **Lắc cổ tay trái / phải** | Roll ($\text{Tool Z}$) | Hai ngón kẹp xoay tròn tại chỗ quanh trục kẹp (CW / CCW) để căn góc đón quả cà chua trên bàn. | Đã xác thực đúng logic |
| **Vặn chìa khóa quanh cẳng tay** | Tilt ($\vec{u}_{reach}$) | Nghiêng ngàm kẹp sang bên phải (khi vặn phải) hoặc nghiêng sang bên trái (khi vặn trái). | Đã xác thực đúng logic |
| **Bấm giữ đồng thời Cò (Trigger) + Grip** | Tịnh tiến + Xoay | Tịnh tiến dọc theo cánh tay vươn kết hợp gập ngửa và chỉnh góc ngàm đồng bộ, không giật cục. | Đã xác thực đúng logic |
| **Bấm nút A hoặc X** | Toggle Gripper | Kích hoạt đóng / mở ngàm kẹp gắp cà chua với phản hồi rung Haptic trên tay cầm. | Đã xác thực đúng logic |

#### 3.5. Kỹ Thuật Ly Hợp Điều Khiển Tầm Xa (Smart Clutching)
Khi cần điều khiển robot di chuyển quãng đường dài vượt quá tầm với tự nhiên của sải tay:
1. Bấm giữ Cò trỏ và đưa tay về phía trước đến hết tầm với thoải mái.
2. **Nhả Cò trỏ:** Cánh tay robot sẽ dừng lại và giữ nguyên vị trí hiện tại trên bàn.
3. Rút tay cầm về gần ngực hoặc vị trí thoải mái nhất của cơ thể.
4. **Bấm giữ lại Cò trỏ:** Hệ thống tự động neo mốc mới (Reset Anchor). Robot tiếp tục di chuyển tiến tới từ vị trí đang đứng một cách mượt mà, hoàn toàn không bị giật hay nhảy cóc tọa độ (Zero Teleport).

#### 3.6. Kết Nối Kính Meta Quest Qua Cáp USB (ADB Reverse Cực Nhanh)
Khi cắm cáp USB giữa PC và kính Meta Quest:
1. Tiến trình ngầm `start_adb_watcher` trong [bootstrap.py](file:///home/nguyen/ur_ws/pyroki_ursim/teleop_quest/bootstrap.py) sẽ tự động nhận diện thiết bị và chạy script chuyển tiếp cổng.
2. Nếu kính báo `unauthorized`, hãy đeo kính lên và nhấn **"Allow USB Debugging"** (và tích chọn "Always allow from this computer").
3. Khi thông báo kết nối thành công xuất hiện trên terminal, chỉ cần mở Meta Quest Browser trên kính và truy cập:
   ```text
   https://localhost:8443
   ```
4. Bấm **Advanced** -> **Proceed to localhost (Tiếp tục truy cập an toàn)** -> Bấm **Enter VR** màu xanh lục để bắt đầu điều khiển.

* **Xử lý sự cố khi không tải được trang (`This site can't be reached` hoặc reload không được):**
  Nếu trình duyệt kính không tải được hoặc cổng chuyển tiếp bị treo, chạy lệnh sau ở terminal máy tính để khởi động lại dịch vụ ADB và thiết lập lại toàn bộ cổng chuyển tiếp:
  ```bash
  adb kill-server && adb start-server
  ./teleop_quest/setup_quest_adb.sh
  ```

#### 3.7. Trực Quan Hóa Mô Hình Tay Cầm Meta Quest 3/3S Đồng Bộ Với Gizmo
Mô hình tay cầm VR hoạt động như một bản sao kỹ thuật số (Digital Twin) trực quan đồng bộ cùng Gizmo trên giao diện Viser 3D:
- **Hiển thị song song cùng Gizmo:** Gizmo 3D (các trục mũi tên tọa độ Đỏ/Xanh lá/Xanh dương) đặt chính xác tại điểm TCP của robot, trong khi mô hình tay cầm 3D đặt ngay phía trước điểm TCP theo đúng hướng di chuyển, cung cấp góc nhìn kép trực quan hoàn hảo.
- **Căn chỉnh theo đúng hướng di chuyển (Arm-Plane Forward):**
  - Mũi tay cầm và các phím bấm luôn tự động quay theo đúng hướng vươn của cánh tay robot: $\vec{v}_{\text{forward}} = [\cos(q_0), \sin(q_0), 0.0]$.
  - Cán cầm luôn đứng thẳng tự nhiên vuông góc mặt đất (hướng xuống $-Z$), mặt phím ngửa lên trên để dễ quan sát phím bấm.
- **Khoảng cách đệm an toàn động & Bố trí song song (Offset 8.5cm tiến, 7.5cm lệch phải):** Tay cầm luôn được giữ ở khoảng cách $8.5\text{cm}$ phía trước và dịch sang mạn phải $7.5\text{cm}$ theo hệ quy chiếu cánh tay robot ($\vec{v}_{\text{forward}}$ và $\vec{v}_{\text{right}}$). Bố cục này giúp Gizmo (tâm TCP) và tay cầm 3D đứng cạnh nhau song song rất thoáng đãng, hoàn toàn không che khuất nhau dù quan sát từ bất kỳ góc nhìn nào.
- **Đồng bộ góc xoay 1:1 với Gizmo ($R_{\text{ctrl}} = R_{\text{gizmo}} \cdot R_{\text{align}}$):** Khi người dùng bóp Grip xoay cổ tay thực tế một góc biến thiên $\Delta R$, cả Gizmo và tay cầm 3D lập tức xoay đồng bộ $1:1$ theo đúng chuyển động đó.
- **Phản hồi trạng thái phím bấm trực quan (Interactive Button Feedback - 8 Trạng Thái Độc Lập):**
  - **Trạng thái nghỉ (Idle):** Màu xám trắng tự nhiên ($60\%$ opacity).
  - **Bóp Cò (Trigger):** Sáng màu xanh lá cây (báo hiệu điều khiển tịnh tiến).
  - **Bóp Grip:** Sáng màu vàng cam (báo hiệu điều khiển xoay 3 trục).
  - **Bóp cả 2 phím (Trigger + Grip):** Sáng màu xanh dương ngọc (báo hiệu phối hợp 6-DOF).
  - **Bấm nút A / X (Gripper):** Sáng màu tím hồng / Magenta (báo hiệu thao tác đóng/mở ngàm kẹp).
  - **Bấm nút B / Y:** Sáng màu vàng kim / Gold (báo hiệu chuyển đổi Drive Mode và hiệu chuẩn bàn máy).
  - **Xoay / Gạt cần Joystick (Analog Tilt):** Sáng màu tím hoa cà / Violet (báo hiệu thao tác điều hướng joystick).
  - **Nhấn Joystick / Đang chạy về Home (Go Home):** Sáng màu xanh dương công nghệ / Royal Blue (báo hiệu lệnh và quá trình đưa robot về Home an toàn).
- **Thanh công cụ "Hiển thị 3D" trên Viser (Bật/Tắt độc lập):**
  - Hai nút chuyển đổi xếp gọn gàng trên cùng một hàng ngang: `Gizmo` (mặc định bật) và `Tay Cầm` (mặc định bật).
  - Tách biệt độc lập $100\%$: Người dùng có thể bật/tắt linh hoạt cả hai tùy ý (xem cả hai, chỉ xem tay cầm, chỉ xem gizmo, hoặc ẩn cả hai).

---

## 6. Hướng Dẫn Triển Khai Trên Cánh Tay Robot UR3e Thực Tế

### 6.1. Kết Nối Phần Cứng & Cấu Hình Mạng LAN

Cắm cáp mạng Ethernet trực tiếp từ cổng mạng của máy tính (PC) vào cổng mạng trên hộp điều khiển UR Control Box (hoặc qua Switch / Router mạng nội bộ).

#### Lựa chọn 1: Cấu hình tự động qua DHCP (Khuyên dùng khi cắm qua Router phòng Lab hoặc mạng chia sẻ)
1. Trên màn hình PolyScope Teach Pendant: Vào **Settings** -> **System** -> **Network**.
2. Chọn **DHCP** -> Bấm **Apply**.
3. Chờ 3 – 5 giây để trạng thái chuyển sang màu xanh lá: **`Network is connected`**.
4. Ghi lại địa chỉ IP hiển thị trên màn hình PolyScope (ví dụ: `10.35.44.150` hoặc `192.168.1.xxx`).
5. Trên máy tính điều khiển PC, kiểm tra kết nối với IP vừa nhận:
   ```bash
   ping -c 3 <IP_ROBOT>
   nc -zv <IP_ROBOT> 29999 30002 30004
   ```
> **Mẹo khi cắm cáp trực tiếp PC <-> Robot không qua Router:** Trên Ubuntu PC, vào **Settings** -> **Network** -> **Wired** (bấm bánh răng cài đặt) -> Tab **IPv4** -> Chọn **"Shared to other computers"** -> Bấm **Apply**. Lúc này Ubuntu PC sẽ tự phát DHCP dải `10.42.0.x` cho Robot, Robot chọn DHCP sẽ lập tức kết nối thành công và nhận IP `10.42.0.xxx`!

#### Lựa chọn 2: Cài đặt IP Tĩnh (Static IP: 10.42.0.242)
1. **Trên màn hình PolyScope Teach Pendant:**
   - Vào menu: **Settings** -> **System** -> **Network**.
   - Chọn **Static Address**:
     - IP Address: `10.42.0.242`
     - Subnet Mask: `255.255.255.0`
     - Default Gateway: `10.42.0.1`
   - Bấm **Apply** và xác nhận dòng trạng thái hiển thị: *Network is connected*.
   > **Xử lý lỗi "Not connected to network":** Nếu PolyScope báo *Not connected*, nguyên nhân là do máy tính PC chưa đặt IP tĩnh cùng dải `10.42.0.1` nên Robot không ping thấy Gateway. Hãy thực hiện bước cài đặt trên PC dưới đây để thông mạng.
2. **Cài đặt IP trên máy tính điều khiển (PC) qua Củ Chuyển USB-to-Ethernet Adapter:**

> **Lưu ý phần cứng:** Luôn cắm adapter USB-to-Ethernet vào cổng **USB 3.0** (chân cắm màu xanh dương hoặc có ký hiệu `SS`). Tránh cắm qua hub USB chia nhiều cổng không có nguồn phụ để tránh chập chờn điện áp gây ngắt kết nối RTDE.

Khi cắm củ chuyển USB-to-Ethernet vào máy tính chạy Ubuntu, hệ điều hành sẽ tự nhận một giao diện mạng mới (thường có tên dạng `enx<MAC_ADDRESS>` như `enx00e04c680001` hoặc `eth1`). Người vận hành thực hiện cấu hình IP tĩnh theo 1 trong 2 cách sau:

##### Cách A: Cấu hình nhanh bằng Giao diện Đồ họa Ubuntu (Khuyến nghị - 30 giây)
1. Mở **Settings** -> **Network**.
2. Tại mục kết nối có dây mới xuất hiện (**Wired / USB Ethernet**), bấm vào biểu tượng bánh răng cài đặt.
3. Chuyển sang tab **IPv4**:
   - Chọn chế độ: **Manual**
   - **Address:** `10.42.0.1` (hoặc `10.42.0.50`)
   - **Netmask:** `255.255.255.0`
   - **Gateway:** `10.42.0.1` *(hoặc để trống nếu chỉ kết nối mạng nội bộ riêng với Robot)*
4. Bấm **Apply**, sau đó gạt công tắc Wired Tắt rồi Bật lại để hệ thống kích hoạt IP mới.

##### Cách B: Cấu hình nhanh bằng Dòng lệnh CLI (`nmcli`)
```bash
# Bước 1: Quét danh sách thiết bị để tìm tên card mạng USB (dạng enx... hoặc eth...):
nmcli device status

# Bước 2: Tạo hoặc gán IP tĩnh 10.42.0.1/24 cho card mạng USB (thay <TÊN_CARD_USB> bằng tên thực tế):
sudo nmcli con add type ethernet con-name "USB-Robot-LAN" ifname <TÊN_CARD_USB> ip4 10.42.0.1/24

# Bước 3: Kích hoạt kết nối:
sudo nmcli con up "USB-Robot-LAN"
```

##### Bước kiểm tra thông link từ máy tính sang Robot:
```bash
# Kiểm tra ping trực tiếp sang Robot:
ping -c 3 10.42.0.242

# Quét 3 cổng dịch vụ cốt lõi của URControl: 29999 (Dashboard), 30002 (Primary), 30004 (RTDE):
nc -zv 10.42.0.242 29999 30002 30004
```

### 6.2. Cấu Hình URCap & Tương Thích Thanh Ghi Fieldbus
1. **Cài đặt URCap (Tùy chọn thiết bị ngoại vi):**
   - Nếu sử dụng ngàm kẹp thực tế (ví dụ Robotiq 2F-85/Hand-E), cài đặt URCap chính hãng tương ứng qua **Settings** -> **System** -> **URCaps**.
   - Nếu phối hợp với ROS 2 Driver, nạp daemon *External Control URCap* trên PolyScope và thiết lập IP máy trạm PC.
2. **Xử lý xung đột thanh ghi RTDE (`One of the RTDE input registers are already in use!`):**
   - **Nguyên nhân:** Các URCap giao tiếp hoặc giao thức công nghiệp (Profinet / EtherNet-IP) có thể chiếm dụng mặc định dải thanh ghi số nguyên thấp (registers 0-23) của RTDE.
   - **Cách xử lý trên Teach Pendant:** Nếu không sử dụng PLC ngoại vi, hãy tắt dải truyền thông công nghiệp tại: **Installation** -> **Fieldbus** -> **Profinet / EtherNet-IP** -> Bỏ chọn *Enable*.
   - **Cơ chế tự động của hệ thống:** Module `ServoJExecutor` tự động gửi tín hiệu `stop\n` tới Dashboard Server (Port 29999) trước khi bắt tay kết nối, đồng thời tự động kích hoạt cờ fallback `FLAG_UPPER_RANGE_REGISTERS` (chuyển sang dải thanh ghi trên 24-47) giúp hệ thống kết nối song song trơn tru mà không xung đột với URCap khác.

### 6.3. Quy Trình Cấu Hình Chi Tiết Trên Tay Dạy PolyScope (Teach Pendant e-Series)

Quy trình 10 bước chuẩn hóa từ A-Z khi đứng trước cánh tay robot thật UR3e:

#### Bước 1: Khởi Động Nguồn Control Box & Nạp Giao Diện PolyScope
- Bật công tắc nguồn cơ học trên mặt trước của tủ điều khiển UR Control Box.
- Chờ màn hình cảm ứng Teach Pendant khởi động và tải hoàn tất hệ điều hành PolyScope 5.

#### Bước 2: Thiết Lập Mạng LAN (Network Setup)
- Mở menu chính (biểu tượng 3 gạch góc trên bên phải) -> **Settings** -> **System** -> **Network**.
- Lựa chọn phương thức kết nối:
  - **Cách A (DHCP):** Chọn **DHCP** -> Bấm **Apply**. Đợi quả cầu mạng chuyển màu xanh lá *Network is connected* và đọc địa chỉ IP hiển thị.
  - **Cách B (Static IP):** Chọn **Static Address** -> Nhập IP `10.42.0.242`, Subnet `255.255.255.0`, Gateway `10.42.0.1` -> Bấm **Apply**.

#### Bước 3: Cấp Quyền & Kích Hoạt Remote Control (Bắt buộc cho kênh RTDE Control)
- Vào **Settings** -> **System** -> **Remote Control** -> Chọn **Enable**.
- Quan sát thanh tiêu đề header trên cùng (cạnh đồng hồ), nhấn vào biểu tượng chế độ điều khiển đang ở chữ **Local** -> Chọn chuyển sang **Remote Control**.
- Biểu tượng sẽ chuyển từ hình tay dạy sang **hình mạng máy tính**.
*(Lưu ý: Nếu không chuyển sang Remote Control, URControl sẽ chặn socket ghi lệnh qua RTDE và bắn lỗi `Please enable remote control on the robot!`)*.

#### Bước 4: Khai Báo Đầu Công Cụ (TCP), Tải Trọng (Payload) & Trọng Tâm (CoG)
*(Bước bắt buộc để tránh hiện tượng rung lắc cánh tay hoặc kích hoạt lỗi `Protective Stop: Position deviation`)*:
- Mở tab **Installation** -> **General** -> **TCP**:
  - Chọn TCP làm việc, khai báo vị trí đầu gắp: nếu lắp ngàm Robotiq 2F-85 và mount camera RealSense, offset trục $Z \approx 135\text{ mm}$ ($0.135\text{ m}$), $X = 0$, $Y = 0$.
- Mở tab **Installation** -> **General** -> **Payload**:
  - Khai báo tổng khối lượng đầu công cụ: $\approx 1.1\text{ kg}$ (gồm ngàm gắp + camera Intel RealSense D455/D435 + bracket mount).
  - Khai báo trọng tâm CoG: $Cx = 0\text{ mm}$, $Cy = 0\text{ mm}$, $Cz \approx 45\text{ mm}$.
- Nhấn **Save** để lưu file cấu hình cài đặt (`default.installation`).

#### Bước 5: Cấu Hình Ranh Giới An Toàn Phần Cứng (Safety Limits & Boundaries)
- Mở tab **Installation** -> **Safety** -> Nhập mật khẩu Safety (mặc định của UR là `easybot` nếu chưa thay đổi).
- Vào mục **Robot Limits**: Thiết lập giới hạn tốc độ đầu công cụ (TCP Speed Limit) $\le 250\text{ mm/s}$ cho chế độ thử nghiệm ban đầu.
- Vào mục **Planes**: Thêm một mặt phẳng hạn chế (*Restrict Plane*) mô phỏng mặt bàn thao tác, đặt cách mặt bàn vật lý $\ge 2\text{ cm}$ về phía trên. Bộ điều khiển an toàn phần cứng URControl sẽ cưỡng chế dừng cánh tay nếu quỹ đạo điều khiển có nguy cơ va quẹt mặt bàn.
- Nhấn nút **Apply & Restart** cấu hình an toàn.

#### Bước 6: Tắt Xung Đột Truyền Thông Công Nghiệp Fieldbus
- Mở tab **Installation** -> **Fieldbus**:
  - Chọn **Profinet** -> Bỏ tick chọn **Enable**.
  - Chọn **EtherNet/IP** -> Bỏ tick chọn **Enable**.
- *Mục đích:* Giải phóng dải thanh ghi số nguyên đầu vào RTDE (`input_int_register[0..23]`) khỏi sự chiếm dụng ngầm của các giao thức công nghiệp khi không sử dụng PLC ngoại vi.

#### Bước 7: Bật Nguồn Động Cơ Khớp & Nhả Phanh Cơ Học
- Nhìn góc dưới cùng bên trái màn hình PolyScope, nhấn vào nút trạng thái robot đang hiển thị màu đỏ (**Power Off**):
  1. Nhấn nút **ON** (Đèn trạng thái chuyển sang màu vàng - **Idle**).
  2. Nhấn nút **START / RELEASE BRAKES** (Lắng nghe các tiếng lách tách nhả phanh cơ học tại 6 khớp động cơ, đèn trạng thái chuyển sang màu xanh lá cây - **RUNNING / NORMAL**).
- Đóng cửa sổ điều khiển nguồn.

#### Bước 8: Đưa Cánh Tay về Tư Thế Khởi Đầu An Toàn (Ready Pose) Chống Kẹt Kỳ Dị
*(Thao tác khi robot đang ở Local mode trước khi chuyển sang Remote mode, hoặc dùng nút Freedrive)*:
- Nhấn giữ nút **Freedrive** (nút màu đen lớn ở mặt sau tay dạy Teach Pendant) để dắt tay robot, hoặc mở tab **Move** để nhập trực tiếp góc 6 khớp:
  - Khớp 1 (Base): $0.0^\circ$ (`0.00` rad)
  - Khớp 2 (Shoulder): $-126.0^\circ$ (`-2.20` rad)
  - Khớp 3 (Elbow): $+109.0^\circ$ (`+1.90` rad)
  - Khớp 4 (Wrist 1): $-90.0^\circ$ (`-1.57` rad)
  - Khớp 5 (Wrist 2): $-90.0^\circ$ (`-1.57` rad)
  - Khớp 6 (Wrist 3): $0.0^\circ$ (`0.00` rad)
- **Lưu ý sống còn:** Tuyệt đối tránh tư thế duỗi thẳng cánh tay hoặc để khớp cổ tay số 5 $q_5 = 0^\circ$ (kỳ dị cổ tay Wrist Singularity khiến giá trị suy biến $\sigma_{min} < 0.035$ và kích hoạt vòng lặp Auto-Snap liên tục).

#### Bước 9: Hạ Tốc Độ Thử Nghiệm & Sẵn Sàng Tay Cầm E-Stop
- Kéo thanh trượt tốc độ (**Speed Slider**) ở cạnh dưới cùng màn hình PolyScope xuống mức **10% – 20%** trong lần vận hành đầu tiên. Khi hệ thống đã bám tracking ổn định mới từ từ đẩy lên $50\% - 100\%$.
- Người vận hành luôn cầm chắc Teach Pendant trên tay, ngón tay cái đặt hờ trên nút nấm đỏ **Emergency Stop** để có thể bấm ngắt nguồn tức thì nếu xảy ra sự cố bất ngờ.

#### Bước 10: Xử Lý Khôi Phục Khi Gặp Lỗi Protective Stop (Recovery Guide)
Nếu có va chạm nhẹ hoặc gia tốc tăng đột ngột, URControl sẽ vào trạng thái **Protective Stop** (màu vàng cam):
- **Cách 1 (Trực tiếp trên Teach Pendant):** Nhấn nút **Unlock Protective Stop** màu vàng hiển thị trên màn hình. Robot sẽ kiểm tra trạng thái tĩnh trong 3-5 giây và tự động khôi phục về trạng thái xanh **RUNNING**.
- **Cách 2 (Mở khóa từ PC qua cổng Dashboard 29999):**
  ```bash
  echo -e "unlock protective stop\nclose safety popup\n" | nc -w 1 10.42.0.242 29999
  ```
- Sau khi mở khóa, chạy lại lệnh vận hành tương ứng từ máy tính.

---

#### Bảng Checklist 10 Điểm Trước Khi Khởi Chạy Code Teleop
| STT | Hạng mục kiểm tra | Trạng thái đạt yêu cầu | Ghi chú |
| :---: | :--- | :---: | :--- |
| 1 | Cáp mạng LAN | Đã cắm chắc chắn | Cáp Cat5e/Cat6 từ PC tới Control Box |
| 2 | Trạng thái Mạng | `Network is connected` | Đèn xanh lá trong Settings -> Network |
| 3 | Chế độ Điều khiển | `Remote Control` | Icon mạng trên thanh header PolyScope |
| 4 | Cấu hình Payload | Đã nhập đúng kg & CoG | Tránh lỗi lệch vị trí Protective Stop |
| 5 | Ranh giới An toàn | Plane $\ge 2\text{ cm}$ | Chống cánh tay va quẹt mặt bàn |
| 6 | Fieldbus | Đã tắt Profinet / EtherNet-IP | Tránh xung đột thanh ghi RTDE |
| 7 | Trạng thái Nguồn | `RUNNING` (Xanh lá) | Đã bấm On và Release Brakes |
| 8 | Tư thế Khởi đầu | Khuỷu gập tự nhiên | $q_1 \approx -126^\circ, q_2 \approx +109^\circ, q_4 \approx -90^\circ$ |
| 9 | Thanh Tốc Độ | Đặt mức $10\% - 20\%$ | Cho lần chạy thử nghiệm ban đầu |
| 10 | An Toàn Con Người | Cầm sẵn nút Emergency Stop | Quan sát không gian xung quanh cánh tay |

### 6.4. Hướng Dẫn Cấu Hình & Kiểm Tra Camera Vật Lý Độc Lập

Trước khi khởi chạy toàn bộ hệ thống điều khiển robot, bạn nên kiểm tra độc lập phần cứng camera để đảm bảo luồng hình ảnh đạt độ phân giải (640x480), tốc độ (30 FPS) và độ trễ thấp (< 35ms).

#### 1. Kiểm tra nhận diện phần cứng trên Linux (USB & Video Devices)
```bash
# A. Quét các thiết bị USB đang kết nối (tìm dòng Intel RealSense hoặc USB Camera):
lsusb

# B. Kiểm tra danh sách cổng video V4L2 có sẵn trong hệ thống:
ls -la /dev/video*
# hoặc dùng v4l2-ctl nếu đã cài v4l-utils (sudo apt install -y v4l-utils):
v4l2-ctl --list-devices

# C. Kiểm tra thiết bị Intel RealSense qua SDK chính hãng:
rs-enumerate-devices --short
```

> **Lưu ý về cổng USB 3.0:** Camera Intel RealSense D455/D435 đòi hỏi băng thông cao. Luôn cắm vào cổng **USB 3.0/3.1** (chân cắm màu xanh dương hoặc có ký hiệu `SS` - SuperSpeed). Nếu cắm nhầm cổng USB 2.0, camera sẽ bị giới hạn FPS hoặc không thể mở đồng thời RGB và Depth.

#### 2. Lệnh kiểm tra độc lập Camera Intel RealSense (D455 / D435 / D405)
Hệ thống tích hợp sẵn script kiểm định tự động toàn diện kiểm tra USB speed, FPS, độ trễ và nén JPEG mô phỏng WebXR:
```bash
cd /home/nguyen/ur_ws/pyroki_ursim
source .venv/bin/activate

# A. Chế độ kiểm tra nhanh tự động (chạy 3 giây benchmark và lưu ảnh snapshot vào logs/):
python tests/test_realsense.py

# B. Chế độ mở cửa sổ OpenCV GUI trực tiếp trên màn hình:
python tests/test_realsense.py --gui

# C. Kiểm tra hệ thống Camera đa phương thức (USB & Ethernet):
python tests/test_dual_camera.py

# D. Tùy chỉnh thông số kiểm tra:
python tests/test_realsense.py --width 640 --height 480 --fps 30 --duration 5.0 --gui
```
Ảnh snapshot kiểm tra chất lượng sẽ được tự động lưu tại: `logs/test_realsense_d455.jpg` (hoặc `logs/test_d455_rgb.jpg`).

#### 3. Lệnh kiểm tra độc lập USB Webcam / Camera V4L2 thông thường
Nếu sử dụng USB Webcam thông thường hoặc camera công nghiệp giao tiếp V4L2:
```bash
# Kiểm tra nhanh 1 frame qua OpenCV:
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print('-> Camera V4L2 hoạt động tốt! Độ phân giải:', frame.shape)
    cv2.imwrite('logs/test_usb_cam.jpg', frame)
else:
    print('-> LỖI: Không thể đọc frame từ /dev/video0')
cap.release()
"
```

#### 4. Lệnh kiểm tra Camera ảo 3D mô phỏng (Synthetic Wrist Camera)
Khi không có phần cứng camera thật, hệ thống hỗ trợ camera ảo 3D mô phỏng góc nhìn cổ tay Eye-in-Hand:
```bash
# Kiểm tra render 4 góc độ và lưu ảnh vào logs/:
python tests/test_camera.py

# Hoặc mở giao diện Web GUI tương tác trực quan (Cổng 8089):
python tests/test_camera.py --gui
```

---

### 6.5. Tổng Hợp Các Lệnh Vận Hành Thực Tế (Robot Thật & Camera Thật)

Tùy theo cấu hình phần cứng sẵn có tại nơi làm việc, bạn lựa chọn lệnh chạy tương ứng:

#### Trường hợp 1: Chạy CẢ HAI: Robot thật + Camera Intel RealSense (D455 / D435) [Khuyến nghị chuẩn]
Lệnh vận hành tiêu chuẩn cho toàn bộ hệ thống viễn thao Teleop đầy đủ:
```bash
cd /home/nguyen/ur_ws/pyroki_ursim
source .venv/bin/activate

# Tự động nhận diện camera Intel RealSense đầu tiên:
python main.py --robot-ip 10.42.0.242 --camera realsense

# Hoặc chỉ định rõ số Serial nếu máy tính cắm nhiều camera RealSense:
python main.py --robot-ip 10.42.0.242 --camera realsense --camera-serial 146322076196
```

#### Trường hợp 2: Chạy CẢ HAI qua Biến Môi Trường (Ngắn gọn)
```bash
export ROBOT_IP=10.42.0.242
python main.py --camera realsense
```

#### Trường hợp 3: Chạy CẢ HAI kèm Thu thập Dữ liệu Huấn luyện AI (Data Logging JSONL)
Tự động ghi lại toàn bộ chuỗi cử động tay VR Quest, góc 6 khớp robot, tọa độ TCP và trạng thái hút kẹp:
```bash
python main.py --robot-ip 10.42.0.242 --camera realsense --record logs/quest_runs/episode_real_01.jsonl
```

#### Trường hợp 4: Chạy kiểm tra độc lập Camera RealSense (Không cần bật Robot)
Phù hợp khi tinh chỉnh góc nhìn camera, ánh sáng hoặc gắn ngàm gắp:
```bash
python tests/test_realsense.py --gui
# hoặc kiểm tra kết nối Camera đa phương thức (USB & Ethernet):
python tests/test_dual_camera.py
```

#### Trường hợp 5: Chạy Robot thật độc lập (Sử dụng Camera mô phỏng 3D)
Phù hợp khi cần kiểm tra tay máy di chuyển trước hoặc chưa lắp camera lên cổ tay:
```bash
python main.py --robot-ip 10.42.0.242
```

#### Trường hợp 6: Chạy CẢ HAI với USB Webcam thông thường (OpenCV V4L2)
```bash
# Mặc định sử dụng /dev/video0:
python main.py --robot-ip 10.42.0.242 --camera real --camera-device 0
```

---

### 6.6. Bảng Tham Số Dòng Lệnh Đầy Đủ (CLI Options Reference)

| Cờ lệnh (Flag) | Kiểu dữ liệu | Giá trị mặc định | Mô tả chi tiết & Ví dụ |
| :--- | :---: | :---: | :--- |
| `--robot-ip` | `string` | `127.0.0.1` | Địa chỉ IP của Robot thật hoặc URSim. Nếu khác `127.0.0.1`, hệ thống bỏ qua Docker URSim và kết nối trực tiếp cổng RTDE 30003/30004 của robot thật. <br>Ví dụ: `--robot-ip 10.42.0.242` |
| `--camera` | `string` | `synthetic` | Lựa chọn nguồn camera cổ tay: <br>• `realsense` (hoặc `d455`, `d435`, `d405`): Dùng SDK `pyrealsense2` chuyên dụng (Khuyến nghị). <br>• `real`: Tự động tìm RealSense, nếu không có sẽ dùng OpenCV V4L2. <br>• `synthetic`: Camera ảo 3D mô phỏng. |
| `--camera-device` | `int` | `0` | Chỉ số Device ID của camera OpenCV V4L2 (`0` tương ứng `/dev/video0`, `2` tương ứng `/dev/video2`). |
| `--camera-serial` | `string` | `None` (Auto) | Mã số Serial number của camera Intel RealSense (lấy từ lệnh `rs-enumerate-devices`). Dùng khi máy tính gắn 2 camera RealSense trở lên. |
| `--record` | `string` | `None` | Kích hoạt bộ ghi log dữ liệu hành vi người dùng ra định dạng JSONL theo thời gian thực. <br>Ví dụ: `--record logs/pick_place_01.jsonl` |

---

### 6.7. Dấu Hiệu Nhận Biết Kết Nối Phần Cứng Thành Công Trên Terminal

Khi khởi chạy với robot thật và camera thật, terminal sẽ hiển thị chuỗi xác thực chuẩn sau:
```text
============================================================
 KHỞI ĐỘNG HỆ THỐNG VR TELEOP (META QUEST 3S) 
============================================================
[Hệ thống] Đang kết nối PyRoki (CPU JIT) và Robot UR3e tại IP: 10.42.0.242...
[Quest] WSS Server running at wss://0.0.0.0:8444
[Quest] WS Server (Plain) running at ws://0.0.0.0:8445
[Quest] HTTPS Server running at https://0.0.0.0:8443
[Setup] Connecting RTDEReceive...
[Setup] Connected. Joints: [0.001, -2.198, 1.896, -1.571, -1.570, 0.000]
[PyRoki] Khởi tạo bộ giải IK CPU cho UR3e...
[PyRoki] Bộ giải IK PyRoki CPU đã sẵn sàng!
[Hệ thống] Khởi tạo Driver Intel RealSense chuyên dụng (pyrealsense2, Serial: Auto)
[Camera Pipeline] Khởi động luồng truyền video cổ tay (FPS: 30)...
[ServoJ] RTDEControl connected: True
[Viser UI Event] Người dùng BẬT Teleop từ giao diện.
[Teleop] KÍCH HOẠT ĐIỀU KHIỂN.
```

---

### 6.8. Check-list 5 Phút An Toàn Trước Khi Bật Teleop Trên Robot Thật

Trước khi tích chọn **Bật Điều Khiển (Teleop)** trên Viser UI hoặc bấm **Nút B** trên tay cầm Quest, hãy rà soát nhanh 6 điểm an toàn sau:

1. [ ] **Chế độ Remote Control:** Góc trên bên phải màn hình PolyScope đang hiển thị biểu tượng Remote (hình mạng máy tính), không phải hình tay dạy cầm tay (Local).
2. [ ] **Thanh tốc độ PolyScope:** Kéo thanh trượt tốc độ xuống mức **10% – 20%** cho lần di chuyển đầu tiên.
3. [ ] **Khoảng cách vật lý:** Đảm bảo không có người hoặc vật cản nằm trong bán kính vươn 50cm của cánh tay UR3e.
4. [ ] **Nút dừng khẩn cấp:** Người vận hành hoặc người hỗ trợ luôn cầm Teach Pendant trên tay, ngón tay đặt sẵn trên nút E-Stop màu đỏ.
5. [ ] **Tọa độ Widget 3D trên Viser:** Kiểm tra mô hình robot 3D trên Viser (`http://localhost:8080`) đang hiển thị góc khớp và vị trí đầu TCP khớp hoàn toàn với vị trí thực tế của robot ngoài đời.
6. [ ] **Hiệu chuẩn hướng bàn máy:** Đứng thẳng hướng về phía robot, bóp giữ Grip và bấm Nút B để chốt mỏ neo hướng Tiến thẳng về phía bàn máy.

---

---

## 7. Kiểm Thử Hệ Thống & Quản Lý Dữ Liệu

### 1. Kiểm thử tự động Gắp Cà chua & Xuất HDF5
Chạy kiểm thử toàn bộ chu trình gắp thả và lưu dữ liệu:
```bash
python tests/test_simulated_pick_and_place.py
```
Kết quả kiểm tra:
- Tập tin Demonstration HDF5 được tạo tại: `logs/ai_training_data/test_grasp_sim/episode_X.hdf5`.
- Ảnh chụp camera ảo cổ tay tại thời điểm hoàn thành: `logs/simulated_wrist_camera_view.jpg`.

### 2. Kiểm thử Camera ảo Cổ tay (Synthetic Wrist Camera)
Kiểm tra khả năng render camera cổ tay ở 4 góc nhìn khác nhau:
```bash
python tests/test_camera.py
```
Các file ảnh kết xuất sẽ được lưu vào thư mục `logs/` (`test_cam_center.jpg`, `test_cam_left.jpg`,...).

Nếu muốn kiểm tra trực quan bằng thanh trượt trên giao diện Web Viser:
```bash
python tests/test_camera.py --gui
```
Sau đó mở trình duyệt tại `http://localhost:8089` để kéo tọa độ quả cà chua trực tiếp.

### 3. Kiểm thử Lớp Bảo Vệ Điểm Kỳ Dị (Singularity Guard)
```bash
python tests/test_singularity_guard.py
```
Kiểm tra tính toán ma trận Jacobian metric ($\sigma_{min}$) trên CPU ở cả tư thế an toàn lẫn tư thế duỗi thẳng cổ tay ($q_5 \approx 0$).

### 4. Kiểm định chất lượng Tập Dữ Liệu (Sanity Check)
```bash
python core/verify_dataset.py
```
Kiểm tra toàn vẹn các episode: xác minh không có giá trị `NaN`/`Inf`, kiểm tra độ lệch chu kỳ lấy mẫu `dt`, kiểm tra số frame và độ phân giải ảnh camera.

### 5. Tái lập Quỹ đạo Demonstration (Replay)
- **Offline Dry-Run (kiểm tra quỹ đạo an toàn trên Viser, không gửi lệnh sang robot):**
  ```bash
  python core/replay_episode.py --file logs/ai_training_data/test_grasp_sim/episode_0.hdf5
  ```
- **Live Replay (gửi lệnh điều khiển trực tiếp tới URSim / Robot thật):**
  ```bash
  python core/replay_episode.py --file logs/ai_training_data/test_grasp_sim/episode_0.hdf5 --execute --ip 127.0.0.1
  ```

---

## 8. Thông Số Kỹ Thuật Điều Khiển Chính

| Tham số | Giá trị | Giải thích kỹ thuật |
| :--- | :--- | :--- |
| **Tần số vòng lặp chính** | 20 Hz (`dt = 0.05`s) | Đồng bộ giữa WebXR, Viser, PyRoki và RTDE servoJ. |
| **Tần số Camera Stream** | 20 FPS | Luồng nền `CameraStreamPipeline` độc lập, triệt tiêu giật lag cho vòng lặp RTDE. |
| **Thời gian giải PyRoki IK (CPU)** | `~1.0` ms | Giải trên CPU bằng JAX JIT sau warm-up, Levenberg-Marquardt qua `jaxls`. |
| **Chi phí dáng mồi IK (rest_cost)** | `weight = 0.02` | Neo giữ nghiệm bám theo hạt giống `seed` của frame trước, đảm bảo tính liên tục. |
| **Hạt giống giải IK (seed)** | `q_var.with_value(seed)` | Khởi tạo nghiệm xấp xỉ bằng góc khớp hiện tại, hội tụ cực nhanh. |
| **ServoJ Lookahead Time** | `0.12` s | Nội suy dự đoán đa chu kỳ chuẩn Universal Robots, triệt tiêu vi rung giật. |
| **ServoJ Gain** | `250` | Độ cứng vị trí tối ưu trên robot vật lý, giúp chuyển động lướt êm ái không bị gắt cơ học. |
| **Vận tốc khớp tối đa (Rate Limiter)** | `1.0` rad/s ($\approx 57.3^\circ$/s) | Scaled Vector Limiter co giãn đồng bộ cả 6 khớp, phản hồi tức thì, bắt kịp chuyển động tay VR. |
| **Bán kính làm việc tối đa (r_max)** | `0.52` m | Mở rộng ranh giới hình trụ bảo vệ Descartes thêm 3cm, bao phủ trọn vẹn mép bàn thao tác. |
| **Độ cao an toàn tối thiểu (z_min)** | `0.035` m ($35\text{mm}$) | Cho phép hạ ngàm sát đáy khay gắp vật mỏng, vẫn cách mặt bàn gỗ $7.5\text{cm}$ chống va chạm. |
| **Hộp không gian làm việc (Bounding Box)** | $XY \pm 0.52\text{m}$, $Z \in [0.035, 0.60]\text{m}$ | Đồng bộ hóa không gian thao tác đa hướng cho cả tay cầm VR Quest và chuột PC Viser. |
| **Ngưỡng Singularity Warning & Danger** | $\sigma_{warn} = 0.042$, $\sigma_{danger} = 0.028$ | Continuous Damping hãm mượt (sàn 0.45) khi áp sát nguy hiểm, không cản trở chuyển động vươn xa. |
| **Dung sai vị trí IK (pos_err)** | `< 0.018` m ($18\text{mm}$) | Nâng tỷ lệ hội tụ IK vùng biên vươn xa lên >98%, triệt tiêu lỗi từ chối lệnh ảo. |
| **Ngưỡng phát hiện vặn xoắn khớp** | `2.2` rad | Chặn những bước nhảy đột ngột do đảo nhánh IK đột ngột gần điểm kỳ dị. |
| **Giới hạn khớp khuỷu (elbow_joint)** | `[+0.10, +2.40]` rad | Chống cùi chỏ tự gập sát đâm vào thân robot, đồng thời cho phép duỗi vươn xa mép bàn. |
| **Giới hạn khớp cổ tay (wrist_1_joint)** | `[-3.20, +0.10]` rad | Chống bẻ ngược cổ tay vào bắp tay, mở góc ngửa gắp đặt tự nhiên. |
| **Giới hạn khớp cổ tay (wrist_2_joint)** | `[-2.95, -0.20]` rad | Giới hạn an toàn chống tự va chạm liên kết cổ tay và ngàm. |
| **Giới hạn khớp nâng vai (shoulder_lift)** | `[-2.65, +0.20]` rad | Cho phép vươn tự nhiên, chống chúi lật cắm bắp tay xuống mặt bàn. |
| **Bán kính bảo vệ thân robot (r_min)** | `0.14` m | Cột trụ ảo xuyên suốt trục Z bảo vệ đế robot, mở rộng không gian thao tác tối đa. |
| **Thuật toán quy hoạch Home** | URScript 2 pha `movej` | Chạy trực tiếp ở tần số 500Hz trên URControl qua cổng Secondary 30002 (`a=0.8`, `v=0.5`). |
| **Bán kính bo cung waypoints Home** | `0.03` m | Bo cong góc mượt mà qua điểm trung gian nâng cao trước khi về Home. |
| **Tần số cập nhật Viser khi về Home** | 25 Hz (`dt = 0.04`s) | Cập nhật dáng robot Viser 3D mượt mà theo góc khớp thực trong suốt quá trình về Home. |
| **Khôi phục Teleop sau khi về Home** | Tự động (Zero-Click) | Tự động kết nối lại RTDE và reset mỏ neo tay cầm VR ngay khi robot về đích. |
| **Deadband tĩnh tay cầm VR** | 3 mm (`0.003` m) | Loại bỏ vi rung tay sinh học với cơ chế trừ mềm (`POS_DEADBAND`). |
| **Tỷ lệ truyền chuyển động tay (scale)** | `0.75` | Tỷ lệ 75% giúp robot bám sát chuyển động tay nhanh nhạy, giảm mỏi khi vung tay. |
| **Độ lệch tay cầm tối đa (max_delta_m)** | `0.22` m | Phạm vi vươn dài hơn mỗi nhịp bóp cò, thoải mái bao quát không gian làm việc. |
| **Bộ lọc làm mượt EMA (Alpha)** | `0.28` | Lọc tín hiệu thích ứng nhanh nhạy, giúp bám tay tức thì mà vẫn êm ái. |
| **Ngưỡng thời gian Auto-Snap** | 1.2 s | Tự động kéo XYZ về đầu tay máy và reset seed khi bị kẹt tư thế hoặc vướng vật cản. |
| **Khóa trục xoay cổ tay (Soft Snapping)** | $<55\%$ (cắt về 0), $55-75\%$ (làm mềm) | Triệt tiêu góc xoay phụ khi gập/chúc cổ tay, bảo vệ trục quay chính xác. |
| **Cơ chế Gripper** | Toggle Mode | Nhấn 1 lần nút A/X để kẹp/hút, nhấn lần nữa để thả; kèm rung Haptic. |
| **Khởi động URSim** | Tự động qua Docker | Tự kiểm tra cổng 29999 và chạy `docker start ursim` khi chạy `127.0.0.1`. |
| **Chuyển tiếp cổng ADB Quest** | Tự động chạy nền | Kiểm tra kết nối cáp USB mỗi 2.5s và tự động ánh xạ cổng sang `localhost`. |
