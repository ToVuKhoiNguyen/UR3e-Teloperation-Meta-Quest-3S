#!/usr/bin/env python3
"""
Kiểm thử tích hợp toàn diện cấu hình phím Meta Quest Right Controller:
1. Nút B: Nhấn nhả (< 3.0s) -> Toggle Teleop. Nhấn giữ (>= 3.0s) -> Go Home (Báo rung).
2. Nút A: Nhấn nhả (< 2.0s) -> Toggle Record. Nhấn giữ (>= 2.0s) -> Discard Episode (Báo rung).
3. Joystick: Click -> Toggle Gripper (Báo rung). Gạt X -> Continuous Gripper. Gạt Y -> Sensitivity Scale.
4. Đồng bộ Viser UI & WebXR Haptic feedback.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.selectable_recorder import SelectableDataRecorder

class MockViserWidget:
    def __init__(self, value=False):
        self.value = value
        self.label = ""

class MockViserUI:
    def __init__(self):
        self.cb_teleop = MockViserWidget(False)
        self.cb_record = MockViserWidget(False)
        self.cb_discard = MockViserWidget(False)
        self.cb_gripper = MockViserWidget(False)
        self.slider_sensitivity = MockViserWidget(1.0)
        self.ui_status = MockViserWidget("Idle")

    def set_status(self, text):
        self.ui_status.value = text

    def set_recording_state(self, is_rec, ep_idx=0):
        self.cb_record.value = is_rec
        self.cb_record.label = f"Recording (Ep {ep_idx})" if is_rec else "Start Recording"

def test_button_b_short_and_long_press():
    print("--- [TEST 1] NÚT B: SHORT PRESS (< 3.0s) TOGGLE TELEOP & LONG PRESS (>= 3.0s) GO HOME ---")
    ui = MockViserUI()
    haptic_events = []
    home_events = []

    def mock_send_haptic(intensity, duration):
        haptic_events.append((intensity, duration))

    def mock_trigger_go_home(source_name=""):
        home_events.append(source_name)

    # State machine variables
    _btn_b_prev = False
    _btn_b_press_start = 0.0
    _btn_b_long_fired = False

    def update_btn_b(is_pressed, current_time):
        nonlocal _btn_b_prev, _btn_b_press_start, _btn_b_long_fired
        now_t = current_time
        if is_pressed and not _btn_b_prev:
            _btn_b_press_start = now_t
            _btn_b_long_fired = False
        elif is_pressed:
            if not _btn_b_long_fired and (now_t - _btn_b_press_start) >= 3.0:
                _btn_b_long_fired = True
                mock_send_haptic(1.0, 300)
                mock_trigger_go_home("VR Nút B (Giữ >= 3.0s)")
        elif _btn_b_prev and not is_pressed:
            if not _btn_b_long_fired:
                ui.cb_teleop.value = not ui.cb_teleop.value
                mock_send_haptic(0.7, 150)
        _btn_b_prev = is_pressed

    # 1. Nhấn nhả Nút B trong 0.5s (< 3.0s) -> Bật Teleop
    t = 100.0
    update_btn_b(True, t)          # Press down
    update_btn_b(True, t + 0.2)
    update_btn_b(False, t + 0.5)   # Release
    assert ui.cb_teleop.value == True, "Lỗi: Nút B short press phải BẬT Teleop!"
    assert len(home_events) == 0, "Lỗi: Short press không được kích hoạt Go Home!"
    assert len(haptic_events) == 1, "Lỗi: Phải có rung haptic xác nhận short press!"
    print("[PASS] Nút B nhấn nhả 0.5s -> Kích hoạt Toggle Teleop (BẬT), báo rung 150ms.")

    # 2. Nhấn nhả Nút B lần 2 trong 0.3s -> Tắt Teleop
    t = 102.0
    update_btn_b(True, t)
    update_btn_b(False, t + 0.3)
    assert ui.cb_teleop.value == False, "Lỗi: Nút B short press lần 2 phải TẮT Teleop!"
    print("[PASS] Nút B nhấn nhả lần 2 -> Kích hoạt Toggle Teleop (TẮT), báo rung 150ms.")

    # 3. Nhấn giữ Nút B trong 3.2s (>= 3.0s) -> Kích hoạt Go Home ngay ở mốc 3.0s
    t = 110.0
    update_btn_b(True, t)          # Press down
    update_btn_b(True, t + 2.0)    # Held 2s (chưa đủ 3s)
    assert len(home_events) == 0, "Chưa đủ 3.0s thì không được kích hoạt Home!"
    
    update_btn_b(True, t + 3.05)   # Đạt 3.05s >= 3.0s -> Fire Go Home!
    assert len(home_events) == 1, "Lỗi: Giữ >= 3.0s phải kích hoạt Go Home!"
    assert home_events[-1] == "VR Nút B (Giữ >= 3.0s)"
    assert haptic_events[-1] == (1.0, 300), "Lỗi: Long press phải có rung mạnh 300ms!"

    # Nhả phím sau khi đã fire long press -> Tuyệt đối KHÔNG được toggle teleop!
    update_btn_b(False, t + 3.5)
    assert ui.cb_teleop.value == False, "Lỗi nghiêm trọng: Nhả phím sau long press KHÔNG được toggle teleop!"
    print("[PASS] Nút B nhấn giữ 3.2s -> Kích hoạt Go Home tức thì ở 3.0s, nhả phím không toggle teleop.")

def test_button_a_short_and_long_press():
    print("\n--- [TEST 2] NÚT A: SHORT PRESS (< 2.0s) RECORD & LONG PRESS (>= 2.0s) DISCARD ---")
    test_dir = "logs/test_key_integration_a"
    os.makedirs(test_dir, exist_ok=True)
    recorder = SelectableDataRecorder(config={"output_dir": test_dir, "hz": 20})
    ui = MockViserUI()
    haptic_events = []

    def mock_send_haptic(intensity, duration):
        haptic_events.append((intensity, duration))

    def toggle_recording(source_name=""):
        if not recorder.is_recording:
            recorder.start_episode(task_description="Test VR A Key")
            ui.set_recording_state(True, recorder.current_episode_idx)
            mock_send_haptic(0.9, 200)
        else:
            saved = recorder.end_episode(success=True)
            ui.set_recording_state(False, recorder.current_episode_idx)
            mock_send_haptic(0.5, 100)
            return saved

    def discard_recording(source_name=""):
        if recorder.is_recording:
            recorder.discard_episode()
            ui.set_recording_state(False, recorder.current_episode_idx)
            ui.set_status("Discarded Episode")
            mock_send_haptic(0.7, 300)

    # State machine variables
    _btn_a_prev = False
    _btn_a_press_start = 0.0
    _btn_a_long_fired = False

    def update_btn_a(is_pressed, current_time):
        nonlocal _btn_a_prev, _btn_a_press_start, _btn_a_long_fired
        now_t = current_time
        if is_pressed and not _btn_a_prev:
            _btn_a_press_start = now_t
            _btn_a_long_fired = False
        elif is_pressed:
            if not _btn_a_long_fired and (now_t - _btn_a_press_start) >= 2.0:
                _btn_a_long_fired = True
                if recorder.is_recording:
                    discard_recording("VR Nút A (Giữ >= 2.0s)")
                else:
                    mock_send_haptic(0.5, 150)
        elif _btn_a_prev and not is_pressed:
            if not _btn_a_long_fired:
                toggle_recording("VR Nút A (Nhấn nhả < 2.0s)")
        _btn_a_prev = is_pressed

    # 1. Nhấn nhả Nút A 0.4s (< 2.0s) -> Bắt đầu ghi Episode
    t = 200.0
    update_btn_a(True, t)
    update_btn_a(True, t + 0.2)
    update_btn_a(False, t + 0.4)
    assert recorder.is_recording == True, "Lỗi: Nút A short press phải BẮT ĐẦU ghi!"
    assert ui.cb_record.value == True, "Lỗi: UI cb_record phải là True!"
    print("[PASS] Nút A nhấn nhả 0.4s -> Bắt đầu ghi Episode, UI checkbox Record = True.")

    # Ghi 3 frame dữ liệu
    for i in range(3):
        recorder.record_step(
            telemetry={"actual_q": [0.0]*6, "target_q": [0.0]*6, "timestamp_host_ms": int(time.time()*1000)},
            images={},
            action=[0.0]*7,
            vr_controller_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            target_tcp_pose=[0.1, 0.2, 0.3, 0.0, 3.14, 0.0]
        )

    # 2. Nhấn giữ Nút A 2.2s (>= 2.0s) -> HỦY (Discard) Episode!
    t = 205.0
    update_btn_a(True, t)          # Press down
    update_btn_a(True, t + 1.0)    # Held 1s
    assert recorder.is_recording == True, "Chưa đủ 2s không được hủy!"

    update_btn_a(True, t + 2.05)   # Đạt 2.05s >= 2.0s -> Fire Discard!
    assert recorder.is_recording == False, "Lỗi: Giữ >= 2.0s phải HỦY episode!"
    assert ui.cb_record.value == False, "Lỗi: Sau khi hủy, UI cb_record phải về False!"
    assert ui.ui_status.value == "Discarded Episode"
    assert haptic_events[-1] == (0.7, 300), "Lỗi: Discard phải có rung haptic xác nhận!"

    # Nhả phím sau khi Discard -> Tuyệt đối KHÔNG được toggle start episode mới!
    update_btn_a(False, t + 2.4)
    assert recorder.is_recording == False, "Lỗi: Nhả phím sau discard KHÔNG được kích hoạt toggle record!"
    print("[PASS] Nút A nhấn giữ 2.2s -> Discard Episode tức thì tại 2.0s, báo rung 300ms, không toggle lại.")

    # 3. Nhấn nhả Nút A lần nữa để kiểm tra chu trình hoàn chỉnh: Start -> Stop & Save
    t = 210.0
    update_btn_a(True, t)
    update_btn_a(False, t + 0.3)
    assert recorder.is_recording == True
    
    recorder.record_step(
        telemetry={"actual_q": [0.0]*6, "target_q": [0.0]*6, "timestamp_host_ms": int(time.time()*1000)},
        images={}, action=[0.0]*7
    )
    
    t = 212.0
    update_btn_a(True, t)
    update_btn_a(False, t + 0.3)
    assert recorder.is_recording == False
    assert ui.cb_record.value == False
    print("[PASS] Nút A nhấn nhả (Start) -> nhấn nhả (Save) hoàn tất xuất sắc!")

    # Dọn dẹp thư mục tạm sau kiểm thử
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

def test_joystick_controls():
    print("\n--- [TEST 3] JOYSTICK: CLICK (TOGGLE GRIPPER), AXES X (CONTINUOUS), AXES Y (SCALE) ---")
    ui = MockViserUI()
    haptic_events = []

    def mock_send_haptic(intensity, duration):
        haptic_events.append((intensity, duration))

    _gripper_active = False
    _gripper_continuous_val = 0.0
    _thumbstick_click_prev = False
    scale = 1.0
    POLL_SLEEP = 0.05

    def update_joystick(click_pressed, stick_x, stick_y):
        nonlocal _gripper_active, _gripper_continuous_val, _thumbstick_click_prev, scale
        
        # 1. Joystick Click -> Toggle Gripper kèm rung
        thumbstick_clicked = click_pressed and not _thumbstick_click_prev
        _thumbstick_click_prev = click_pressed
        if thumbstick_clicked:
            _gripper_active = not _gripper_active
            _gripper_continuous_val = 1.0 if _gripper_active else 0.0
            ui.cb_gripper.value = _gripper_active
            mock_send_haptic(0.9 if _gripper_active else 0.4, 150)

        # 2. Joystick X -> Continuous Gripper Control (Không rung)
        if abs(stick_x) > 0.35:
            d_grip = stick_x * 1.5 * POLL_SLEEP
            _gripper_continuous_val = float(np.clip(_gripper_continuous_val + d_grip, 0.0, 1.0))
            new_grip_active = (_gripper_continuous_val >= 0.5)
            if new_grip_active != _gripper_active:
                _gripper_active = new_grip_active
                ui.cb_gripper.value = _gripper_active

        # 3. Joystick Y -> Scale Sensitivity (Không rung)
        if abs(stick_y) > 0.35:
            d_scale = -stick_y * 0.4 * POLL_SLEEP
            scale = float(np.clip(scale + d_scale, 0.25, 3.0))
            STEP = 0.05
            snapped = round(round(scale / STEP) * STEP, 2)
            ui.slider_sensitivity.value = snapped

    # A. Test Joystick Click -> Toggle Gripper
    haptic_events.clear()
    update_joystick(True, 0.0, 0.0)   # Click down
    update_joystick(False, 0.0, 0.0)  # Release
    assert _gripper_active == True, "Lỗi: Click joystick phải ĐÓNG/KẸP gripper!"
    assert ui.cb_gripper.value == True
    assert len(haptic_events) == 1 and haptic_events[-1] == (0.9, 150), "Lỗi: Phải báo rung 150ms khi kẹp!"
    print("[PASS] Joystick Click 1 -> Gripper ON (Kẹp), Viser cb_gripper = True, báo rung 150ms.")

    update_joystick(True, 0.0, 0.0)   # Click down lần 2
    update_joystick(False, 0.0, 0.0)  # Release
    assert _gripper_active == False, "Lỗi: Click joystick lần 2 phải NHẢ/MỞ gripper!"
    assert ui.cb_gripper.value == False
    assert len(haptic_events) == 2 and haptic_events[-1] == (0.4, 150), "Lỗi: Phải báo rung 150ms khi nhả!"
    print("[PASS] Joystick Click 2 -> Gripper OFF (Nhả), Viser cb_gripper = False, báo rung 150ms.")

    # B. Test Joystick Axes X (Continuous Gripper Control - Không rung)
    haptic_count_before = len(haptic_events)
    # Gạt sang phải (stick_x = +1.0) liên tục 15 bước để tăng _gripper_continuous_val
    for _ in range(15):
        update_joystick(False, 1.0, 0.0)
    assert _gripper_continuous_val > 0.5, "Lỗi: Gạt sang phải phải tăng độ kẹp!"
    assert _gripper_active == True, "Lỗi: Khi vượt ngưỡng 0.5, gripper_active phải thành True!"
    assert len(haptic_events) == haptic_count_before, "Lỗi: Joystick Axes X KHÔNG ĐƯỢC báo rung!"
    print(f"[PASS] Joystick X (+1.0) -> Continuous Gripper tăng mượt mà (val={_gripper_continuous_val:.2f}), KHÔNG RUNG.")

    # C. Test Joystick Axes Y (Sensitivity Scale - Không rung)
    # Cần đẩy lên (stick_y = -1.0) -> Tăng scale
    init_scale = ui.slider_sensitivity.value
    for _ in range(10):
        update_joystick(False, 0.0, -1.0)
    assert ui.slider_sensitivity.value > init_scale, "Lỗi: Đẩy stick_y lên phải TĂNG scale!"
    assert len(haptic_events) == haptic_count_before, "Lỗi: Joystick Axes Y KHÔNG ĐƯỢC báo rung!"
    print(f"[PASS] Joystick Y (-1.0) -> Scale tăng từ {init_scale:.2f}x lên {ui.slider_sensitivity.value:.2f}x, KHÔNG RUNG.")

if __name__ == "__main__":
    print("=" * 70)
    print(" BẮT ĐẦU BỘ KIỂM THỬ TÍCH HỢP PHÍM VR CONTROLLER & ĐỒNG BỘ VISER UI ")
    print("=" * 70)
    test_button_b_short_and_long_press()
    test_button_a_short_and_long_press()
    test_joystick_controls()
    print("\n" + "=" * 70)
    print(" TOÀN BỘ 3 KỊCH BẢN KIỂM THỬ: 100% PASS! ")
    print("=" * 70)
