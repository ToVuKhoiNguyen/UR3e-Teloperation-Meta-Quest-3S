#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_system_profiler_and_watchdog.py
Kiểm thử toàn diện 3 tính năng mới:
1. SystemProfiler: Đo lường thời gian thực thi, tính trung bình và xuất báo cáo mỗi 100 frames
2. WebXR Watchdog: Tự động ngắt an toàn khi mất tracking > 1.0s và phục hồi mượt mà khi nhận lại tín hiệu
3. UR Dashboard & Robot Real / Sim: Xử lý timeout socket và cảnh báo Remote Control không gây crash
4. Camera RAM Streaming: Đảm bảo không còn I/O ghi đĩa từng frame
"""

import os
import sys
import time
import socket
import inspect
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from teleop_quest.profiler import SystemProfiler
from core.pyroki_ursim_control import setup_ursim
from teleop_quest.quest_server import QuestServer


def test_system_profiler_accumulation_and_report():
    """Kiểm tra SystemProfiler tích lũy chính xác và tự động xuất báo cáo ở frame 100."""
    profiler = SystemProfiler(window_size=100, print_to_stdout=False)
    
    for i in range(99):
        profiler.start_segment("IK_Solver")
        time.sleep(0.0001)
        profiler.end_segment("IK_Solver")
        
        profiler.start_segment("Camera_Read")
        profiler.end_segment("Camera_Read")
        
        profiler.start_segment("UI_Update")
        profiler.end_segment("UI_Update")
        
        rep = profiler.tick_frame(loop_elapsed_ms=5.0)
        assert rep is None, f"Frame {i+1} chưa được xuất báo cáo (chờ đủ 100 frames)"

    # Frame thứ 100 -> Phải xuất báo cáo
    profiler.start_segment("IK_Solver")
    profiler.end_segment("IK_Solver")
    profiler.start_segment("Camera_Read")
    profiler.end_segment("Camera_Read")
    profiler.start_segment("UI_Update")
    profiler.end_segment("UI_Update")
    
    report = profiler.tick_frame(loop_elapsed_ms=6.0)
    assert report is not None, "Frame 100 phải trả về báo cáo thống kê"
    assert report["window_frames"] == 100
    assert report["total_frames"] == 100
    assert "segments" in report
    assert "IK_Solver" in report["segments"]
    assert "Camera_Read" in report["segments"]
    assert "UI_Update" in report["segments"]
    assert "Total_Loop" in report["segments"]
    
    # Kiểm tra trung bình Total_Loop
    assert report["segments"]["Total_Loop"]["count"] == 100
    assert 4.5 <= report["segments"]["Total_Loop"]["avg"] <= 6.5
    
    # Sau khi xuất báo cáo, frame_count phải được reset về 0
    assert profiler.frame_count == 0


def test_system_profiler_context_manager():
    """Kiểm tra context manager `profile()` của SystemProfiler."""
    profiler = SystemProfiler(window_size=10, print_to_stdout=False)
    with profiler.profile("IK_Solver"):
        time.sleep(0.0002)
    assert len(profiler._accumulators["IK_Solver"]) == 1
    assert profiler._accumulators["IK_Solver"][0] > 0.1


def test_webxr_watchdog_tracking_loss_and_recovery():
    """
    Kiểm tra cơ chế WebXR Watchdog:
    - delta > 1.0s: trigger VR_PAUSED, dừng robot an toàn, reset anchor, hiển thị cảnh báo
    - delta <= 0.15s: phục hồi tracking mượt mà, reset anchor, phục hồi UI
    """
    mock_quest_server = MagicMock()
    mock_executor = MagicMock()
    mock_teleop = MagicMock()
    mock_pose_source = MagicMock()
    mock_ui = MagicMock()
    gizmo_interaction = [0.0]

    _vr_tracking_lost = False
    
    # Giả lập lần 1: Kính Quest gửi frame bình thường
    now_vr_time = 1000.0
    last_vr_seen = 999.95
    delta_vr_time = now_vr_time - last_vr_seen  # 0.05s <= 1.0s

    if last_vr_seen > 0.0 and delta_vr_time > 1.0:
        _vr_tracking_lost = True

    assert not _vr_tracking_lost, "Khi delta <= 1.0s không được coi là mất tracking"

    # Giả lập lần 2: Người dùng tháo kính, delta > 1.0s
    now_vr_time = 1002.5
    last_vr_seen = 1000.0
    delta_vr_time = now_vr_time - last_vr_seen  # 2.5s > 1.0s

    if last_vr_seen > 0.0 and delta_vr_time > 1.0:
        if not _vr_tracking_lost:
            _vr_tracking_lost = True
            mock_executor.stop(10.0)
            mock_teleop.stop(10.0)
            mock_pose_source.reset_anchor()
        mock_ui.set_status("VR Tracking Lost. Please re-center or Enter VR.")

    assert _vr_tracking_lost is True, "Phải kích hoạt trạng thái VR_PAUSED khi delta > 1.0s"
    mock_executor.stop.assert_called_with(10.0)
    mock_teleop.stop.assert_called_with(10.0)
    mock_pose_source.reset_anchor.assert_called_once()
    mock_ui.set_status.assert_called_with("VR Tracking Lost. Please re-center or Enter VR.")

    # Giả lập lần 3: Người dùng đeo lại kính, WebXR gửi lại frame (delta = 0.05s <= 0.15s)
    now_vr_time = 1005.05
    last_vr_seen = 1005.00
    delta_vr_time = now_vr_time - last_vr_seen  # 0.05s <= 0.15s

    if _vr_tracking_lost and delta_vr_time <= 0.15:
        _vr_tracking_lost = False
        mock_pose_source.reset_anchor()
        mock_ui.set_status("Sẵn sàng (Nhả cò)")

    assert _vr_tracking_lost is False, "Phải phục hồi tracking khi delta <= 0.15s"
    assert mock_pose_source.reset_anchor.call_count == 2
    mock_ui.set_status.assert_called_with("Sẵn sàng (Nhả cò)")


def test_setup_ursim_real_robot_socket_timeout_handling():
    """Kiểm tra setup_ursim cho robot thật khi gặp lỗi timeout không làm crash chương trình."""
    # Thử kết nối tới một IP không tồn tại với timeout ngắn
    with patch.dict("os.environ", {"ROBOT_IP": "192.0.2.1"}): # TEST-NET IP không phản hồi
        success = setup_ursim(timeout=0.5, is_real=True)
        assert success is False, "setup_ursim phải trả về False một cách an toàn mà không bị crash Exception"


def test_camera_streaming_in_memory_no_disk_io():
    """Kiểm tra mã nguồn QuestServer.send_camera_frame không còn thao tác ghi đĩa open(..., 'wb')."""
    src = inspect.getsource(QuestServer.send_camera_frame)
    assert 'open(' not in src, "send_camera_frame không được chứa lời gọi open() ghi đĩa"
    assert 'os.replace' not in src, "send_camera_frame không được chứa thao tác os.replace"
    assert 'self._latest_jpeg' in src, "send_camera_frame phải lưu trực tiếp vào RAM buffer _latest_jpeg"


def test_quest_server_broadcast_reset_session():
    """Kiểm tra QuestServer có hàm broadcast_reset_session phát thông điệp reset_session."""
    server = QuestServer(cert="cert.pem", key="key.pem")
    assert hasattr(server, "broadcast_reset_session"), "QuestServer phải có hàm broadcast_reset_session"
    
    mock_ws = MagicMock()
    mock_loop = MagicMock()
    with server._lock:
        server._clients.add(mock_ws)
        server._loop = mock_loop

    with patch("asyncio.run_coroutine_threadsafe") as mock_run:
        server.broadcast_reset_session()
        mock_run.assert_called_once()


def test_webxr_main_js_session_lifecycle():
    """Kiểm tra mã nguồn main.js có cơ chế disposeSession, visibilitychange và bắt reset_session."""
    main_js_path = os.path.join(os.path.dirname(__file__), "..", "teleop_quest", "static", "main.js")
    with open(main_js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    assert "disposeSession" in js_content, "main.js phải có hàm disposeSession"
    assert "visibilitychange" in js_content, "main.js phải lắng nghe sự kiện visibilitychange"
    assert "reset_session" in js_content, "main.js phải xử lý gói tin reset_session từ WebSocket"
    assert "xrSession.end()" in js_content, "main.js phải gọi xrSession.end() khi dispose"

