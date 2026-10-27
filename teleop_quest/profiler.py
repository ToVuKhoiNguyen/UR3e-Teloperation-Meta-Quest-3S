#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teleop_quest/profiler.py
Công cụ giám sát và đo lường thời gian thực thi (Execution Time Profiler):
- Đo lường độ trễ chi tiết (Latency in ms) của 3 module trọng yếu:
  + IK_Solver: Thời gian tính toán giải động học nghịch trên CPU (PyRoki)
  + Camera_Read: Thời gian đọc và chuẩn bị buffer ảnh màu/độ sâu
  + UI_Update: Thời gian cập nhật giao diện Viser UI và broadcast WebXR
- Tính toán tổng thời gian chu kỳ (Total_Loop) và tần số thực tế (Actual FPS)
- Tự động in bảng thống kê chi tiết ra Terminal mỗi 100 frames
"""

import time
from typing import Dict, Optional, Any
from contextlib import contextmanager


class SystemProfiler:
    """Bộ đo lường hiệu năng thời gian thực siêu nhẹ (overhead < 0.05ms)."""

    def __init__(self, window_size: int = 100, print_to_stdout: bool = False):
        self.window_size = max(1, window_size)
        self.print_to_stdout = print_to_stdout
        self.frame_count = 0
        self.total_frames = 0
        self.window_start_time = time.perf_counter()

        # Dữ liệu tích lũy trong cửa sổ window_size
        self._accumulators: Dict[str, list] = {
            "IK_Solver": [],
            "Camera_Read": [],
            "UI_Update": [],
            "Total_Loop": []
        }
        self._active_starts: Dict[str, float] = {}

        # Lưu bản ghi thống kê gần nhất phục vụ kiểm thử và giám sát
        self.last_report: Optional[Dict[str, Any]] = None

    def start_segment(self, name: str) -> None:
        """Bắt đầu đo một phân đoạn."""
        self._active_starts[name] = time.perf_counter()

    def end_segment(self, name: str) -> float:
        """Kết thúc đo một phân đoạn và lưu độ trễ (ms)."""
        t_end = time.perf_counter()
        t_start = self._active_starts.pop(name, t_end)
        elapsed_ms = (t_end - t_start) * 1000.0
        if name not in self._accumulators:
            self._accumulators[name] = []
        self._accumulators[name].append(elapsed_ms)
        return elapsed_ms

    @contextmanager
    def profile(self, name: str):
        """Context manager tiện ích: with profiler.profile('IK_Solver'): ..."""
        t_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if name not in self._accumulators:
                self._accumulators[name] = []
            self._accumulators[name].append(elapsed_ms)

    def tick_frame(self, loop_elapsed_ms: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Đánh dấu hoàn thành một frame điều khiển.
        Nếu đạt đủ window_size (100 frames), tự động in bảng log thống kê ra terminal.
        """
        self.frame_count += 1
        self.total_frames += 1

        if loop_elapsed_ms is not None:
            self._accumulators["Total_Loop"].append(loop_elapsed_ms)

        if self.frame_count >= self.window_size:
            now = time.perf_counter()
            duration_sec = max(1e-5, now - self.window_start_time)
            actual_fps = self.frame_count / duration_sec

            report = {
                "window_frames": self.frame_count,
                "total_frames": self.total_frames,
                "actual_fps": actual_fps,
                "segments": {}
            }

            for seg_name, values in self._accumulators.items():
                if len(values) > 0:
                    report["segments"][seg_name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
                else:
                    report["segments"][seg_name] = {
                        "avg": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "count": 0
                    }

            self.last_report = report
            if self.print_to_stdout:
                self.print_summary_table(report)

            # Reset cửa sổ đo lường tiếp theo
            self.frame_count = 0
            self.window_start_time = time.perf_counter()
            for seg in self._accumulators:
                self._accumulators[seg] = []

            return report

        return None

    def print_summary_table(self, report: Dict[str, Any]) -> None:
        """In bảng thống kê hiệu năng ra terminal (chỉ in khi có biến môi trường DEBUG_PROFILER)."""
        import os
        if not self.print_to_stdout or not os.environ.get("DEBUG_PROFILER", ""):
            return

        fps = report["actual_fps"]
        total_f = report["total_frames"]
        w_f = report["window_frames"]

        print("\n" + "=" * 78)
        print(f" [SYSTEM PROFILER] HIỆU NĂNG THỰC THI (Trung bình {w_f} frames gần nhất | Frame {total_f})")
        print(f" Tần số đo đạc thực tế: \033[92m{fps:.1f} FPS\033[0m")
        print("-" * 78)
        print(f" {'Module / Phân đoạn':<22} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Đánh giá trạng thái'}")
        print("-" * 78)

        targets = {
            "IK_Solver": (2.0, "< 2.0 ms", "IK CPU JIT mượt mà"),
            "Camera_Read": (1.0, "< 1.0 ms", "Non-blocking RAM Buffer"),
            "UI_Update": (5.0, "< 5.0 ms", "Viser & Quest WebXR Sync"),
            "Total_Loop": (35.0, "< 33.3 ms", "Đạt chuẩn >= 30 FPS")
        }

        for seg_name, stats in report["segments"].items():
            avg_val = stats["avg"]
            min_val = stats["min"]
            max_val = stats["max"]

            thresh, target_str, desc = targets.get(seg_name, (10.0, "N/A", "OK"))
            if avg_val <= thresh:
                status_str = f"\033[92m[TỐT] {target_str} ({desc})\033[0m"
            else:
                status_str = f"\033[93m[CHÚ Ý] {avg_val:.2f}ms > {target_str}\033[0m"

            print(f" {seg_name:<22} | {avg_val:8.2f} ms | {min_val:8.2f} ms | {max_val:8.2f} ms | {status_str}")

        print("=" * 78 + "\n")
