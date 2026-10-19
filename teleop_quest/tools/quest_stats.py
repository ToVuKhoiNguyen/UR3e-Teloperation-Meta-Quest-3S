#!/usr/bin/env python3
"""
quest_stats.py - In thong ke file ghi log WebXR (.jsonl):
- Tong so dong (goi tin)
- So lan bop co (trigger)
- Thoi luong va tan so goi tin (Hz)
- Do dai trung binh moi lan bop co
"""
import json
import sys
import os

def print_stats(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' khong ton tai.")
        sys.exit(1)

    total_lines = 0
    trigger_starts = 0
    trigger_ends = 0
    in_trigger = False
    current_clutch_frames = 0
    clutch_durations = []
    first_time = None
    last_time = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            total_lines += 1
            t = data.get("timestamp")
            if t is not None:
                if first_time is None:
                    first_time = t
                last_time = t

            # Kiem tra event bop co hoac trang thai trigger
            event = data.get("event")
            trigger = bool(
                data.get("trigger", False) or 
                (data.get("buttons") and data.get("buttons").get("trigger", False))
            )

            if event == "trigger_start":
                trigger_starts += 1
                in_trigger = True
                current_clutch_frames = 0
            elif event == "trigger_end":
                trigger_ends += 1
                if in_trigger:
                    clutch_durations.append(current_clutch_frames)
                in_trigger = False
            else:
                # Fallback edge-detection neu event chua duoc gan
                if trigger and not in_trigger:
                    trigger_starts += 1
                    in_trigger = True
                    current_clutch_frames = 0
                elif not trigger and in_trigger:
                    trigger_ends += 1
                    if in_trigger:
                        clutch_durations.append(current_clutch_frames)
                    in_trigger = False

            if in_trigger:
                current_clutch_frames += 1

    if in_trigger:
        clutch_durations.append(current_clutch_frames)

    print("=" * 60)
    print(f"THỐNG KÊ FILE LOG QUEST: {file_path}")
    print("=" * 60)
    print(f"Tổng số dòng (gói tin):       {total_lines}")
    print(f"Số lần bóp cò (trigger):      {trigger_starts}")
    if first_time is not None and last_time is not None:
        duration = max(0.0, last_time - first_time)
        fps = (total_lines / duration) if duration > 0.05 else 0.0
        print(f"Thời lượng ghi:               {duration:.2f} giây (~{fps:.1f} Hz)")
    if clutch_durations:
        avg_frames = sum(clutch_durations) / len(clutch_durations)
        print(f"Thời lượng bóp cò trung bình: {avg_frames:.1f} frames/lần")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Cách dùng: python3 {sys.argv[0]} <duong_dan_file.jsonl>")
        sys.exit(1)
    print_stats(sys.argv[1])
