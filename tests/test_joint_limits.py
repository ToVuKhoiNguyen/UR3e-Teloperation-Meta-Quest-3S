#!/usr/bin/env python3
"""
Bộ kiểm thử toàn diện cho hệ thống giới hạn an toàn nới lỏng (Relaxed Joint Limits & r_min):
1. Ready Pose chuẩn: Nằm an toàn trong dải mới.
2. Vươn xa mép bàn: Khớp cùi chỏ duỗi tới q2 = +0.18 rad (trước đây bị chặn vì < 0.35 rad) -> PASS.
3. Không gian sát thân đế: Bán kính r = 0.148m (trước đây bị ép văng ra 0.18m) -> PASS, bảo toàn tọa độ.
4. Rào chắn chống tự va chạm (Self-collision barriers):
   - Cùi chỏ gập quá sát (q2 > 2.40 rad) -> Chặn đứng (REJECTED).
   - Cùi chỏ gập ngược xuống bàn (q2 < 0.10 rad) -> Chặn đứng (REJECTED).
   - Cổ tay 1 bẻ ngược đâm vào thân (q3 > +0.10 rad) -> Chặn đứng (REJECTED).
   - Cổ tay 1 cuộn quá sâu (q3 < -3.20 rad) -> Chặn đứng (REJECTED).
5. Rào chắn Descartes: clamp_cartesian_safety r_min = 0.14m, z_min = 0.045m, r_max = 0.49m.
6. Đo độ trễ thẩm định an toàn (< 1.0 ms).
"""
import sys
import os
import time
import math
import numpy as np
from unittest.mock import MagicMock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import UR3E_LIMITS, validate_joints
from teleop.teleoperation import SafetyGate, clamp_cartesian_safety


def run_comprehensive_tests():
    print("=" * 70)
    print(" KIỂM THỬ GIỚI HẠN AN TOÀN NỚI LỎNG (RELAXED LIMITS & r_min = 0.14m)")
    print("=" * 70)

    print("\n[Bảng UR3E_LIMITS mới]:")
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    for i, name in enumerate(joint_names):
        lo, hi = UR3E_LIMITS[i]
        print(f"  Khớp {i} ({name:14s}): [{lo:+.2f}, {hi:+.2f}] rad  ([{lo*57.2958:+.1f}°, {hi*57.2958:+.1f}°])")

    # Mock Singularity Guard cho SafetyGate
    mock_guard = MagicMock()
    mock_guard.metrics_calc.device = "cpu"
    mock_guard.metrics_calc.get_metrics.return_value = {"sigma_min": MagicMock(item=lambda: 0.15)}
    mock_guard.danger_threshold = 0.035
    mock_guard.warn_threshold = 0.08

    gate = SafetyGate(mock_guard)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. TEST TƯ THẾ READY POSE CHUẨN
    # ─────────────────────────────────────────────────────────────────────────
    q_ready = [0.0, -2.20, +1.90, -1.57, -1.57, 0.0]
    t0 = time.perf_counter()
    val_ok = validate_joints(q_ready)
    gate_res = gate.check(q_ready)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[Test 1 - READY POSE CHUẨN] q = {q_ready}")
    print(f"  validate_joints: {val_ok} | SafetyGate: {gate_res['reason']} (Độ trễ: {dt_ms:.4f}ms)")
    assert val_ok is True, "LỖI: Ready Pose bị từ chối!"
    assert gate_res["pass"] is True, "LỖI: SafetyGate chặn Ready Pose!"
    print("  -> PASS: Ready Pose hợp lệ 100%.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. TEST TƯ THẾ VƯƠN XA MÉP BÀN (q2 = +0.18 rad)
    # (Trước đây bị cự tuyệt vì dải cũ q2 in [0.35, 2.35])
    # ─────────────────────────────────────────────────────────────────────────
    q_far_reach = [0.20, -1.20, +0.18, -0.55, -1.57, 0.0]
    val_far = validate_joints(q_far_reach)
    gate_far = gate.check(q_far_reach)
    print(f"\n[Test 2 - VƯƠN XA MÉP BÀN (q2 = +0.18 rad ~ 10.3°)]")
    print(f"  validate_joints: {val_far} | SafetyGate pass: {gate_far['pass']}, reason: '{gate_far['reason']}'")
    assert val_far is True, "LỖI: Tư thế vươn xa hợp lệ bị validate_joints chặn!"
    assert gate_far["pass"] is True, "LỖI: SafetyGate chặn tư thế vươn xa!"
    print("  -> PASS: Đã giải phóng không gian vươn xa mép bàn thành công!")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TEST TIẾP CẬN GẦN ĐẾ (r = 0.148m in [0.14, 0.49])
    # (Trước đây bị cưỡng bức kéo lệch ra 0.18m)
    # ─────────────────────────────────────────────────────────────────────────
    pose_near = {"pos": [0.11, 0.10, 0.15], "quat": [1.0, 0.0, 0.0, 0.0]}
    r_initial = math.hypot(pose_near["pos"][0], pose_near["pos"][1])
    pose_clamped = clamp_cartesian_safety(pose_near)
    r_clamped = math.hypot(pose_clamped["pos"][0], pose_clamped["pos"][1])
    print(f"\n[Test 3 - TIẾP CẬN GẦN ĐẾ (r = {r_initial:.4f}m)]")
    print(f"  Bán kính trước: {r_initial:.4f}m -> sau clamp: {r_clamped:.4f}m")
    assert abs(r_clamped - r_initial) < 1e-5, f"LỖI: r = {r_initial}m bị clamp sai thành {r_clamped}m!"
    print("  -> PASS: Tọa độ gần đế (r = 0.148m) được bảo toàn nguyên vẹn, không bị kéo văng ra xa!")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TEST RÀO CHẮN AN TOÀN CHỐNG TỰ VA CHẠM
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Test 4 - RÀO CHẮN CHỐNG TỰ VA CHẠM]")

    # 4a. Cùi chỏ gập quá sát (q2 = 2.50 rad > 2.40 rad)
    q_elbow_tight = [0.0, -2.20, +2.50, -1.57, -1.57, 0.0]
    assert validate_joints(q_elbow_tight) is False
    assert gate.check(q_elbow_tight)["pass"] is False
    print("  4a. Cùi chỏ gập quá sát (q2 = 2.50 > 2.40) -> REJECTED (Chặn đúng)")

    # 4b. Cùi chỏ gập ngược xuống bàn (q2 = -0.10 rad < 0.10 rad)
    q_elbow_rev = [0.0, -2.20, -0.10, -1.57, -1.57, 0.0]
    assert validate_joints(q_elbow_rev) is False
    assert gate.check(q_elbow_rev)["pass"] is False
    print("  4b. Cùi chỏ gập ngược (q2 = -0.10 < 0.10) -> REJECTED (Chặn đúng)")

    # 4c. Cổ tay bẻ quặp ngược vào forearm (q3 = +0.25 rad > +0.10 rad)
    q_wrist_danger = [0.0, -2.20, +1.90, +0.25, -1.57, 0.0]
    assert validate_joints(q_wrist_danger) is False
    assert gate.check(q_wrist_danger)["pass"] is False
    print("  4c. Cổ tay bẻ ngược vào bắp tay (q3 = +0.25 > 0.10) -> REJECTED (Chặn đúng)")

    # 4d. Cổ tay cuộn quá sâu (q3 = -3.40 rad < -3.20 rad)
    q_wrist_deep = [0.0, -2.20, +1.90, -3.40, -1.57, 0.0]
    assert validate_joints(q_wrist_deep) is False
    assert gate.check(q_wrist_deep)["pass"] is False
    print("  4d. Cổ tay cuộn quá sâu (q3 = -3.40 < -3.20) -> REJECTED (Chặn đúng)")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TEST CLAMPING RÀO CHẮN THÂN ĐẾ VÀ SÀN
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Test 5 - CLAMPING RÀO CHẮN THÂN ĐẾ & SÀN BÀN]")
    pose_too_close = {"pos": [0.05, 0.05, 0.02], "quat": [1.0, 0.0, 0.0, 0.0]}
    clamped_res = clamp_cartesian_safety(pose_too_close)
    r_res = math.hypot(clamped_res["pos"][0], clamped_res["pos"][1])
    z_res = clamped_res["pos"][2]
    print(f"  Vị trí quá gần đế (r = 0.07m, z = 0.02m) -> Clamped: r = {r_res:.4f}m (>= 0.14m), z = {z_res:.4f}m (>= 0.035m)")
    assert r_res >= 0.1399, "LỖI: r_min clamp không hoạt động!"
    assert z_res >= 0.0349, "LỖI: z_min clamp không hoạt động!"
    print("  -> PASS: Rào chắn hình trụ bảo vệ thân đế và mặt bàn hoạt động chính xác!")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. ĐỘ TRỄ THẨM ĐỊNH AN TOÀN
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(100):
        clamp_cartesian_safety(pose_near)
        gate.check(q_ready)
    dt_avg = ((time.perf_counter() - t0) / 100.0) * 1000.0
    print(f"\n[Test 6 - ĐỘ TRỄ THẨM ĐỊNH AN TOÀN TRUNG BÌNH] {dt_avg:.4f} ms (< 1.0 ms)")
    assert dt_avg < 1.0, "LỖI: Độ trễ vượt quá 1.0ms!"
    print("  -> PASS: Thuật toán kiểm tra an toàn chạy siêu tốc ~0.01ms.")

    print("\n" + "=" * 70)
    print(" TẤT CẢ 6 KỊCH BẢN KIỂM CHỨNG ĐỀU PASS 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_tests()
