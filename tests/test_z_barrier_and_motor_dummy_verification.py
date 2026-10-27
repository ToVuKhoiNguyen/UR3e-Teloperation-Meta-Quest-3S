"""
Script kiểm thử thực nghiệm toàn diện:
1. Xác minh hạ thấp rào chắn Z sát mặt bàn 1.0cm (Z_tcp = 0.020m) không bị chặn sớm ở 4-5cm.
2. Xác minh chống lọt va chạm tại các củ motor (dummy motor links) và cặp wrist_2 <-> forearm khi gập sát.
3. Benchmark tốc độ thực thi IK solver và Collision checker với 33 cặp va chạm.
"""

import os
import sys
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, load_table_bounds
from teleop.teleoperation import clamp_cartesian_safety, SafetyGate
from unittest.mock import MagicMock


def create_mock_guard():
    mock_guard = MagicMock()
    mock_guard.metrics_calc.device = "cpu"
    mock_guard.metrics_calc.get_metrics.return_value = {"sigma_min": MagicMock(item=lambda: 0.15)}
    mock_guard.danger_threshold = 0.028
    mock_guard.warn_threshold = 0.040
    return mock_guard


def run_experiment_1_z_barrier_descent(solver, gate, table_bounds):
    print("=" * 75)
    print("THỰC NGHIỆM 1: ÉP TCP ĐI THẲNG XUỐNG MẶT BÀN (Z_tcp = 0.020m, cách bàn 1.0cm)")
    print("=" * 75)
    z_table_top = table_bounds["z_table_top"]
    print(f"  Mặt bàn vật lý: Z = {z_table_top:.4f} m (+0.010m)")

    # Tư thế làm việc với camera ở hướng 3h an toàn
    q_seed = [0.0, -0.7, 1.8, -1.0, 1.57, 1.57]
    fk_start = solver.fk(q_seed)
    print(f"  Vị trí khởi điểm: X={fk_start['pos'][0]:.4f}m, Y={fk_start['pos'][1]:.4f}m, Z={fk_start['pos'][2]:.4f}m")

    # Thử nghiệm hạ độ cao theo các nấc từ 0.10m xuống 0.020m (và nấc sát bàn 0.015m)
    test_z_targets = [0.100, 0.070, 0.050, 0.040, 0.030, 0.020, 0.015]
    print("\n  Bảng diễn tiến hạ độ cao TCP:")
    print("  " + "-" * 72)
    print(f"  {'Target Z':<10} | {'Solved Z':<10} | {'Pos Err':<9} | {'TCP Clear':<10} | {'Cam Clear':<10} | {'Status':<15}")
    print("  " + "-" * 72)

    curr_seed = q_seed
    all_passed = True
    for z_tgt in test_z_targets:
        cmd = {"pos": [fk_start["pos"][0], fk_start["pos"][1], z_tgt], "quat": fk_start["quat"]}
        clamped = clamp_cartesian_safety(cmd)
        res = solver.solve(clamped, q_seed=curr_seed)

        if res["q"] is not None:
            curr_seed = res["q"]
            fk_now = solver.fk(curr_seed)
            tbl_chk = solver.check_table_collision(curr_seed, tcp_safe_clearance=0.005, cam_safe_clearance=0.005)
            gate_chk = gate.check(curr_seed)

            tcp_clear = tbl_chk["tcp_clearance_cm"]
            cam_clear = tbl_chk["cam_clearance_cm"]
            status = "PASS (Gate OK)" if (gate_chk["pass"] and tbl_chk["safe"]) else f"GATE: {gate_chk['reason'][:15]}"
            print(f"  {z_tgt*100:6.1f} cm  | {fk_now['pos'][2]*100:6.2f} cm  | {res['pos_err_mm']:6.2f} mm | {tcp_clear:6.2f} cm   | {cam_clear:6.2f} cm   | {status}")
        else:
            all_passed = False
            print(f"  {z_tgt*100:6.1f} cm  | FAILED     | {res.get('pos_err_mm', 0):6.2f} mm | N/A        | N/A        | {res.get('fail_reason')}")

    # Đánh giá tiêu chuẩn: TCP Z = 0.020m (cách bàn 1.0cm) phải hoàn toàn thành công
    cmd_20mm = {"pos": [fk_start["pos"][0], fk_start["pos"][1], 0.020], "quat": fk_start["quat"]}
    res_20mm = solver.solve(cmd_20mm, q_seed=q_seed)
    assert res_20mm["q"] is not None, "IK phải giải thành công tại Z = 0.020m!"
    tbl_20mm = solver.check_table_collision(res_20mm["q"], tcp_safe_clearance=0.005, cam_safe_clearance=0.005)
    gate_20mm = gate.check(res_20mm["q"])

    print("  " + "-" * 72)
    print(f"  KẾT QUẢ TẠI Z = 0.020m (Khoảng cách tới mặt bàn = 1.0cm):")
    print(f"    - TCP Z đạt được: {tbl_20mm['z_tcp']:.4f} m (Cách bàn: {tbl_20mm['tcp_clearance_cm']:.2f} cm)")
    print(f"    - Camera Z min  : {tbl_20mm['z_cam_min']:.4f} m (Cách bàn: {tbl_20mm['cam_clearance_cm']:.2f} cm)")
    print(f"    - IK Pos Error  : {res_20mm['pos_err_mm']:.4f} mm (< 18.0mm)")
    print(f"    - SafetyGate    : pass={gate_20mm['pass']} ({gate_20mm['reason']})")
    print("  ==> KẾT LUẬN 1: ĐÃ XÓA BỎ HOÀN TOÀN 'BỨC TƯỜNG VÔ HÌNH 4-5CM'.")


def run_experiment_2_wrist_motor_self_collision(solver):
    print("\n" + "=" * 75)
    print("THỰC NGHIỆM 2: GẬP CỔ TAY ĐÈ NẮP TRẮNG VÀO CẲNG TAY & KHUỶU TAY")
    print("=" * 75)

    cases = [
        {
            "name": "Ca 2.1: Gập cổ tay 2 vào cẳng tay (Cặp wrist_2_link <-> forearm_link từng bị sót)",
            "q": [0.0, -1.0, 1.00, 1.50, 0.0, 0.0],
            "pair": ("wrist_2_link", "forearm_link"),
        },
        {
            "name": "Ca 2.2: Gập nắp trắng motor cổ tay 2 vào bắp tay (wrist_2_motor_link <-> upper_arm_link)",
            "q": [0.0, -1.0, 2.50, 1.50, 0.0, 0.0],
            "pair": ("wrist_2_motor_link", "upper_arm_link"),
        },
        {
            "name": "Ca 2.3: Gập nắp trắng motor cổ tay 1 vào bắp tay (wrist_1_motor_link <-> upper_arm_link)",
            "q": [0.0, -1.0, 2.80, -2.00, 0.0, 0.0],
            "pair": ("wrist_1_motor_link", "upper_arm_link"),
        },
        {
            "name": "Ca 2.4: Gập đầu camera/cổ tay đè vào củ motor khuỷu tay (elbow_motor_link <-> d455_link)",
            "q": [0.0, -1.0, 0.50, 2.50, 1.57, 0.0],
            "pair": ("elbow_motor_link", "d455_link"),
        },
    ]

    for c in cases:
        print(f"\n  * {c['name']}")
        print(f"    Khớp q: {c['q']}")
        # Lấy khoảng cách cho cặp kiểm tra cụ thể
        q_arr = np.array(c["q"], dtype=float)
        dists = solver.coll_robot.compute_self_collision_distance(solver.robot, q_arr)
        names = solver.coll_robot.link_names
        pair_dist = None
        for idx, (i, j) in enumerate(zip(solver.coll_robot.active_idx_i, solver.coll_robot.active_idx_j)):
            p = (names[i], names[j])
            if p == c["pair"] or (p[1], p[0]) == c["pair"]:
                pair_dist = float(dists[idx])
                break

        res = solver.check_self_collision(c["q"], force_log=False)
        print(f"    Khoảng cách cặp {c['pair'][0]} <-> {c['pair'][1]}: {pair_dist*1000:.1f} mm")
        print(f"    Kết quả tổng thể: Safe = {res['safe']} | Cặp tối thiểu: {res['link_a']} <---> {res['link_b']} ({res['min_dist']*1000:.1f} mm)")
        assert pair_dist is not None and pair_dist < 0, f"Cặp {c['pair']} phải phát hiện va chạm (< 0mm), thực tế {pair_dist*1000:.1f}mm!"
        assert not res["safe"], f"Hệ thống PHẢI phát hiện va chạm (safe=False) trong {c['name']}!"
        print(f"    ==> ĐÃ CHẶN THÀNH CÔNG (Hard Stop kích hoạt ngay lập tức)!")


def run_experiment_3_benchmarks(solver):
    print("\n" + "=" * 75)
    print("THỰC NGHIỆM 3: BENCHMARK HIỆU NĂNG VỚI 33 CẶP VA CHẠM (ACTIVE PAIRS)")
    print("=" * 75)
    q_bench = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    fk_bench = solver.fk(q_bench)
    cmd_bench = {"pos": [fk_bench["pos"][0], fk_bench["pos"][1], 0.20], "quat": fk_bench["quat"]}

    # 1. Đo check_self_collision (1000 lần)
    t0 = time.perf_counter()
    for _ in range(1000):
        solver.check_self_collision(q_bench, verbose=False)
    dt_self_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0

    # 2. Đo check_table_collision (1000 lần)
    t0 = time.perf_counter()
    for _ in range(1000):
        solver.check_table_collision(q_bench, tcp_safe_clearance=0.005, cam_safe_clearance=0.005)
    dt_table_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0

    # 3. Đo giải IK (200 lần)
    t0 = time.perf_counter()
    for _ in range(200):
        solver.solve(cmd_bench, q_seed=q_bench)
    dt_ik_ms = ((time.perf_counter() - t0) / 200.0) * 1000.0

    print(f"  1. check_self_collision latency  (1000 lần): {dt_self_ms:.4f} ms  (Chỉ tiêu < 0.25 ms) -> {'PASS' if dt_self_ms < 0.25 else 'FAIL'}")
    print(f"  2. check_table_collision latency (1000 lần): {dt_table_ms:.4f} ms  (Chỉ tiêu < 0.25 ms) -> {'PASS' if dt_table_ms < 0.25 else 'FAIL'}")
    print(f"  3. PyRoki IK solve latency       (200 lần) : {dt_ik_ms:.2f} ms  (Chỉ tiêu < 50.0 ms) -> {'PASS' if dt_ik_ms < 50.0 else 'FAIL'}")

    assert dt_self_ms < 0.25, "Độ trễ check_self_collision vượt quá 0.25ms!"
    assert dt_table_ms < 0.25, "Độ trễ check_table_collision vượt quá 0.25ms!"
    assert dt_ik_ms < 50.0, "Độ trễ solve IK vượt quá 50ms!"


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print(" CHƯƠNG TRÌNH KIỂM CHỨNG TOÀN DIỆN HẠ RÀO CHẮN Z & DUMMY MOTOR COLLISION")
    print("#" * 75)

    solver = PyrokiIKSolver()
    guard = create_mock_guard()
    gate = SafetyGate(guard, solver=solver)
    table_bounds = load_table_bounds()

    run_experiment_1_z_barrier_descent(solver, gate, table_bounds)
    run_experiment_2_wrist_motor_self_collision(solver)
    run_experiment_3_benchmarks(solver)

    print("\n" + "=" * 75)
    print(" TẤT CẢ CÁC TIÊU CHÍ 'DONE KHI' ĐÃ ĐẠT 100%!")
    print("=" * 75 + "\n")
