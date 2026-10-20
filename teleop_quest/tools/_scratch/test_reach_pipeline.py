#!/usr/bin/env python3
"""
Tái hiện và kiểm chứng chuyển động 3 trục (Tiến, Phải, Cao) qua đường ống thật:
WidgetPoseSource -> TeleoperationController -> PyRoki IK -> SafetyGate -> ServoJExecutor (URSim)
"""

import os
import sys
import time
import math
import numpy as np
import torch
import rtde_receive

# Đảm bảo đường dẫn import
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teleop_quest.pose_sources import WidgetPoseSource
from teleop.teleoperation import (
    PyrokiIKSolverWrapper, SafetyGate, ServoJExecutor, TeleoperationController
)
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config
from reset_home import unlock_dashboard, send_movej_script, SAFE_HOME_JOINTS

# Giới hạn khớp chuẩn UR3e (từ assets/ur3e_custom.urdf và master)
JOINT_LIMITS = [
    (-6.28, 6.28),      # q0: shoulder_pan
    (-6.28, 6.28),      # q1: shoulder_lift
    (-3.14, 3.14),      # q2: elbow
    (-2.00, 2.00),      # q3: wrist_1
    (-2.00, 2.00),      # q4: wrist_2
    (-12.56, 12.56),    # q5: wrist_3
]

class MockWidget:
    def __init__(self, position, wxyz):
        self.position = list(position)
        self.wxyz = list(wxyz)

def reset_to_home(robot_ip="127.0.0.1"):
    unlock_dashboard(robot_ip)
    send_movej_script(robot_ip, SAFE_HOME_JOINTS, accel=0.8, speed=0.5)
    rc = rtde_receive.RTDEReceiveInterface(robot_ip)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        q = rc.getActualQ()
        if max(abs(q[i] - SAFE_HOME_JOINTS[i]) for i in range(6)) < 0.008:
            break
        time.sleep(0.1)
    rc.disconnect()
    time.sleep(0.5)

def run_axis_test(axis_name, direction_vec, robot_ip="127.0.0.1"):
    print(f"\n========================================================")
    print(f" BAT DAU TEST TRUC: {axis_name} (Huong: {direction_vec})")
    print(f"========================================================")
    reset_to_home(robot_ip)

    rc = rtde_receive.RTDEReceiveInterface(robot_ip)
    q_start = rc.getActualQ()
    
    config = load_singularity_config(REPO_ROOT)
    guard = SingularityGuard(config)
    gate = SafetyGate(guard)
    solver = PyrokiIKSolverWrapper()
    executor = ServoJExecutor(dt=0.05, lookahead_time=0.12, gain=250, robot_ip=robot_ip)

    fk_home = solver.fk(q_start)
    home_pos = np.array(fk_home["pos"])
    home_quat = fk_home["quat"]
    print(f"[Home] Pos: {np.round(home_pos, 4)}, Quat: {np.round(home_quat, 4)}")

    widget = MockWidget(position=home_pos.copy(), wxyz=list(home_quat))
    pose_source = WidgetPoseSource(widget, alpha=0.55)
    
    teleop = TeleoperationController(
        pose_source=pose_source,
        pyroki_solver=solver,
        safety_gate=gate,
        executor=executor,
        dt=0.05
    )
    teleop.enabled = True
    teleop._last_valid_q = list(q_start)

    results = []
    blocked_at = None

    for step in range(1, 16):
        target_cm = step * 1.0  # cm
        target_pos = home_pos + direction_vec * (target_cm * 0.01)
        widget.position = list(target_pos)

        # Cho pipeline chay 10 ticks (0.5 giay) de EMA va rate limiter cap nhat den muc tieu
        ik_err_mm = 0.0
        final_status = ""
        for tick in range(10):
            q_real = rc.getActualQ()
            vr_pose = pose_source.get_pose()
            teleop.step(q_real, pose=vr_pose)
            final_status = teleop.get_status()
            time.sleep(0.05)

        # Doc trang thai robot that sau khi ServoJ thuc thi
        q_curr = rc.getActualQ()
        fk_curr = solver.fk(q_curr)
        curr_pos = np.array(fk_curr["pos"])
        actual_dist_cm = np.linalg.norm(curr_pos - home_pos) * 100.0

        # Tinh min margin khop
        margins = [min(q_curr[i] - JOINT_LIMITS[i][0], JOINT_LIMITS[i][1] - q_curr[i]) for i in range(6)]
        min_idx = int(np.argmin(margins))
        min_margin = float(margins[min_idx])

        # Lay sai so IK tu solver truc tiep
        res_ik = solver.solve({"pos": list(target_pos), "quat": list(home_quat)}, q_curr)
        ik_err_mm = res_ik.get("pos_err_mm", 0.0)

        # Kiem tra xem buoc nay co bi chan khong
        is_blocked = ("vật cản" in final_status.lower() or 
                      "vat can" in final_status.lower() or
                      "dừng" in final_status.lower() or 
                      "dung" in final_status.lower() or
                      "thất bại" in final_status.lower() or
                      res_ik.get("q") is None)

        blocker_info = final_status
        if res_ik.get("q") is None:
            blocker_info = f"IK None ({final_status.split(' - ')[0]})"

        step_res = {
            "step": step,
            "cmd_cm": target_cm,
            "actual_cm": actual_dist_cm,
            "ik_err_mm": ik_err_mm,
            "min_margin_joint": f"q{min_idx}",
            "min_margin_rad": min_margin,
            "status": final_status,
            "blocker": blocker_info,
            "is_blocked": is_blocked
        }
        results.append(step_res)

        if is_blocked and blocked_at is None:
            blocked_at = step

    teleop.enabled = False
    rc.disconnect()

    return results, blocked_at

def print_markdown_table(axis_name, results, blocked_at):
    print(f"\n### Kết quả kiểm thử: Trục {axis_name}")
    if blocked_at is not None:
        print(f"**Bị chặn tại Bước {blocked_at} ({blocked_at} cm)**\n")
    else:
        print(f"**Hoàn thành trọn vẹn 15/15 bước (15 cm)**\n")

    print("| Bước | Lệnh (cm) | TCP thực tế (cm) | Sai số IK (mm) | Min Margin khớp | Trạng thái / Nơi chặn |")
    print("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for r in results:
        status_flag = "❌ " if r["is_blocked"] else "✅ "
        print(f"| {r['step']:2d} | {r['cmd_cm']:4.1f} | {r['actual_cm']:5.2f} | {r['ik_err_mm']:6.2f} | {r['min_margin_joint']} ({r['min_margin_rad']:.3f} rad) | {status_flag}{r['blocker']} |")

if __name__ == "__main__":
    axes = [
        ("Tiến (+X)", np.array([1.0, 0.0, 0.0])),
        ("Phải (-Y)", np.array([0.0, -1.0, 0.0])),
        ("Cao (+Z)", np.array([0.0, 0.0, 1.0])),
    ]

    all_res = {}
    for name, direction in axes:
        res, blocked = run_axis_test(name, direction)
        all_res[name] = (res, blocked)
        print_markdown_table(name, res, blocked)

    reset_to_home()
