import os
import sys
import time
import math
import numpy as np

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "teleop"))
sys.path.insert(0, os.path.join(workspace_root, "teleop_quest"))

from pyroki_solver import PyrokiIKSolver
from teleoperation import TeleoperationController, SafetyGate

class MockPoseSource:
    def __init__(self, initial_pose):
        self.pose = dict(initial_pose)

    def get_pose(self):
        return dict(self.pose)

class MockExecutor:
    def __init__(self):
        self.sent_commands = []
        self.stop_calls = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def send(self, q):
        self.sent_commands.append(list(q))

    def stop(self, a=10.0):
        self.stop_calls.append(a)

def run_benchmarks():
    print("================================================================================")
    print("       BẮT ĐẦU KIỂM CHỨNG TOÀN DIỆN ELASTIC TRACKING & SMOOTH CONTROL")
    print("================================================================================")

    solver = PyrokiIKSolver(scene_path="assets/obstacle_scene.yml")
    from safety.singularity_guard.guard_core import SingularityGuard
    from teleop_quest.bootstrap import load_singularity_config

    config = load_singularity_config(workspace_root)
    guard = SingularityGuard(config)
    gate = SafetyGate(guard=guard, solver=solver)
    executor = MockExecutor()

    # Khởi tạo tư thế Home an toàn
    q_home = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]
    fk_home = solver.fk(q_home)
    initial_pose = {"pos": list(fk_home["pos"]), "quat": list(fk_home["quat"])}

    pose_source = MockPoseSource(initial_pose)
    teleop = TeleoperationController(pose_source, solver, gate, executor, dt=0.05)
    teleop.enabled = True
    teleop._last_valid_q = list(q_home)

    # ─────────────────────────────────────────────────────────────────────────
    # BENCHMARK 1: FAST DRAG TEST (100 frames nhảy đột ngột 10cm - 25cm)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- [BENCHMARK 1] FAST DRAG TEST (100 FRAMES NHẢY ĐỘT NGỘT 10-25CM) ---")
    executor.sent_commands.clear()
    executor.stop_calls.clear()

    q_current = list(q_home)
    blocked_pos_err_count = 0
    total_frames = 100

    # Di chuyển mục tiêu zic-zac đột ngột xa 10cm - 25cm
    np.random.seed(42)
    for i in range(total_frames):
        # Đột ngột nhảy vị trí
        offset = np.random.uniform(-0.15, 0.15, size=3)
        # Giữ Z trong khoảng làm việc
        target_p = np.array(fk_home["pos"]) + offset
        target_p[2] = np.clip(target_p[2], 0.05, 0.45)
        pose_source.pose["pos"] = target_p.tolist()

        teleop.step(q_current)
        st = teleop.get_status()
        if "IK_POS_ERROR" in st or "18mm" in st:
            blocked_pos_err_count += 1

        if executor.sent_commands:
            q_current = list(executor.sent_commands[-1])

    print(f"Số frame thử nghiệm: {total_frames}")
    print(f"Số lần bị chặn do IK_POS_ERROR: {blocked_pos_err_count} / {total_frames} (0%)")
    print(f"Số lần gọi executor.stop(): {len(executor.stop_calls)} (0 lần)")
    print(f"Số lệnh servoJ gửi thành công: {len(executor.sent_commands)} / {total_frames}")
    
    assert blocked_pos_err_count == 0, f"LỖI: Vẫn còn {blocked_pos_err_count} frame bị chặn IK_POS_ERROR!"
    assert len(executor.stop_calls) == 0, f"LỖI: executor.stop() bị gọi {len(executor.stop_calls)} lần!"
    assert len(executor.sent_commands) == total_frames, "Thiếu lệnh gửi servoJ!"
    print(">>> BENCHMARK 1 ĐẠT 100%: 0% IK_POS_ERROR, 0 LẦN PHANH CỨNG! <<<")

    # ─────────────────────────────────────────────────────────────────────────
    # BENCHMARK 2: OUT-OF-REACH TEST (Kéo xa R = 0.65m vượt tầm với UR3e)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- [BENCHMARK 2] OUT-OF-REACH TEST (R = 0.65m VƯỢT TẦM VỚI UR3E) ---")
    executor.sent_commands.clear()
    executor.stop_calls.clear()

    # Đặt target ở R = 0.65m
    pose_source.pose["pos"] = [0.45, 0.45, 0.20] # R = sqrt(0.45^2 + 0.45^2) = 0.636m > 0.52m
    q_current = list(q_home)

    out_of_reach_frames = 40
    for _ in range(out_of_reach_frames):
        teleop.step(q_current)
        if executor.sent_commands:
            q_current = list(executor.sent_commands[-1])

    # Kiểm tra bán kính đạt được của robot
    fk_end = solver.fk(q_current)
    r_end = np.hypot(fk_end["pos"][0], fk_end["pos"][1])
    print(f"Bán kính mục tiêu yêu cầu: 0.636m")
    print(f"Bán kính robot vươn tới tối đa an toàn: {r_end:.3f}m (<= 0.52m)")
    print(f"Số lần gọi executor.stop(): {len(executor.stop_calls)} (0 lần)")
    print(f"Trạng thái teleop: {teleop.get_status()}")

    assert len(executor.stop_calls) == 0, "Out-of-reach làm kích hoạt phanh cứng!"
    assert r_end <= 0.525, f"Bán kính {r_end} vượt ngoài biên an toàn!"
    print(">>> BENCHMARK 2 ĐẠT 100%: VƯƠN HẾT TẦM ÊM ÁI, HOÀN TOÀN KHÔNG GIẬT CỤC! <<<")

    # ─────────────────────────────────────────────────────────────────────────
    # BENCHMARK 3: BARRIER SLIDING TEST (Ép sâu xuống mặt bàn Z <= 0.0m)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- [BENCHMARK 3] BARRIER SLIDING TEST (ÉP SÂU XUỐNG DƯỚI MẶT BÀN Z <= 0.0m) ---")
    executor.sent_commands.clear()
    executor.stop_calls.clear()

    # Kéo mục tiêu lướt ngang ở Z = -0.05m (âm sâu dưới mặt bàn)
    q_current = list(q_home)
    slide_frames = 40
    for step_i in range(slide_frames):
        x = 0.20 + 0.15 * (step_i / slide_frames)
        pose_source.pose["pos"] = [x, 0.10, -0.05] # Z âm dưới bàn!
        teleop.step(q_current)
        if executor.sent_commands:
            q_current = list(executor.sent_commands[-1])

    fk_slide = solver.fk(q_current)
    z_final = fk_slide["pos"][2]
    tbl_chk = solver.check_table_collision(q_current, tcp_safe_clearance=0.0, cam_safe_clearance=0.0)

    print(f"Độ cao Z mục tiêu đưa vào: -0.050m (dưới mặt bàn)")
    print(f"Độ cao Z thực tế của robot: {z_final:.3f}m (trên mặt bàn giáp đỏ Z=+0.020m)")
    print(f"Kiểm tra an toàn va chạm mặt bàn: safe = {tbl_chk['safe']}")
    print(f"Số lần gọi executor.stop(): {len(executor.stop_calls)} (0 lần)")

    assert tbl_chk["safe"] == True, "Robot đâm thủng mặt bàn!"
    assert z_final >= 0.018, f"TCP bị lún dưới rào chắn: Z={z_final}"
    assert len(executor.stop_calls) == 0, "Bị dập phanh cứng khi trượt mặt bàn!"
    print(">>> BENCHMARK 3 ĐẠT 100%: TRƯỢT MẶT BÀN HOÀN TOÀN MƯỢT MÀ, KHÔNG KHỰNG! <<<")

    print("\n================================================================================")
    print("             TẤT CẢ 3 BÀI BENCHMARK ĐÃ VƯỢT QUA XUẤT SẮC (100% PASS)")
    print("================================================================================")

if __name__ == "__main__":
    run_benchmarks()
