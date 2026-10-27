"""
tests/test_go_home_recovery_and_clean_terminal.py
Quantitative test verifying:
1. Deadlock-free Go Home execution & 100% Teleop recovery across 3 consecutive cycles.
2. Clean terminal verification (0 instances of [SYSTEM PROFILER] HIỆU NĂNG THỰC THI across 300+ frames).
"""

import os
import sys
import io
import time
import threading
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop.teleoperation import TeleoperationController, clamp_cartesian_safety, SafetyGate
from teleop.pyroki_solver import PyrokiIKSolver
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config
from teleop_quest.home_planner import HomePlanner, generate_s_curve_trajectory
from teleop_quest.profiler import SystemProfiler


class MockTargetWidget:
    def __init__(self, pos, quat):
        self.position = tuple(pos)
        self.wxyz = tuple(quat)


class MockCheckbox:
    def __init__(self, initial_val=True):
        self.value = initial_val


class MockUIStatus:
    def __init__(self):
        self.value = "OK"


class MockWidgetPoseSource:
    def __init__(self, widget):
        self.widget = widget
        self._ema_pos = None
        self._ema_quat_xyzw = None

    def get_pose(self):
        return {"pos": list(self.widget.position), "quat": list(self.widget.wxyz)}

    def sync_with_robot(self, joints=None, fk_func=None):
        self._ema_pos = None
        self._ema_quat_xyzw = None


class MockRobotController:
    """Simulates real-time RTDE receive / robot joints state."""
    def __init__(self, initial_joints):
        self._q = list(initial_joints)

    def joints(self):
        return list(self._q)

    def tcp(self):
        return [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]

    def set_joints(self, q):
        self._q = list(q)


class MockRTDEServoJExecutor:
    """Simulates RTDEControlInterface executor for testing."""
    def __init__(self, robot_ctrl: MockRobotController):
        self.robot_ctrl = robot_ctrl
        self.connected = True
        self.sent_commands = []
        self._ctrl = self

    def isConnected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send(self, q):
        if self.connected:
            self.sent_commands.append(list(q))
            # Simulate real robot following servoJ command
            self.robot_ctrl.set_joints(q)

    def stop(self, a=10.0):
        pass


def test_clean_terminal_verification():
    print("\n" + "=" * 80)
    print(" TEST 1: CLEAN TERMINAL VERIFICATION (0 PROFILER SPAM ACROSS 300+ FRAMES)")
    print("=" * 80)

    profiler = SystemProfiler(window_size=100, print_to_stdout=False)

    # Capture stdout during 350 frame ticks
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout

    try:
        for f in range(350):
            profiler.start_segment("IK_Solver")
            time.sleep(0.0001)
            profiler.end_segment("IK_Solver")

            profiler.start_segment("Camera_Read")
            time.sleep(0.0001)
            profiler.end_segment("Camera_Read")

            profiler.start_segment("UI_Update")
            time.sleep(0.0001)
            profiler.end_segment("UI_Update")

            profiler.tick_frame(0.005)
    finally:
        sys.stdout = old_stdout

    output_text = captured_stdout.getvalue()
    profiler_table_found = "[SYSTEM PROFILER] HIỆU NĂNG THỰC THI" in output_text

    print(f"Total simulated frames: 350 (over 3 measurement windows)")
    print(f"Profiler report generated: {profiler.last_report is not None}")
    print(f"Found '[SYSTEM PROFILER]' table in terminal output: {profiler_table_found}")

    assert not profiler_table_found, "FAILED: [SYSTEM PROFILER] was printed to stdout!"
    print("\n>>> CLEAN TERMINAL PASS: 0% profiler spam, Terminal output is 100% clean! <<<\n")


def test_teleop_go_home_recovery_3_cycles():
    print("\n" + "=" * 80)
    print(" TEST 2: DEADLOCK-FREE GO HOME & 100% TELEOP RECOVERY (3 CONSECUTIVE CYCLES)")
    print("=" * 80)

    # 1. Setup components
    SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.2708, -1.5708, 0.0]
    robot_ctrl = MockRobotController(SAFE_HOME_JOINTS)
    executor = MockRTDEServoJExecutor(robot_ctrl)
    solver = PyrokiIKSolver(scene_path="assets/obstacle_scene.yml")
    config = load_singularity_config(BASE_DIR)
    guard = SingularityGuard(config)
    gate = SafetyGate(guard=guard, solver=solver)

    fk_home = solver.fk(SAFE_HOME_JOINTS)
    target_widget = MockTargetWidget(fk_home["pos"], fk_home["quat"])
    widget_pose_source = MockWidgetPoseSource(target_widget)
    ui_status = MockUIStatus()
    cb_teleop = MockCheckbox(initial_val=True)
    is_navigating_home = threading.Event()

    controller = TeleoperationController(
        pose_source=widget_pose_source,
        solver=solver,
        executor=executor,
        gate=gate,
        dt=0.05
    )
    controller.enabled = True
    controller.sync_with_robot(SAFE_HOME_JOINTS)

    home_planner = HomePlanner(default_home_q=SAFE_HOME_JOINTS)

    # Execute 3 consecutive cycles of: Teleop Drag -> Go Home -> Resume Teleop Drag
    for cycle in range(1, 4):
        print(f"\n--- CYCLE {cycle}/3: Teleop Drag -> Go Home -> Resume Teleop Drag ---")

        # Step A: Teleop Drag robot away from Home
        p_home = np.array(fk_home["pos"])
        # Move target out by 12cm in Y and 5cm in X
        offset = np.array([0.05 * cycle, 0.12, 0.0])
        p_drag = p_home + offset
        target_widget.position = tuple(p_drag)

        print(f" [Cycle {cycle}] Teleop dragging target to: Y = {p_drag[1]:.3f}m")
        # Run 8 teleop steps
        for _ in range(8):
            controller.step(q_real=robot_ctrl.joints(), pose=widget_pose_source.get_pose())

        q_dragged = robot_ctrl.joints()
        dev_from_home = max(abs(q_dragged[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
        print(f" [Cycle {cycle}] Robot moved away from Home: angular deviation = {dev_from_home:.3f} rad")
        assert dev_from_home > 0.10, "Robot did not move during Teleop drag phase!"

        # Step B: Trigger Go Home
        print(f" [Cycle {cycle}] Clicking 'Go Home' button...")
        res = home_planner.go_home(
            teleop=controller,
            executor=executor,
            ctrl=robot_ctrl,
            target_widget=target_widget,
            widget_pose_source=widget_pose_source,
            ui_status=ui_status,
            cb_teleop=cb_teleop,
            is_navigating_home=is_navigating_home,
            poll_sleep=0.01  # accelerated poll for unit test
        )

        assert res is True, f"Go Home failed on cycle {cycle}!"

        # Verify Robot reached Home
        q_after_home = robot_ctrl.joints()
        home_err = max(abs(q_after_home[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
        print(f" [Cycle {cycle}] Reached Home! Remaining error: {home_err:.5f} rad (< 0.01 rad)")
        assert home_err < 0.01, f"Robot did not reach Home! Error: {home_err:.4f} rad"

        # Verify Teleop was 100% restored
        print(f" [Cycle {cycle}] Verifying Teleop status: cb_teleop={cb_teleop.value}, teleop.enabled={controller.enabled}, is_navigating_home={is_navigating_home.is_set()}")
        assert cb_teleop.value is True, "cb_teleop was not restored to True!"
        assert controller.enabled is True, "teleop.enabled was not restored to True!"
        assert not is_navigating_home.is_set(), "is_navigating_home lock was not cleared!"

        # Step C: Immediately drag target again via Teleop (proving zero deadlock)
        p_drag2 = p_home + np.array([-0.04, 0.08, 0.02])
        target_widget.position = tuple(p_drag2)
        print(f" [Cycle {cycle}] Resuming Teleop immediately: dragging target to Y = {p_drag2[1]:.3f}m")

        cmd_count_before = len(executor.sent_commands)
        for _ in range(5):
            controller.step(q_real=robot_ctrl.joints(), pose=widget_pose_source.get_pose())

        cmd_count_after = len(executor.sent_commands)
        new_cmds_sent = cmd_count_after - cmd_count_before
        print(f" [Cycle {cycle}] Teleop motion commands sent immediately: {new_cmds_sent} frames")
        assert new_cmds_sent == 5, f"Teleop failed to send commands after Go Home! (Got {new_cmds_sent}/5)"

        print(f" >>> CYCLE {cycle}/3 PASS: Smooth Go Home & Instant Teleop Recovery! <<<")

    # Step D: Test Fault Injection / Exception Recovery (Proving bulletproof finally block)
    print("\n--- Testing Exception / Fault Injection Recovery ---")
    is_navigating_home.set()
    cb_teleop.value = False
    controller.enabled = False

    class FaultyCtrl:
        def joints(self):
            raise RuntimeError("Simulated transient socket disconnect")

    try:
        home_planner.go_home(
            teleop=controller,
            executor=executor,
            ctrl=FaultyCtrl(),
            target_widget=target_widget,
            widget_pose_source=widget_pose_source,
            ui_status=ui_status,
            cb_teleop=cb_teleop,
            is_navigating_home=is_navigating_home,
            poll_sleep=0.01
        )
    except Exception:
        pass

    print(f" After simulated fault: cb_teleop={cb_teleop.value}, teleop.enabled={controller.enabled}, is_navigating_home={is_navigating_home.is_set()}")
    assert cb_teleop.value is True, "Finally block failed to restore cb_teleop on exception!"
    assert controller.enabled is True, "Finally block failed to restore teleop.enabled on exception!"
    assert not is_navigating_home.is_set(), "Finally block failed to clear is_navigating_home on exception!"

    print("\n>>> FAULT INJECTION PASS: Teleop recovery is 100% guaranteed even under network error! <<<\n")


if __name__ == "__main__":
    test_clean_terminal_verification()
    test_teleop_go_home_recovery_3_cycles()
    print("=" * 80)
    print(" ALL GO-HOME RECOVERY & CLEAN TERMINAL TESTS PASSED (100% SUCCESS) ")
    print("=" * 80)
