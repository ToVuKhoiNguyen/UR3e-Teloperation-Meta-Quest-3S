"""
tests/test_click_go_home_button_integration.py
Direct integration test verifying:
1. No TypeError when triggering Go Home while cb_teleop.value = True (testing exact callback flow).
2. Robot reaches SAFE_HOME_JOINTS (< 0.01 rad error).
3. Viser 3D Gizmo jumps to Home FK position.
4. cb_teleop.value is restored to True and subsequent Teleop drag moves the robot immediately.
"""

import os
import sys
import time
import threading
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from teleop.teleoperation import TeleoperationController, clamp_cartesian_safety, SafetyGate
from teleop.pyroki_solver import PyrokiIKSolver
from safety.singularity_guard.guard_core import SingularityGuard
from teleop_quest.bootstrap import load_singularity_config
from teleop_quest.home_planner import HomePlanner


class MockTargetWidget:
    def __init__(self, pos, quat):
        self.position = tuple(pos)
        self.wxyz = tuple(quat)


class MockCheckboxWithCallback:
    """Simulates Viser UI Checkbox with on_update callback invocation."""
    def __init__(self, initial_val=True):
        self._val = initial_val
        self._callbacks = []

    def on_update(self, cb):
        self._callbacks.append(cb)
        return cb

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, new_val):
        changed = (self._val != new_val)
        self._val = new_val
        if changed:
            # Trigger registered on_update callbacks synchronously (just like Viser)
            for cb in list(self._callbacks):
                cb(self)


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
    def __init__(self, initial_joints):
        self._q = list(initial_joints)

    def joints(self):
        return list(self._q)

    def tcp(self):
        return [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]

    def set_joints(self, q):
        self._q = list(q)


class MockRTDEServoJExecutor:
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
            self.robot_ctrl.set_joints(q)

    def stop(self, a=10.0):
        pass


def test_click_go_home_button_exact_callback_flow():
    print("\n" + "=" * 80)
    print(" TEST: GO HOME BUTTON CLICK WITH SYNCHRONOUS CB_TELEOP CALLBACK & STOP(10.0)")
    print("=" * 80)

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

    # Viser Checkbox with real callback attached (calling teleop.stop(10.0))
    cb_teleop = MockCheckboxWithCallback(initial_val=True)

    controller = TeleoperationController(
        pose_source=widget_pose_source,
        solver=solver,
        executor=executor,
        gate=gate,
        dt=0.05
    )
    controller.enabled = True
    controller.sync_with_robot(SAFE_HOME_JOINTS)

    # Register exactly the callback that crashed before:
    def _on_teleop_update(_):
        is_enabled = bool(cb_teleop.value)
        controller.enabled = is_enabled
        if is_enabled:
            widget_pose_source.sync_with_robot()
            executor.connect()
        else:
            # THIS IS THE EXACT CALL THAT THREW: TypeError: TeleoperationController.stop() takes 1 positional argument but 2 were given
            executor.stop(10.0)
            controller.stop(10.0)

    cb_teleop.on_update(_on_teleop_update)

    # Drag robot away from Home via Teleop
    p_drag = np.array(fk_home["pos"]) + np.array([0.08, 0.15, 0.02])
    target_widget.position = tuple(p_drag)
    for _ in range(10):
        controller.step(q_real=robot_ctrl.joints(), pose=widget_pose_source.get_pose())

    q_dragged = robot_ctrl.joints()
    dist_away = max(abs(q_dragged[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
    print(f"[Initial Teleop] Robot dragged away from Home: deviation = {dist_away:.4f} rad")
    assert dist_away > 0.15, "Robot was not dragged away from home!"

    # Simulate User Clicking Go Home button:
    # This calls trigger_go_home which sets cb_teleop.value = False
    is_navigating_home = threading.Event()
    home_planner = HomePlanner(default_home_q=SAFE_HOME_JOINTS)

    print("\n[Simulating Button Click] Invoking trigger_go_home flow...")
    type_error_occurred = False

    try:
        # Trigger Go Home flow:
        is_navigating_home.set()
        # Setting cb_teleop.value = False triggers _on_teleop_update -> controller.stop(10.0)
        cb_teleop.value = False

        # Run go_home synchronously or in thread (here synchronous for direct assertion)
        res = home_planner.go_home(
            teleop=controller,
            executor=executor,
            ctrl=robot_ctrl,
            target_widget=target_widget,
            widget_pose_source=widget_pose_source,
            ui_status=ui_status,
            cb_teleop=cb_teleop,
            is_navigating_home=is_navigating_home,
            poll_sleep=0.01
        )
    except TypeError as te:
        print(f"FAILED: Caught TypeError during Go Home: {te}")
        type_error_occurred = True
        raise te

    print(f"[TypeError Verification] No TypeError occurred: {not type_error_occurred}")
    assert not type_error_occurred, "TypeError still occurred during teleop.stop(10.0)!"

    # 1. Verify Robot is at Home (< 0.01 rad)
    q_final = robot_ctrl.joints()
    err_home = max(abs(q_final[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
    print(f"[Home Position Verification] Final error from SAFE_HOME_JOINTS: {err_home:.5f} rad (< 0.01 rad)")
    assert err_home < 0.01, f"Robot did not reach Home: error = {err_home:.4f} rad"

    # 2. Verify Gizmo position jumped to Home FK position
    fk_actual = solver.fk(q_final)
    gizmo_pos_err = np.linalg.norm(np.array(target_widget.position) - np.array(fk_actual["pos"]))
    print(f"[Gizmo Sync Verification] Gizmo position distance to Home TCP: {gizmo_pos_err*1000:.2f} mm")
    assert gizmo_pos_err < 0.005, f"Gizmo was not synced to Home TCP position (err={gizmo_pos_err*1000:.1f}mm)!"

    # 3. Verify cb_teleop is True and not locked
    print(f"[Teleop State Verification] cb_teleop.value={cb_teleop.value}, teleop.enabled={controller.enabled}, is_navigating_home={is_navigating_home.is_set()}")
    assert cb_teleop.value is True, "cb_teleop was not restored to True!"
    assert controller.enabled is True, "teleop.enabled was not restored to True!"
    assert not is_navigating_home.is_set(), "is_navigating_home is still locked!"

    # 4. Verify Immediate Teleop Motion Response (dragging target after Home)
    p_drag_after = np.array(fk_home["pos"]) + np.array([-0.05, 0.08, 0.0])
    target_widget.position = tuple(p_drag_after)

    cmd_count_before = len(executor.sent_commands)
    for _ in range(6):
        controller.step(q_real=robot_ctrl.joints(), pose=widget_pose_source.get_pose())
    cmd_count_after = len(executor.sent_commands)

    frames_sent = cmd_count_after - cmd_count_before
    print(f"[Resume Teleop Verification] Motion frames sent after Home: {frames_sent}/6")
    assert frames_sent == 6, f"Teleop failed to respond immediately after Home (sent {frames_sent}/6 frames)!"

    print("\n>>> GO HOME BUTTON CLICK TEST PASSED: 0 TypeError, 100% Home Accuracy, Instant Teleop Resume! <<<\n")


if __name__ == "__main__":
    test_click_go_home_button_exact_callback_flow()
    print("=" * 80)
    print(" ALL GO HOME INTEGRATION TESTS PASSED (100% SUCCESS) ")
    print("=" * 80)
