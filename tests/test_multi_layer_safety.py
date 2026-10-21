"""
Unit and Mock Test for Multi-layer Safety Architecture:
1. Cartesian Safety Envelope (5cm obstacle clearance, latency < 1ms)
2. Self-collision Base Cylinder (5cm clearance)
3. Singularity Damping and Hard Stop
"""
import sys
import os
import time
import unittest
import numpy as np
import torch

WS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if WS_DIR not in sys.path:
    sys.path.insert(0, WS_DIR)

from teleop.teleoperation import clamp_cartesian_safety, SafetyGate
from unittest.mock import MagicMock

class TestMultiLayerSafety(unittest.TestCase):
    def test_cartesian_safety_clamp_table_plunge(self):
        # Mock Test 1: Vector plunging into table Z = -0.10m
        test_pose = {"pos": [0.25, 0.15, -0.10], "quat": [1.0, 0.0, 0.0, 0.0]}
        
        t0 = time.perf_counter()
        safe_pose = clamp_cartesian_safety(test_pose, z_min=0.05, z_max=0.55, r_min=0.12, r_max=0.485)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        print(f"[Mock Test 1] Plunge into table: {test_pose['pos']} -> Clamped: {safe_pose['pos']} (Latency: {dt_ms:.4f}ms)")
        self.assertAlmostEqual(safe_pose["pos"][2], 0.05, places=4)
        self.assertAlmostEqual(safe_pose["pos"][0], 0.25, places=4)
        self.assertAlmostEqual(safe_pose["pos"][1], 0.15, places=4)
        self.assertLess(dt_ms, 1.0) # Latency must be < 1ms

    def test_cartesian_safety_clamp_base_self_collision(self):
        # Mock Test 2: TCP moving too close to Base cylinder at low height (Z = 0.15m)
        # R_base = 0.07m, clearance = 0.05m -> R_min = 0.12m
        test_pose = {"pos": [0.05, 0.05, 0.15], "quat": [1.0, 0.0, 0.0, 0.0]} # r_xy = 0.0707m
        
        t0 = time.perf_counter()
        safe_pose = clamp_cartesian_safety(test_pose, z_min=0.05, z_max=0.55, r_min=0.12, r_max=0.485)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        r_out = float(np.hypot(safe_pose["pos"][0], safe_pose["pos"][1]))
        print(f"[Mock Test 2] Base self-collision: r_xy={np.hypot(0.05, 0.05):.4f}m -> Clamped r_xy: {r_out:.4f}m (Latency: {dt_ms:.4f}ms)")
        self.assertAlmostEqual(r_out, 0.12, places=3)
        self.assertLess(dt_ms, 1.0)

    def test_cartesian_safety_clamp_over_reach(self):
        # Mock Test 3: TCP commanding reach beyond 500mm (0.60m)
        test_pose = {"pos": [0.45, 0.45, 0.25], "quat": [1.0, 0.0, 0.0, 0.0]} # r_xy = 0.636m
        
        t0 = time.perf_counter()
        safe_pose = clamp_cartesian_safety(test_pose, z_min=0.05, z_max=0.55, r_min=0.12, r_max=0.485)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        r_out = float(np.hypot(safe_pose["pos"][0], safe_pose["pos"][1]))
        print(f"[Mock Test 3] Over-reach: r_xy={np.hypot(0.45, 0.45):.4f}m -> Clamped r_xy: {r_out:.4f}m (Latency: {dt_ms:.4f}ms)")
        self.assertAlmostEqual(r_out, 0.485, places=3)
        self.assertLess(dt_ms, 1.0)

    def test_singularity_damping_and_stop(self):
        # Mock Test 4: Singularity Guard metrics
        guard_mock = MagicMock()
        guard_mock.danger_threshold = 0.035
        guard_mock.warn_threshold = 0.060
        guard_mock.metrics_calc.device = torch.device("cpu")
        
        gate = SafetyGate(guard=guard_mock)

        # Case A: Safe zone (sigma = 0.08 >= warn_threshold 0.06)
        guard_mock.metrics_calc.get_metrics.return_value = {"sigma_min": torch.tensor(0.080)}
        res_safe = gate.check([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertTrue(res_safe["pass"])
        self.assertEqual(res_safe["damping"], 1.0)

        # Case B: Warning zone (sigma = 0.045 between 0.035 and 0.060)
        guard_mock.metrics_calc.get_metrics.return_value = {"sigma_min": torch.tensor(0.045)}
        res_warn = gate.check([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertTrue(res_warn["pass"])
        self.assertGreater(res_warn["damping"], 0.2)
        self.assertLess(res_warn["damping"], 1.0)
        print(f"[Mock Test 4] Singularity warning zone sigma=0.045 -> Smooth Damping factor: {res_warn['damping']:.3f}")

        # Case C: Danger zone (sigma = 0.020 < danger_threshold 0.035)
        guard_mock.metrics_calc.get_metrics.return_value = {"sigma_min": torch.tensor(0.020)}
        res_danger = gate.check([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertFalse(res_danger["pass"])
        self.assertIn("singularity", res_danger["reason"])

if __name__ == "__main__":
    unittest.main()
