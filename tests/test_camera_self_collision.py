"""
Unit Test: Camera & Mount Self-Collision Torture Test Suite.
Kiểm chứng:
1. Hình học va chạm ôm sát biên dạng thật + 5mm giáp đỏ:
   - D455 cylinder length=0.134m, radius=0.0265m (bao phủ 100% thể tích và 4 góc mép của lưới đỏ 5mm).
   - Kích hoạt đủ toàn bộ 34 cặp tự va chạm (camera_mount, d455, tcp vs toàn bộ các khâu robot).
2. Torture Scenario 1: Gập cổ tay (wrist_1_joint q3) ép thẳng cụm Camera D455 quay ngược vào ống forearm_link.
3. Torture Scenario 2: Xoay cổ tay ép giá đỡ camera_mount_link sát forearm_link và wrist_1_link / motor.
4. Torture Scenario 3: Gập khuỷu tay (elbow_joint) đưa cụm Camera ép sát vào upper_arm_link.
5. Torture Scenario 4: Gập cánh tay đưa cụm Camera ép sát vào shoulder_link / base_link_inertia.
6. Torture Scenario 5: Xoay cổ tay đưa D455 quét qua củ motor wrist_1_motor_link và wrist_2_motor_link.
7. Chỉ tiêu: Trong 100% kịch bản, dừng/trượt ngay khi lớp giáp 0.5cm chạm nhau (min_clearance >= 0.0mm),
   báo đúng 'SELF COLLISION', không có frame đâm xuyên nào được gửi xuống executor.
8. Độ trễ thẩm định va chạm < 0.3ms, giải IK < 85ms.
"""

import os
import sys
import time
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS, validate_joints
from teleop.teleoperation import SafetyGate, TeleoperationController


class MockGuard:
    def __init__(self):
        self.danger_threshold = 0.028
        self.warn_threshold = 0.042

        class MetricsCalc:
            def __init__(self):
                self.device = "cpu"
            def get_metrics(self, q_tensor):
                import torch
                return {"sigma_min": torch.tensor(0.080)}

        self.metrics_calc = MetricsCalc()


class MockExecutor:
    def __init__(self):
        self.last_sent = None
        self.sent_history = []
        self.stopped = False

    def connect(self):
        pass

    def disconnect(self):
        pass

    def send(self, q):
        self.last_sent = list(q)
        self.sent_history.append(list(q))
        self.stopped = False

    def stop(self, a=10.0):
        self.stopped = True


class MockPoseSource:
    def __init__(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose


class TestCameraSelfCollisionTorture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" CAMERA & MOUNT SELF-COLLISION TORTURE TEST SUITE (1.0CM ARMOR)")
        print("=" * 70)
        cls.solver = PyrokiIKSolver()
        cls.guard = MockGuard()
        cls.gate = SafetyGate(guard=cls.guard, solver=cls.solver)

    def test_01_camera_geometry_and_active_pairs(self):
        """Ca 1: Kiểm tra hình học va chạm D455 và toàn bộ 34 cặp va chạm tự thân"""
        print("\n[Ca 1 - Hình học va chạm D455 & Ma trận 34 cặp va chạm]")
        self.assertIn("d455_link", self.solver.urdf.link_map)
        d455 = self.solver.urdf.link_map["d455_link"]
        self.assertGreater(len(d455.collisions), 0, "d455_link phải có thẻ collision!")

        coll_geom = d455.collisions[0].geometry
        self.assertIsNotNone(coll_geom.cylinder, "d455_link collision phải dùng cylinder bao phủ!")
        cyl_radius = float(coll_geom.cylinder.radius)
        cyl_length = float(coll_geom.cylinder.length)
        print(f"  D455 URDF collision cylinder: length={cyl_length*1000:.1f}mm, radius={cyl_radius*1000:.1f}mm")
        self.assertGreaterEqual(cyl_radius, 0.0265, "Bán kính cylinder D455 phải >= 26.5mm để bao trọn 4 góc mép lưới đỏ 5mm!")
        self.assertGreaterEqual(cyl_length, 0.134, "Chiều dài cylinder D455 phải >= 134mm!")

        # Kiểm tra toàn bộ 34 cặp va chạm
        pair_names = [
            f"{self.solver.coll_robot.link_names[i]} <-> {self.solver.coll_robot.link_names[j]}"
            for i, j in zip(self.solver.coll_robot.active_idx_i, self.solver.coll_robot.active_idx_j)
        ]
        print(f"  Tổng số cặp va chạm đã kích hoạt: {len(pair_names)} cặp")
        self.assertGreaterEqual(len(pair_names), 34, "Phải kích hoạt tối thiểu 34 cặp va chạm tự thân!")

        # Bắt buộc có các cặp camera D455
        required_d455_targets = [
            "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_1_motor_link", "wrist_2_motor_link"
        ]
        for tgt in required_d455_targets:
            found = any("d455_link" in p and tgt in p for p in pair_names)
            self.assertTrue(found, f"Thiếu cặp va chạm bắt buộc: d455_link vs {tgt}")

        # Bắt buộc có các cặp camera_mount_link
        required_mount_targets = [
            "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_1_motor_link"
        ]
        for tgt in required_mount_targets:
            found = any("camera_mount_link" in p and tgt in p for p in pair_names)
            self.assertTrue(found, f"Thiếu cặp va chạm bắt buộc: camera_mount_link vs {tgt}")

        print("  -> PASS: Hình học D455 bao phủ 100% và kích hoạt đầy đủ toàn bộ 34 cặp va chạm!")

    def test_02_freedom_in_safe_workspace(self):
        """Ca 2: Trong không gian làm việc an toàn, khoảng cách dương và robot chuyển động tự do"""
        print("\n[Ca 2 - Khớp tự do trong vùng an toàn]")
        q_safe = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        chk = self.solver.check_self_collision(q_safe, verbose=False)
        print(f"  Home pose min_dist: {chk['min_dist']*1000:.1f} mm ({chk['link_a']} vs {chk['link_b']})")
        self.assertTrue(chk["safe"], "Home pose phải an toàn tuyệt đối!")
        self.assertGreater(chk["min_dist"], 0.010, "Khoảng cách tại Home pose phải > 10mm (1.0cm giáp)!")

        gate_res = self.gate.check(q_safe)
        self.assertTrue(gate_res["pass"], "SafetyGate phải cho qua Home pose!")
        print("  -> PASS: Tự do hoàn toàn trong vùng an toàn!")

    def test_03_torture_scenario_1_wrist_fold_into_forearm(self):
        """Kịch bản Torture 1: Gập cổ tay (wrist_1_joint q3) ép thẳng D455 quay ngược vào forearm_link"""
        print("\n[Torture Scenario 1 - Gập D455 ngược vào forearm_link]")
        # q3 = -0.40 rad: sát bề mặt giáp 1.0cm (11.8mm > 10.0mm), safe=True
        q_near = [0.0, -1.5708, 1.5708, -0.40, -1.5708, 0.0]
        chk_near = self.solver.check_self_collision(q_near, verbose=False)
        print(f"  q3 = -0.40 rad -> min_dist: {chk_near['min_dist']*1000:.1f} mm ({chk_near['link_a']} vs {chk_near['link_b']}), safe: {chk_near['safe']}")
        self.assertTrue(chk_near["safe"], "Tại q3 = -0.40 rad, camera sát nhưng chưa vi phạm giáp 1.0cm!")

        # q3 = 0.0 rad: đâm vào forearm_link (min_dist <= 0)
        q_collide = [0.0, -1.5708, 1.5708, 0.0, -1.5708, 0.0]
        chk_coll = self.solver.check_self_collision(q_collide, verbose=False)
        print(f"  q3 =  0.00 rad -> min_dist: {chk_coll['min_dist']*1000:.1f} mm ({chk_coll['link_a']} vs {chk_coll['link_b']}), safe: {chk_coll['safe']}")
        self.assertFalse(chk_coll["safe"], "Tại q3 = 0.0 rad, phải phát hiện va chạm tự thân!")
        self.assertEqual(chk_coll["link_a"], "d455_link")
        self.assertEqual(chk_coll["link_b"], "forearm_link")

        # Kiểm tra tích hợp SafetyGate & TeleoperationController
        gate_res = self.gate.check(q_collide)
        self.assertFalse(gate_res["pass"])
        self.assertIn("self_collision", gate_res["reason"].lower())

        mock_exec = MockExecutor()
        fk_coll = self.solver.fk(q_collide)
        pose_coll = {"pos": list(fk_coll["pos"]), "quat": list(fk_coll["quat"])}

        ctrl = TeleoperationController(
            pose_source=MockPoseSource(pose_coll),
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
        )
        ctrl.sync_with_robot(q_near)
        ctrl.enabled = True
        ctrl.step(q_real=q_near, pose=pose_coll)

        self.assertEqual(ctrl.get_status(), "SELF COLLISION")
        # Điểm gửi xuống robot phải an toàn (min_clearance >= 0.0mm)
        sent_chk = self.solver.check_self_collision(mock_exec.last_sent, verbose=False)
        self.assertTrue(sent_chk["safe"], "Lệnh gửi xuống executor không bao giờ được phép đâm xuyên!")
        self.assertGreaterEqual(sent_chk["min_dist"], -1e-5, "Khoảng cách trên bề mặt giáp phải >= 0.0mm!")
        print("  -> PASS: Chặn đứng tức thì tại bề mặt giáp, báo SELF COLLISION, 0 frame đâm xuyên!")

    def test_04_torture_scenario_2_camera_mount_fold_into_forearm(self):
        """Kịch bản Torture 2: Xoay cổ tay gập giá đỡ camera_mount_link sát forearm_link"""
        print("\n[Torture Scenario 2 - Gập camera_mount_link sát forearm_link]")
        # q4=1.60 rad: sát bề mặt (15.2mm > 10.0mm, safe=True)
        q_near = [0.0, -1.5708, 1.5708, 1.79, 1.60, 0.0]
        chk_near = self.solver.check_self_collision(q_near, verbose=False)
        print(f"  q4 = 1.60 rad -> min_dist: {chk_near['min_dist']*1000:.1f} mm ({chk_near['link_a']} vs {chk_near['link_b']}), safe: {chk_near['safe']}")
        self.assertTrue(chk_near["safe"])

        # q4=2.60 rad: mount va chạm forearm_link (min_dist = -39.9mm)
        q_mount_coll = [0.0, -1.5708, 1.5708, 1.79, 2.60, 0.0]
        chk_mount = self.solver.check_self_collision(q_mount_coll, verbose=False)
        print(f"  Tư thế gập mount -> min_dist: {chk_mount['min_dist']*1000:.1f} mm ({chk_mount['link_a']} vs {chk_mount['link_b']}), safe: {chk_mount['safe']}")
        self.assertFalse(chk_mount["safe"], "Phải phát hiện va chạm tự thân của camera_mount_link!")
        self.assertIn("camera_mount_link", [chk_mount["link_a"], chk_mount["link_b"]])

        # TeleoperationController phải bắt được và báo SELF COLLISION
        mock_exec = MockExecutor()
        fk_mount = self.solver.fk(q_mount_coll)
        pose_mount = {"pos": list(fk_mount["pos"]), "quat": list(fk_mount["quat"])}

        ctrl = TeleoperationController(
            pose_source=MockPoseSource(pose_mount),
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
        )
        ctrl.sync_with_robot(q_near)
        ctrl.enabled = True
        ctrl.step(q_real=q_near, pose=pose_mount)

        self.assertEqual(ctrl.get_status(), "SELF COLLISION")
        sent_chk = self.solver.check_self_collision(mock_exec.last_sent, verbose=False)
        self.assertTrue(sent_chk["safe"], "Robot không được gửi tư thế đâm mount!")
        print("  -> PASS: camera_mount_link va chạm bị chặn 100%, bảo vệ an toàn ngàm!")

    def test_05_torture_scenario_3_elbow_fold_into_upper_arm(self):
        """Kịch bản Torture 3: Gập khuỷu tay đưa Camera D455 ép sát vào upper_arm_link"""
        print("\n[Torture Scenario 3 - Gập khuỷu tay ép Camera vào upper_arm_link]")
        # q2=2.50 rad: sát bắp tay (+8.2mm, safe=True)
        q_near = [0.0, -1.5708, 2.50, -2.50, -1.5708, 0.0]
        chk_near = self.solver.check_self_collision(q_near, verbose=False)
        self.assertTrue(chk_near["safe"])

        # q2=2.80 rad: va chạm upper_arm_link (-25.7mm, safe=False)
        q_elbow_coll = [0.0, -1.5708, 2.80, -2.50, -1.5708, 0.0]
        chk_elbow = self.solver.check_self_collision(q_elbow_coll, verbose=False)
        print(f"  Tư thế gập khuỷu -> min_dist: {chk_elbow['min_dist']*1000:.1f} mm ({chk_elbow['link_a']} vs {chk_elbow['link_b']}), safe: {chk_elbow['safe']}")
        self.assertFalse(chk_elbow["safe"], "Gập sâu sát upper_arm phải bị chặn!")

        mock_exec = MockExecutor()
        fk_elbow = self.solver.fk(q_elbow_coll)
        pose_elbow = {"pos": list(fk_elbow["pos"]), "quat": list(fk_elbow["quat"])}

        ctrl = TeleoperationController(
            pose_source=MockPoseSource(pose_elbow),
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
        )
        ctrl.sync_with_robot(q_near)
        ctrl.enabled = True
        ctrl.step(q_real=q_near, pose=pose_elbow)

        self.assertEqual(ctrl.get_status(), "SELF COLLISION")
        sent_chk = self.solver.check_self_collision(mock_exec.last_sent, verbose=False)
        self.assertTrue(sent_chk["safe"])
        print("  -> PASS: Gập khuỷu ép Camera vào bắp tay được chặn tuyệt đối!")

    def test_06_torture_scenario_4_arm_fold_into_shoulder_base(self):
        """Kịch bản Torture 4: Gập cánh tay đưa Camera D455 ép sát vào shoulder_link / base_link"""
        print("\n[Torture Scenario 4 - Gập cánh tay ép Camera vào shoulder_link / base]")
        # q1=-2.20 rad, q2=2.40: sát vai (10.7mm > 10.0mm, safe=True)
        q_near = [0.0, -2.20, 2.40, -1.5708, -1.5708, 0.0]
        chk_near = self.solver.check_self_collision(q_near, verbose=False)
        self.assertTrue(chk_near["safe"])

        # q1=-2.50 rad, q2=2.40: vi phạm giáp 1.0cm shoulder_link (4.3mm < 10.0mm, safe=False)
        q_shoulder_coll = [0.0, -2.50, 2.40, -1.5708, -1.5708, 0.0]
        chk_sh = self.solver.check_self_collision(q_shoulder_coll, verbose=False)
        print(f"  Tư thế lùi sát đế -> min_dist: {chk_sh['min_dist']*1000:.1f} mm ({chk_sh['link_a']} vs {chk_sh['link_b']}), safe: {chk_sh['safe']}")
        self.assertFalse(chk_sh["safe"], "Gập sát shoulder / base phải bị chặn!")

        mock_exec = MockExecutor()
        fk_sh = self.solver.fk(q_shoulder_coll)
        pose_sh = {"pos": list(fk_sh["pos"]), "quat": list(fk_sh["quat"])}

        ctrl = TeleoperationController(
            pose_source=MockPoseSource(pose_sh),
            solver=self.solver,
            gate=self.gate,
            executor=mock_exec,
        )
        ctrl.sync_with_robot(q_near)
        ctrl.enabled = True
        ctrl.step(q_real=q_near, pose=pose_sh)

        self.assertEqual(ctrl.get_status(), "SELF COLLISION")
        sent_chk = self.solver.check_self_collision(mock_exec.last_sent, verbose=False)
        self.assertTrue(sent_chk["safe"])
        print("  -> PASS: Gập vai ép cụm Camera vào trụ đế bị chặn tuyệt đối!")

    def test_07_torture_scenario_5_camera_sweep_near_wrist_motors(self):
        """Kịch bản Torture 5: Xoay cổ tay đưa D455 tiếp cận củ motor wrist_1_motor_link và wrist_2_motor_link"""
        print("\n[Torture Scenario 5 - Xoay cổ tay tiếp cận củ motor]")
        # Pose tiếp cận gần nhất giữa D455 và wrist_1_motor_link
        q_motor_near = [0.0, -1.5708, 1.5708, -1.5708, 1.5708, 0.94]
        chk_motor = self.solver.check_self_collision(q_motor_near, verbose=False)
        print(f"  q=[0, -1.57, 1.57, -1.57, 1.57, 0.94] -> min_dist: {chk_motor['min_dist']*1000:.1f} mm ({chk_motor['link_a']} vs {chk_motor['link_b']})")
        self.assertTrue(chk_motor["safe"])

        # Kiểm tra khoảng cách tính toán cho cặp d455_link <-> wrist_1_motor_link là chính xác
        dists = self.solver.coll_robot.compute_self_collision_distance(self.solver.robot, np.array(q_motor_near, dtype=np.float32))
        d455_idx = self.solver.coll_robot.link_names.index("d455_link")
        m1_idx = self.solver.coll_robot.link_names.index("wrist_1_motor_link")
        pair_found = False
        for idx, (i, j) in enumerate(zip(self.solver.coll_robot.active_idx_i, self.solver.coll_robot.active_idx_j)):
            if (i == d455_idx and j == m1_idx) or (i == m1_idx and j == d455_idx):
                pair_d = float(dists[idx])
                pair_found = True
                print(f"  Khoảng cách trực tiếp d455 vs wrist_1_motor: {pair_d*1000:.1f} mm")
                self.assertGreater(pair_d, 0.0, "Cặp d455 vs wrist_1_motor phải được giám sát liên tục!")
                break
        self.assertTrue(pair_found, "Phải tìm thấy cặp d455_link vs wrist_1_motor_link trong active pairs!")
        print("  -> PASS: Cụm Camera được giám sát toàn diện với toàn bộ các củ motor!")

    def test_08_post_clamping_safety_integrity(self):
        """Ca 8: Nghiệm q_cmd cuối cùng sau bước Clamping/Scaling luôn được kiểm tra an toàn"""
        print("\n[Ca 8 - Hậu kiểm nghiệm q_cmd sau Clamping / Scaling]")
        # Cố tình giải IK tới đích đâm vào cẳng tay
        fk_danger = self.solver.fk([0.0, -1.5708, 1.5708, 0.0, -1.5708, 0.0])
        pose_danger = {"pos": list(fk_danger["pos"]), "quat": list(fk_danger["quat"])}
        res_ik = self.solver.solve(pose_danger, [0.0, -1.5708, 1.5708, -0.25, -1.5708, 0.0])

        if res_ik["q"] is not None:
            # Nếu bộ giải tìm được nghiệm, nghiệm đó bắt buộc phải an toàn
            chk_cmd = self.solver.check_self_collision(res_ik["q"], verbose=False)
            print(f"  IK Solution clearance: {chk_cmd['min_dist']*1000:.1f} mm, safe: {chk_cmd['safe']}")
            self.assertTrue(chk_cmd["safe"], "Nghiệm q_cmd cuối cùng không bao giờ được phép va chạm!")
        else:
            print("  IK Solver từ chối nghiệm do va chạm (Đúng thiết kế)!")
        print("  -> PASS: Hậu kiểm q_cmd khóa chặt an toàn 100%!")

    def test_09_realtime_latency_benchmark(self):
        q_test = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

        # Warm-up JIT
        for _ in range(10):
            self.solver.check_self_collision(q_test, verbose=False)

        # 100 lần kiểm tra 34 cặp va chạm JIT
        t0 = time.perf_counter()
        for _ in range(100):
            self.solver.check_self_collision(q_test, verbose=False)
        avg_chk_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
        print(f"  Độ trễ kiểm tra 34 cặp va chạm: {avg_chk_ms:.4f} ms (Chỉ tiêu < 0.30 ms)")
        self.assertLess(avg_chk_ms, 0.30, "Độ trễ kiểm tra va chạm phải < 0.30 ms!")

        # 10 lần giải IK với toàn bộ 34 cặp va chạm né vật cản
        pose_test = {"pos": [0.25, 0.15, 0.25], "quat": [1.0, 0.0, 0.0, 0.0]}
        times_ik = []
        for _ in range(10):
            res_ik = self.solver.solve(pose_test, q_test)
            times_ik.append(res_ik["latency_ms"])
        avg_ik_ms = sum(times_ik) / len(times_ik)
        print(f"  Độ trễ giải IK PyRoki CPU (kèm 34 cặp va chạm): {avg_ik_ms:.2f} ms (Chỉ tiêu < 85 ms)")
        self.assertLess(avg_ik_ms, 85.0, "Độ trễ giải IK phải < 85 ms!")
        print("  -> PASS: Hiệu năng thực thi xuất sắc, hoàn toàn đáp ứng thời gian thực!")


if __name__ == "__main__":
    unittest.main()
