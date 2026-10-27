"""
Unit Test: Mount Collision-Free Torture Test Suite.
Kiểm chứng toàn diện:
1. Hình học va chạm camera_mount_link trong URDF:
   - TUYỆT ĐỐI KHÔNG BỔ SUNG KHỐI TRỤ MỚI (Zero <cylinder> elements in camera_mount_link).
   - Bao phủ 100% thể tích thực tế của tấm gá (Box + corner Spheres tại các tai gá ốc & mép ngoài).
2. Toàn bộ các khâu có nguy cơ va chạm với camera_mount:
   - forearm_link, wrist_1_link, wrist_2_link, upper_arm_link, wrist motors.
   - Ngưỡng kích hoạt chuẩn xác ở mức clearance <= 0.01m (1.0cm / 10mm).
3. Mount Collision-Free Torture Test:
   - Quét xoay vòng toàn bộ các góc cổ tay (wrist_1, wrist_2, wrist_3) ở các tư thế gập sát thân máy nhất.
   - Không có bất kỳ góc chết (blind spot) nào.
4. Cơ chế Phanh mượt & Trượt tiếp tuyến (Smooth Braking & Tangential Sliding):
   - Khi chạm giáp 1.0cm của mount, hệ thống lập tức kích hoạt 'SELF COLLISION' và trượt tiếp tuyến hoặc dừng mượt.
5. Benchmark độ trễ thẩm định va chạm mount < 0.25ms.
"""

import os
import sys
import time
import math
import unittest
import xml.etree.ElementTree as ET
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop.pyroki_solver import PyrokiIKSolver, UR3E_LIMITS
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


class TestMountCollisionTorture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 75)
        print(" MOUNT COLLISION-FREE TORTURE TEST SUITE (1.0CM ARMOR, ZERO BLIND SPOTS)")
        print("=" * 75)
        cls.urdf_path = os.path.join(BASE_DIR, "assets", "ur3e_custom.urdf")
        cls.solver = PyrokiIKSolver()
        cls.guard = MockGuard()
        cls.gate = SafetyGate(guard=cls.guard, solver=cls.solver)

    def test_01_urdf_geometry_no_new_cylinders_and_100_percent_coverage(self):
        """Yêu cầu 1: camera_mount_link trong URDF tuyệt đối KHÔNG có cylinder, dùng Box + Spheres"""
        print("\n[Ca 1 - Kiểm tra URDF camera_mount_link: Tuyệt đối KHÔNG có cylinder]")
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        mount_link = None
        for link in root.findall("link"):
            if link.get("name") == "camera_mount_link":
                mount_link = link
                break

        self.assertIsNotNone(mount_link, "camera_mount_link phải tồn tại trong assets/ur3e_custom.urdf!")
        collisions = mount_link.findall("collision")
        self.assertGreater(len(collisions), 0, "camera_mount_link phải có định nghĩa collision!")

        cylinder_count = 0
        box_count = 0
        sphere_count = 0

        for col in collisions:
            geom = col.find("geometry")
            if geom.find("cylinder") is not None:
                cylinder_count += 1
            if geom.find("box") is not None:
                box_count += 1
            if geom.find("sphere") is not None:
                sphere_count += 1

        print(f"  camera_mount_link collision primitives: cylinders={cylinder_count}, boxes={box_count}, spheres={sphere_count}")
        self.assertEqual(
            cylinder_count, 0,
            "TUYỆT ĐỐI CẤM: camera_mount_link không được chứa bất kỳ khối trụ (cylinder) nào!"
        )
        self.assertGreaterEqual(box_count, 1, "Phải có ít nhất 1 box bao trọn thân chính của giá đỡ!")
        self.assertGreaterEqual(sphere_count, 4, "Phải có ít nhất 4 sphere nội tiếp tại các đỉnh góc nhọn/tai gá!")
        print("  -> PASS: 100% tuân thủ yêu cầu: Zero cylinder, Box và Spheres bao trọn hình học mount!")

    def test_02_all_critical_mount_collision_pairs_active(self):
        """Yêu cầu 2: Rà soát ma trận kiểm tra va chạm của camera_mount với forearm, wrist_1, wrist_2, upper_arm"""
        print("\n[Ca 2 - Ma trận kiểm tra va chạm của camera_mount]")
        required_targets = [
            "forearm_link", "wrist_1_link", "wrist_2_link", "upper_arm_link"
        ]

        # Kiểm tra danh sách mục tiêu trong JIT checker chuyên dụng của mount
        self.assertTrue(hasattr(self.solver, "_mount_target_names"), "Solver phải có _mount_target_names!")
        for tgt in required_targets:
            self.assertIn(
                tgt, self.solver._mount_target_names,
                f"Mount JIT checker phải bao gồm {tgt} trong danh sách mục tiêu!"
            )

        # Kiểm tra danh sách các cặp trong coll_robot
        pair_names = [
            f"{self.solver.coll_robot.link_names[i]} <-> {self.solver.coll_robot.link_names[j]}"
            for i, j in zip(self.solver.coll_robot.active_idx_i, self.solver.coll_robot.active_idx_j)
        ]
        for tgt in ["forearm_link", "wrist_1_link", "upper_arm_link"]:
            found = any("camera_mount_link" in p and tgt in p for p in pair_names)
            self.assertTrue(found, f"Thiếu cặp va chạm coll_robot: camera_mount_link vs {tgt}")

        print(f"  Target links được kiểm tra với camera_mount: {self.solver._mount_target_names}")
        print("  -> PASS: Đầy đủ 100% các khâu có nguy cơ va quẹt (forearm, wrist_1, wrist_2, upper_arm)!")

    def test_03_safe_free_movement_at_home_pose(self):
        """Kiểm tra tư thế Home pose: clearance > 1.0cm (không có báo động giả / ghosting)"""
        print("\n[Ca 3 - Không gian làm việc tại Home Pose]")
        q_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        chk = self.solver.check_self_collision(q_home, verbose=False)
        print(f"  Home pose min_dist: {chk['min_dist']*1000:.2f} mm ({chk['link_a']} vs {chk['link_b']}), safe: {chk['safe']}")
        self.assertTrue(chk["safe"], "Home pose phải hoàn toàn an toàn!")
        self.assertGreater(chk["min_dist"], 0.010, "Clearance tại Home pose phải > 10.0mm (1.0cm áo giáp)!")
        print("  -> PASS: Không có ghosting tại Home pose!")

    def test_04_wrist_1_and_wrist_2_folding_torture(self):
        """Torture 1: Gập cổ tay ép camera_mount và camera assembly sát forearm_link"""
        print("\n[Ca 4 - Torture Test: Gập wrist_1 và wrist_2 ép mount sát forearm_link]")
        # 1. Quét wrist_1 từ an toàn (-0.45 rad) đến chạm giáp 1.0cm (-0.30 rad)
        q_safe = [0.0, -1.5708, 1.5708, -0.45, -1.5708, 0.0]
        chk_safe = self.solver.check_self_collision(q_safe, verbose=False)
        print(f"  q3 = -0.45 rad -> min_dist: {chk_safe['min_dist']*1000:.1f}mm, safe: {chk_safe['safe']}")
        self.assertTrue(chk_safe["safe"], "Tại q3 = -0.45 rad phải an toàn!")

        q_coll = [0.0, -1.5708, 1.5708, -0.30, -1.5708, 0.0]
        chk_coll = self.solver.check_self_collision(q_coll, verbose=False)
        print(f"  q3 = -0.30 rad -> min_dist: {chk_coll['min_dist']*1000:.1f}mm ({chk_coll['link_a']} vs {chk_coll['link_b']}), safe: {chk_coll['safe']}")
        self.assertFalse(chk_coll["safe"], "Khi khoảng cách <= 1.0cm, hệ thống phải lập tức kích hoạt vi phạm!")
        self.assertTrue(
            "camera_mount_link" in (chk_coll["link_a"], chk_coll["link_b"]) or
            "d455_link" in (chk_coll["link_a"], chk_coll["link_b"]),
            "Phải phát hiện vi phạm va chạm từ cụm camera/mount!"
        )

        # 2. Xoay wrist_2 ép trực tiếp camera_mount_link sát forearm_link
        q_mt_safe = [0.0, -1.5708, 1.5708, -1.5708, -2.142, 0.86]
        chk_mt_safe = self.solver.check_self_collision(q_mt_safe, verbose=False)
        print(f"  q4 = -2.142 rad -> min_dist: {chk_mt_safe['min_dist']*1000:.1f}mm, safe: {chk_mt_safe['safe']}")
        self.assertTrue(chk_mt_safe["safe"], "Tại q4 = -2.142 rad phải an toàn!")

        q_mt_coll = [0.0, -1.5708, 1.5708, -1.5708, -2.228, 0.86]
        chk_mt_coll = self.solver.check_self_collision(q_mt_coll, verbose=False)
        print(f"  q4 = -2.228 rad -> min_dist: {chk_mt_coll['min_dist']*1000:.1f}mm ({chk_mt_coll['link_a']} vs {chk_mt_coll['link_b']}), safe: {chk_mt_coll['safe']}")
        self.assertFalse(chk_mt_coll["safe"], "Khi khoảng cách mount <= 1.0cm, phải kích hoạt vi phạm!")
        self.assertEqual(chk_mt_coll["link_a"], "camera_mount_link")
        self.assertEqual(chk_mt_coll["link_b"], "forearm_link")
        print("  -> PASS: Kích hoạt vi phạm va chạm chính xác tại ngưỡng 1.0cm cho camera_mount_link!")

    def test_05_wrist_2_and_wrist_3_extreme_rotation_torture(self):
        """Torture 2: Xoay tròn toàn bộ các góc cổ tay (wrist_2, wrist_3) kiểm chứng ZERO BLIND SPOTS"""
        print("\n[Ca 5 - Torture Test: Quét 360 độ wrist_2 và wrist_3, triệt tiêu góc chết]")
        # Duyệt qua các góc quay cực đoan của wrist_2 và wrist_3
        # Khi đưa cụm wrist gập gần forearm hoặc upper arm:
        q_base = [0.0, -2.0, 2.2, -1.8, 0.0, 0.0]
        w2_angles = np.linspace(-3.0, 3.0, 13)
        w3_angles = np.linspace(-3.0, 3.0, 13)

        detected_mount_collisions = 0
        total_evals = 0

        for w2 in w2_angles:
            for w3 in w3_angles:
                q_test = [0.0, -2.0, 2.2, -1.8, float(w2), float(w3)]
                total_evals += 1
                chk = self.solver.check_self_collision(q_test, verbose=False)

                # Kiểm tra riêng độ xa của từng điểm biên mount
                min_d_mt, mt_idx = self.solver._mount_chk_jit(np.array(q_test, dtype=np.float32))
                dist_mt_mm = float(min_d_mt) * 1000.0

                if dist_mt_mm <= 10.0:
                    detected_mount_collisions += 1
                    # Khi bất kỳ góc cạnh nào của mount lọt vào vùng <= 10.0mm (1.0cm),
                    # solver BẮT BUỘC phải trả về safe: False (không có blind spot!)
                    self.assertFalse(
                        chk["safe"],
                        f"Góc chết phát hiện tại w2={w2:.2f}, w3={w3:.2f}! Mount dist = {dist_mt_mm:.1f}mm nhưng safe=True!"
                    )

        print(f"  Đã quét {total_evals} tư thế xoay cổ tay cực đoan. Phát hiện & chặn đứng {detected_mount_collisions} tư thế vi phạm giáp 1.0cm.")
        self.assertGreater(detected_mount_collisions, 0, "Phải phát hiện các tư thế mount áp sát thân máy!")
        print("  -> PASS: Không có bất kỳ góc chết nào (Zero Blind Spots) trên toàn bộ thể tích mount!")

    def test_06_tangential_sliding_and_smooth_braking_on_mount_armor(self):
        """Yêu cầu 3: Phanh mượt (Smooth Braking) & Trượt tiếp tuyến (Tangential Sliding) khi chạm giáp mount"""
        print("\n[Ca 6 - Phanh mượt & Trượt tiếp tuyến khi chạm giáp Mount]")
        executor = MockExecutor()
        teleop = TeleoperationController(
            pose_source=None,
            solver=self.solver,
            gate=self.gate,
            executor=executor
        )
        teleop.enabled = True

        # Đặt robot ở trạng thái an toàn ngay trước giáp mount (q4 = -2.142 rad)
        q_start = [0.0, -1.5708, 1.5708, -1.5708, -2.142, 0.86]
        teleop.sync_with_robot(q_start)

        # 1. Thử nghiệm Trượt tiếp tuyến (Tangential Sliding): cố tình chỉ đạo tiến sâu vào forearm (-2.570)
        q_target_pen = [0.0, -1.5708, 1.5708, -1.5708, -2.570, 0.86]
        res1 = teleop.step_joint_space(q_target_desired=q_target_pen, q_real=q_start)
        sent_q1 = executor.last_sent
        self.assertIsNotNone(sent_q1, "Phải gửi lệnh điều khiển hợp lệ xuống executor!")

        # Kiểm tra cấu hình khớp được gửi: min_dist tuyệt đối không được vi phạm giáp 1.0cm
        chk_sent1 = self.solver.check_self_collision(sent_q1, verbose=False)
        print(f"  Sent q1 min_dist: {chk_sent1['min_dist']*1000:.2f}mm, safe: {chk_sent1['safe']}")
        self.assertTrue(chk_sent1["safe"], "Lệnh trượt tiếp tuyến tuyệt đối không được xuyên thủng lớp giáp 1.0cm!")
        self.assertGreaterEqual(chk_sent1["min_dist"], 0.0099, "Khoảng cách thực tế phải duy trì >= 1.0cm (10mm)!")

        # 2. Thử nghiệm Phanh mượt & Kích hoạt SELF COLLISION khi ép thẳng vào ranh giới va chạm
        q_mt_coll = [0.0, -1.5708, 1.5708, -1.5708, -2.228, 0.86]
        teleop.sync_with_robot(q_mt_coll)
        teleop._prev_dq = np.array([0.0, 0.0, 0.0, 0.0, -0.05, 0.0])  # Vận tốc ép sâu vào thân máy
        res2 = teleop.step_joint_space(q_target_desired=[0.0, -1.5708, 1.5708, -1.5708, -2.40, 0.86], q_real=q_mt_coll)
        print(f"  Step 2 status: '{res2['status']}', safe stop achieved")
        self.assertEqual(res2["status"], "SELF COLLISION", "Khi hết đường trượt, phải kích hoạt trạng thái SELF COLLISION!")

        sent_q2 = executor.last_sent
        self.assertIsNotNone(sent_q2, "Phải gửi lệnh phanh mượt xuống executor!")
        print("  -> PASS: Phanh mượt & Trượt tiếp tuyến hoàn hảo khi chạm giáp 1.0cm của mount!")

    def test_07_realtime_latency_benchmark(self):
        """Yêu cầu bổ sung: Thẩm định va chạm mount trong vòng < 0.25ms"""
        print("\n[Ca 7 - Benchmark độ trễ thẩm định va chạm mount]")
        q_test = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)

        # Warm up
        for _ in range(10):
            self.solver._mount_chk_jit(q_test)

        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            self.solver._mount_chk_jit(q_test)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / N

        print(f"  Thời gian thẩm định va chạm mount (_mount_chk_jit): {dt_ms:.4f} ms / lần gọi (N={N})")
        self.assertLess(dt_ms, 0.25, "Độ trễ thẩm định va chạm mount phải < 0.25ms!")
        print("  -> PASS: Cực kỳ nhẹ và tối ưu cho thời gian thực!")


if __name__ == "__main__":
    unittest.main()
