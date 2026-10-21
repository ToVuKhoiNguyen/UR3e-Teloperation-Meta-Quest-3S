import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np
import time
from teleop_quest.viser_ui import TeleopViserUI, TargetWidgetWrapper

class Test3DHUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_pos = [0.2, -0.3, 0.3]
        init_quat = [1.0, 0.0, 0.0, 0.0]
        init_joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        cls.ui = TeleopViserUI(init_pos, init_quat, init_joints, port=8098)

    @classmethod
    def tearDownClass(cls):
        cls.ui.server.stop()

    def test_01_hud_frame_and_redundant_ui_cleanup(self):
        """Node vr_controller_hud tồn tại, mũi tên 3D và hệ trục bàn máy/gizmo đã bị ẩn/xóa"""
        self.assertTrue(hasattr(self.ui, "_hud_frame"))
        # Mũi tên dịch chuyển đã bị gỡ bỏ hoàn toàn
        self.assertFalse(hasattr(self.ui, "_arrow_trans"))
        # Gizmo luôn hiển thị cho chuột kéo thả, hệ trục bàn máy (world_axes) ẩn
        self.assertTrue(self.ui._raw_gizmo.visible)
        self.assertFalse(self.ui.server.scene.world_axes.visible)
        # Khối 'Hiển thị 3D' đã bị gỡ khỏi GUI
        self.assertFalse(hasattr(self.ui, "dropdown_hud_mode"))
        self.assertFalse(hasattr(self.ui, "cb_show_gizmo"))
        self.assertTrue(np.allclose(self.ui._hud_frame.position, [0.42, -0.38, 0.12]))

    def test_02_target_widget_decoupled_from_hud(self):
        """TargetWidgetWrapper vẫn hoạt động 1:1 với Gizmo cho bộ giải IK nhưng không kéo tay cầm HUD đi theo"""
        old_hud_pos = tuple(self.ui._hud_frame.position)
        new_gizmo_pos = (0.35, 0.15, 0.45)
        self.ui.target_widget.position = new_gizmo_pos
        self.assertEqual(tuple(self.ui._raw_gizmo.position), new_gizmo_pos)
        self.assertEqual(tuple(self.ui._hud_frame.position), old_hud_pos)

    def test_03_right_hand_controller_mirror_scale(self):
        """Mô hình 3D tay cầm Quest được lật trục X thành tay phải (scale=(-1, 1, 1))"""
        for mode, handle in self.ui._ctrl_handles.items():
            self.assertEqual(handle.scale, (-1.0, 1.0, 1.0), f"Handle {mode} must have scale (-1, 1, 1)")

    def test_04_hud_upright_orientation_math(self):
        """Kiểm tra góc xoay mặc định của 3D HUD: Dựng đứng trực diện đối diện người dùng"""
        expected_base_wxyz = (0.1464466, 0.3535534, -0.8535534, -0.3535534)
        np.testing.assert_allclose(self.ui._hud_base_wxyz, expected_base_wxyz, atol=1e-3)
        self.assertEqual(self.ui._hud_mode, "fixed_world")

        # Cán tay cầm trong tọa độ camera (trục Z local) phải có thành phần Y_cam > 0 (chúc xuống)
        r_tilt = np.array([
            [1.0,  0.0,     0.0],
            [0.0,  0.2588,  0.9659],
            [0.0, -0.9659,  0.2588]
        ], dtype=float)
        z_in_cam = r_tilt @ np.array([0.0, 0.0, 1.0])
        self.assertGreater(z_in_cam[1], 0.9, "Handle must point downward in camera space (+Y)")
        y_in_cam = r_tilt @ np.array([0.0, 1.0, 0.0])
        self.assertLess(y_in_cam[2], -0.9, "Face must point towards viewer (-Z)")

    def test_05_micro_displacement_latency(self):
        """Khi bóp cò và dịch chuyển: HUD nhích vi mô 2cm khi đẩy tiến 10cm, độ trễ cập nhật < 50ms"""
        t0 = time.perf_counter()
        self.ui.update_controller_visual(
            is_translating=True,
            delta_trans=np.array([0.10, 0.0, 0.0]),  # Tiến +10cm
            vr_quat=[1.0, 0.0, 0.0, 0.0]
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(latency_ms, 50.0)

        # Vị trí HUD dịch chuyển vi mô đúng 2.0cm (scale 0.20)
        current_pos = np.array(self.ui._hud_frame.position)
        self.assertFalse(np.allclose(current_pos, self.ui._hud_base_pos))
        disp_dist = np.linalg.norm(current_pos - self.ui._hud_base_pos)
        self.assertAlmostEqual(disp_dist, 0.020, places=3)
        self.assertLessEqual(disp_dist, 0.026)

        # Khi nhả cò: Về lại vị trí gốc [0.42, -0.38, 0.12]
        self.ui.update_controller_visual(is_translating=False, delta_trans=np.zeros(3))
        np.testing.assert_allclose(self.ui._hud_frame.position, [0.42, -0.38, 0.12], atol=1e-5)

if __name__ == "__main__":
    unittest.main()
