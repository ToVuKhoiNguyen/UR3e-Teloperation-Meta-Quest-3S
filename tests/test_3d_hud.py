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

    def test_01_hud_frame_and_arrow_initialized(self):
        """Node vr_controller_hud và mũi tên 3D arrow_trans được tạo độc lập"""
        self.assertTrue(hasattr(self.ui, "_hud_frame"))
        self.assertTrue(hasattr(self.ui, "_arrow_trans"))
        self.assertFalse(self.ui._arrow_trans.visible)
        self.assertTrue(np.allclose(self.ui._hud_frame.position, [0.42, -0.38, 0.12]))

    def test_02_target_widget_decoupled_from_hud(self):
        """TargetWidgetWrapper vẫn hoạt động 1:1 với Gizmo nhưng không kéo tay cầm HUD đi theo"""
        old_hud_pos = tuple(self.ui._hud_frame.position)
        # Di chuyển Gizmo đến vị trí TCP mới
        new_gizmo_pos = (0.35, 0.15, 0.45)
        self.ui.target_widget.position = new_gizmo_pos
        self.assertEqual(tuple(self.ui._raw_gizmo.position), new_gizmo_pos)
        # Vị trí của HUD phải giữ nguyên vị trí bục/viewport, không bị bay theo TCP
        self.assertEqual(tuple(self.ui._hud_frame.position), old_hud_pos)

    def test_03_translation_arrow_and_micro_displacement(self):
        """Khi bóp cò và đẩy Tiến: Mũi tên hiển thị, đổi màu Neon Green, HUD nhích vi mô"""
        t0 = time.perf_counter()
        self.ui.update_controller_visual(
            is_translating=True,
            delta_trans=np.array([0.08, 0.0, 0.0]),  # Tiến +8cm
            vr_quat=[1.0, 0.0, 0.0, 0.0]
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Mũi tên phải hiển thị
        self.assertTrue(self.ui._arrow_trans.visible)
        # Độ trễ cập nhật < 50ms (thực tế < 1ms)
        self.assertLess(latency_ms, 50.0)

        # Màu sắc trục Tiến phải là Neon Emerald Green [0, 255, 180]
        expected_color = np.array([[0, 255, 180]], dtype=np.uint8)
        np.testing.assert_array_equal(self.ui._arrow_trans.colors, expected_color)

        # Vị trí HUD phải có đàn hồi vi mô (khác base_pos)
        current_pos = np.array(self.ui._hud_frame.position)
        self.assertFalse(np.allclose(current_pos, self.ui._hud_base_pos))
        # Độ dịch chuyển không vượt quá 2.5cm
        disp_dist = np.linalg.norm(current_pos - self.ui._hud_base_pos)
        self.assertLessEqual(disp_dist, 0.026)

    def test_04_release_trigger_reset(self):
        """Khi nhả cò: Mũi tên lập tức ẩn và vị trí đàn hồi trở lại vị trí gốc"""
        self.ui.update_controller_visual(
            is_translating=False,
            delta_trans=np.zeros(3)
        )
        self.assertFalse(self.ui._arrow_trans.visible)
        np.testing.assert_allclose(self.ui._hud_frame.position, self.ui._hud_base_pos, atol=1e-5)

    def test_05_directional_arrow_colors(self):
        """Kiểm tra màu sắc mũi tên cho các trục chuyển động khác nhau"""
        # 1. Lùi: Vàng hổ phách [255, 190, 0]
        self.ui.update_controller_visual(is_translating=True, delta_trans=np.array([-0.05, 0.0, 0.0]))
        np.testing.assert_array_equal(self.ui._arrow_trans.colors, np.array([[255, 190, 0]], dtype=np.uint8))

        # 2. Phải / Trái: Xanh dương điện quang [0, 160, 255]
        self.ui.update_controller_visual(is_translating=True, delta_trans=np.array([0.0, 0.06, 0.0]))
        np.testing.assert_array_equal(self.ui._arrow_trans.colors, np.array([[0, 160, 255]], dtype=np.uint8))

        # 3. Cao / Thấp: Tím huỳnh quang [220, 60, 255]
        self.ui.update_controller_visual(is_translating=True, delta_trans=np.array([0.0, 0.0, 0.07]))
        np.testing.assert_array_equal(self.ui._arrow_trans.colors, np.array([[220, 60, 255]], dtype=np.uint8))

        # Trở về trạng thái nghỉ
        self.ui.update_controller_visual(is_translating=False, delta_trans=np.zeros(3))

if __name__ == "__main__":
    unittest.main()
