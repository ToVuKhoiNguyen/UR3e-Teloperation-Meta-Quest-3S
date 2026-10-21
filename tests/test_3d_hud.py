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
        """Kiểm tra góc xoay mặc định của 3D HUD: Dựng đứng trực diện đối diện người dùng trong Camera Space"""
        self.assertEqual(self.ui._hud_mode, "viewport")

        # Cán tay cầm trong tọa độ camera (trục Z local) phải có thành phần Y_cam > 0.90 (chúc xuống)
        z_in_cam = self.ui._r_hud_upright @ np.array([0.0, 0.0, 1.0])
        self.assertGreater(z_in_cam[1], 0.90, "Handle must point downward in camera space (+Y)")
        # Mặt phím trong tọa độ camera (trục Y local) phải có thành phần Z_cam < -0.90 (hướng vào mắt người nhìn)
        y_in_cam = self.ui._r_hud_upright @ np.array([0.0, 1.0, 0.0])
        self.assertLess(y_in_cam[2], -0.90, "Face must point towards viewer (-Z)")

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

        # Khi nhả cò: Về lại vị trí gốc
        self.ui.update_controller_visual(is_translating=False, delta_trans=np.zeros(3))
        np.testing.assert_allclose(self.ui._hud_frame.position, self.ui._hud_base_pos, atol=1e-5)

    def test_06_camera_360_orbit_invariance(self):
        """Khi người dùng xoay sa bàn 3D 360 độ: HUD luôn cố định chính xác tại góc dưới-phải màn hình"""
        import transforms3d
        class MockCam:
            def __init__(self, pos, wxyz):
                self.position = pos
                self.wxyz = wxyz

        angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        for deg in angles:
            rad = np.radians(deg)
            # Camera xoay quanh bàn máy bán kính 1.2m, độ cao 0.5m, nhìn về gốc tọa độ
            pos_cam = np.array([1.2 * np.cos(rad), 1.2 * np.sin(rad), 0.5])
            z_cam = -pos_cam / np.linalg.norm(pos_cam)
            up_world = np.array([0.0, 0.0, 1.0])
            x_cam = np.cross(z_cam, -up_world)
            x_cam /= np.linalg.norm(x_cam)
            y_cam = np.cross(z_cam, x_cam)
            R_cam = np.column_stack([x_cam, y_cam, z_cam])
            Q_cam = tuple(transforms3d.quaternions.mat2quat(R_cam))

            mock_cam = MockCam(pos_cam, Q_cam)
            self.ui._sync_hud_to_camera(mock_cam)

            # Quy đổi ngược vị trí HUD thế giới về hệ tọa độ Camera Viewport:
            p_hud_world = np.array(self.ui._hud_frame.position)
            p_in_cam = R_cam.T @ (p_hud_world - pos_cam)

            # Tọa độ trong Camera Viewport luôn phải dính chặt [s_right, s_down, d_fwd] = [0.20, 0.14, 0.50]
            np.testing.assert_allclose(p_in_cam, [0.20, 0.14, 0.50], atol=1e-5,
                                       err_msg=f"HUD bi lech khoi goc man hinh tai goc xoay {deg} do")

if __name__ == "__main__":
    unittest.main()
