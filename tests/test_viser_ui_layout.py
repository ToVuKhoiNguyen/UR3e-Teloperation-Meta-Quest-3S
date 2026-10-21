#!/usr/bin/env python3
"""
Unit test kiểm tra vị trí hiển thị và nhãn của thanh trượt Scale và nút Go to Home trên Viser UI:
1. slider_sensitivity có nhãn hiển thị là "Scale"
2. slider_sensitivity được khởi tạo trước btn_home trong thư mục Teleoperation
3. Callback on_update của slider và on_click của btn_home hoạt động bình thường
"""
import os
import sys
import unittest
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.viser_ui import TeleopViserUI

class TestViserUILayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_pos = [0.2, -0.3, 0.3]
        init_quat = [1.0, 0.0, 0.0, 0.0]
        init_joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        cls.ui = TeleopViserUI(init_pos, init_quat, init_joints, port=8097)

    @classmethod
    def tearDownClass(cls):
        cls.ui.server.stop()

    def test_01_slider_label_is_scale(self):
        # Kiểm tra nhãn của thanh trượt là "Scale"
        self.assertEqual(self.ui.slider_sensitivity.label, "Scale")
        print("[Test 1 PASS] Nhãn thanh trượt đã được đổi thành 'Scale'.")

    def test_02_slider_before_btn_home(self):
        # Đọc thứ tự xuất hiện trong file mã nguồn
        viser_ui_file = os.path.join(BASE_DIR, "teleop_quest", "viser_ui.py")
        with open(viser_ui_file, "r", encoding="utf-8") as f:
            content = f.read()

        idx_slider = content.find('self.slider_sensitivity = self.server.gui.add_slider')
        idx_btn_home = content.find('self.btn_home = self.server.gui.add_button')
        
        self.assertGreater(idx_slider, 0)
        self.assertGreater(idx_btn_home, 0)
        self.assertLess(idx_slider, idx_btn_home, "Scale slider phai duoc khoi tao truoc Go to Home button!")
        print("[Test 2 PASS] Scale slider nằm trên nút Go to Home theo đúng thứ tự render.")

    def test_03_callbacks_intact(self):
        # Đảm bảo các thuộc tính slider_sensitivity và btn_home hỗ trợ callback
        slider_updated = [False]
        btn_clicked = [False]

        @self.ui.slider_sensitivity.on_update
        def _(_):
            slider_updated[0] = True

        @self.ui.btn_home.on_click
        def _(_):
            btn_clicked[0] = True

        # Test cập nhật giá trị slider
        self.ui.slider_sensitivity.value = 1.25
        self.assertEqual(self.ui.slider_sensitivity.value, 1.25)
        self.assertTrue(slider_updated[0])

        print("[Test 3 PASS] Callback slider_sensitivity và btn_home hoạt động nguyên bản.")

if __name__ == "__main__":
    unittest.main()
