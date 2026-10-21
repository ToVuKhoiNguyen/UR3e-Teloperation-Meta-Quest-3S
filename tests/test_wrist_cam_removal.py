#!/usr/bin/env python3
"""
Unit test kiem tra viec go bo khoi Wrist Camera tren Viser UI
va xu ly an toan luong cap nhat anh backend.
"""
import sys
import os
import time
import unittest
import numpy as np
from unittest.mock import MagicMock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from teleop_quest.viser_ui import TeleopViserUI
from teleop_quest.camera_pipeline import CameraStreamPipeline

class TestWristCamRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Khoi tao TeleopViserUI tren port 8099
        cls.ui = TeleopViserUI(
            init_pos=[0.25, 0.15, 0.20],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            init_joints=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            port=8099
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "ui") and hasattr(cls.ui, "server"):
            cls.ui.server.stop()

    def test_ui_wrist_cam_attributes_removed(self):
        # 1. Kiem tra thuoc tinh ui_cam_mode va ui_cam_preview khong con ton tai
        self.assertFalse(hasattr(self.ui, "ui_cam_mode"), "ui_cam_mode van con ton tai tren UI!")
        self.assertFalse(hasattr(self.ui, "ui_cam_preview"), "ui_cam_preview van con ton tai tren UI!")
        print("[Test 1 PASS] ui_cam_mode va ui_cam_preview da duoc go bo triet de.")

    def test_safe_fallbacks_no_attribute_error(self):
        # 2. Goi cac ham du phong, dam bao khong quang ngoai le AttributeError
        mode = self.ui.get_camera_view_mode()
        self.assertEqual(mode, "rgb")

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.ui.update_cam_preview(dummy_img)
            self.ui.update_cam_preview(None)
        except Exception as e:
            self.fail(f"update_cam_preview nem loi bat thuong: {e}")
        print("[Test 2 PASS] get_camera_view_mode tra ve 'rgb' va update_cam_preview chay an toan khong loi.")

    def test_camera_pipeline_throughput_and_isolation(self):
        # 3. Kiem tra CameraStreamPipeline khong bi anh huong, chay an toan
        mock_cam = MagicMock()
        mock_cam.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cam.add_osd_preview.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_quest = MagicMock()

        pipeline = CameraStreamPipeline(
            wrist_cam=mock_cam,
            ui=self.ui,
            quest_server=mock_quest,
            init_pos=[0.25, 0.15, 0.20],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            target_fps=20.0
        )

        pipeline.start()
        time.sleep(0.15) # Cho pipeline chay ~4-5 chu ky

        # Kiem tra Quest server van nhan frame binh thuong
        self.assertTrue(mock_quest.send_camera_frame.called)
        self.assertIsNotNone(pipeline.get_latest_preview_frame())
        pipeline.stop()
        print("[Test 3 PASS] CameraStreamPipeline van truyen hinh anh qua Quest an toan ma khong gap loi.")

if __name__ == "__main__":
    unittest.main()
