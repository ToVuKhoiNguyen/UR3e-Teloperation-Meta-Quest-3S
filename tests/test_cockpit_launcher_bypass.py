#!/usr/bin/env python3
"""
Unit test kiểm tra việc bỏ qua Launcher Screen và tải trực tiếp vào Cockpit:
1. index.html: #launcher-screen bị ẩn (display: none), #vr-overlay mở sẵn (display: flex)
2. index.html: #btn-enter-vr nằm trên thanh công cụ header-right của Cockpit
3. style.css: #vr-overlay mặc định display: flex, #launcher-screen display: none
4. main.js: checkWebXR, onSessionStarted, onSessionEnded hoạt động đúng logic không gây lỗi
"""
import os
import sys
import unittest
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "teleop_quest", "static")

class TestCockpitLauncherBypass(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            self.html = f.read()
        with open(os.path.join(STATIC_DIR, "style.css"), "r", encoding="utf-8") as f:
            self.css = f.read()
        with open(os.path.join(STATIC_DIR, "main.js"), "r", encoding="utf-8") as f:
            self.js = f.read()

    def test_01_launcher_screen_hidden_and_cockpit_active_in_html(self):
        # 1. Launcher Screen phải có style="display: none;"
        self.assertRegex(self.html, r'<div id="launcher-screen"[^>]*style="[^"]*display:\s*none', "Launcher screen phai co display: none")
        # 2. VR Overlay Cockpit phải có style="display: flex;" để tải tức thì 0ms
        self.assertRegex(self.html, r'<div id="vr-overlay"[^>]*style="[^"]*display:\s*flex', "vr-overlay phai co display: flex de tai truc tiep 0ms")
        print("[Test 1 PASS] index.html: Launcher Screen an, Cockpit Overlay bat mac dinh (0ms transition).")

    def test_02_enter_vr_button_in_cockpit_header(self):
        # Nút btn-enter-vr phải nằm trong khối header-right của cockpit
        header_right_match = re.search(r'<div class="header-right">(.*?)</div>', self.html, re.DOTALL)
        self.assertIsNotNone(header_right_match, "Khong tim thay header-right trong HTML")
        header_content = header_right_match.group(1)
        self.assertIn('id="btn-enter-vr"', header_content, "btn-enter-vr phai nam trong header-right")
        self.assertIn('id="btn-exit-vr"', header_content, "btn-exit-vr phai nam trong header-right")
        print("[Test 2 PASS] index.html: Nút 'Enter VR' va 'Exit VR' duoc bo tri chuan tren Cockpit Header.")

    def test_03_css_rules_match_direct_cockpit(self):
        # vr-overlay phải có display: flex
        vr_overlay_css = re.search(r'#vr-overlay\s*\{([^}]+)\}', self.css)
        self.assertIsNotNone(vr_overlay_css)
        self.assertIn('display: flex;', vr_overlay_css.group(1))

        # launcher-screen phải có display: none
        launcher_css = re.search(r'#launcher-screen\s*\{([^}]+)\}', self.css)
        self.assertIsNotNone(launcher_css)
        self.assertIn('display: none;', launcher_css.group(1))
        print("[Test 3 PASS] style.css: #vr-overlay display: flex, #launcher-screen display: none dong bo.")

    def test_04_main_js_logic_flow(self):
        # 1. Khởi tạo an toàn cho host-ip và status
        self.assertIn('if (elHostIp)', self.js)
        self.assertIn("const statusDiv = document.getElementById('status')", self.js)

        # 2. onSessionEnded không được hiện lại launcher-screen
        ended_func = re.search(r'function onSessionEnded\s*\([^\)]*\)\s*\{([^}]+)\}', self.js)
        self.assertIsNotNone(ended_func)
        self.assertNotIn("document.getElementById('launcher-screen').style.display = 'flex'", ended_func.group(1))
        self.assertIn("vrOverlay.style.display = 'flex'", ended_func.group(1))

        # 3. Enter VR click gọi requestSession trực tiếp (User Gesture an toàn)
        enter_click = re.search(r"btnEnterVR\.addEventListener\('click'[\s\S]*?navigator\.xr\.requestSession", self.js)
        self.assertIsNotNone(enter_click, "requestSession phai duoc goi dong bo trong click handler")
        print("[Test 4 PASS] main.js: Luong User Gesture chuan khong loi NotAllowedError, session end duy tri Cockpit.")

if __name__ == "__main__":
    unittest.main()
