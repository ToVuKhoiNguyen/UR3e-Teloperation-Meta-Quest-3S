import os
import time
import socket
import threading
import numpy as np
from core.pyroki_ursim_control import send_urscript


def unlock_robot_dashboard(robot_ip: str = "127.0.0.1"):
    """
    Tự động mở khóa Protective Stop / Safety Popup qua Dashboard Server (Port 29999).
    Điều này giải cứu robot khi bị dừng khẩn cấp do va chạm hoặc vặn xoắn.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.5)
        s.connect((robot_ip, 29999))
        time.sleep(0.1)
        try:
            s.recv(1024)
        except Exception:
            pass

        # Gửi chuỗi lệnh mở khóa an toàn chuẩn của Universal Robots
        s.sendall(b"unlock protective stop\n")
        time.sleep(0.2)
        s.sendall(b"close safety popup\n")
        time.sleep(0.2)
        s.sendall(b"play\n")
        time.sleep(0.2)
        s.close()
        print("[HomePlanner] Đã gửi lệnh mở khóa Protective Stop qua Dashboard Port 29999.")
    except Exception as e:
        print(f"[HomePlanner] Cảnh báo kết nối Dashboard: {e}")


class HomePlanner:
    """
    Quản lý quy hoạch và thực thi quỹ đạo trở về vị trí Home an toàn trên CPU:
      - Tự động mở khóa Protective Stop nếu robot bị kẹt
      - Tạo quỹ đạo không gian khớp an toàn (nâng bắp tay tránh va chạm bàn, sau đó xoay về Home)
      - Thực thi mượt mà qua bộ sinh quỹ đạo phần cứng 500Hz movej của URScript
    """
    def __init__(self, robot_config_path: str = None, scene_file: str = None, viz = None, default_home_q: list = None):
        self.viz = viz
        # Vị trí Home an toàn chuẩn của UR3e (khuỷu tay cong tự nhiên, Z=33cm, tuyệt đối không tự va chạm)
        self.default_home_q = list(default_home_q) if default_home_q is not None else [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
        print("[HomePlanner] Khởi tạo bộ quy hoạch Go Home an toàn (CPU-only).")

    def go_home(
        self,
        teleop,
        executor,
        ctrl,
        target_widget,
        widget_pose_source,
        ui_status,
        cb_teleop,
        is_navigating_home: threading.Event = None,
        poll_sleep: float = 0.05,
        quest_server = None
    ) -> bool:
        """Thực hiện quy hoạch và chạy về vị trí Home an toàn"""
        if is_navigating_home is not None:
            is_navigating_home.set()

        try:
            # 1. Tắt Teleop an toàn và giải phóng tài nguyên RTDE
            cb_teleop.value = False
            teleop.enabled = False
            executor.disconnect()
            time.sleep(0.3)

            # 2. Mở khóa Dashboard nếu robot đang bị dừng khẩn cấp / Protective Stop
            robot_ip = os.environ.get("ROBOT_IP", "127.0.0.1")
            unlock_robot_dashboard(robot_ip)

            print("[HomePlanner] Đang tính toán quỹ đạo về Home...")
            ui_status.value = "Planning to Home..."

            q_curr = list(ctrl.joints())
            target_home = list(self.default_home_q)

            # 3. Tạo quỹ đạo an toàn 2 pha (tránh cào mặt bàn):
            # Pha 1: Nâng cổ tay và khuỷu tay lên vị trí an toàn nếu đang ở thấp
            # Pha 2: Di chuyển toàn bộ các khớp về vị trí target_home
            waypoints = []
            
            # Nếu vai hoặc khuỷu tay đang chúc thấp, tạo một waypoint trung gian nhấc cao
            q_intermediate = list(q_curr)
            if q_curr[1] > -1.8:  # Vai đang chúi về phía trước
                q_intermediate[1] = -2.0
                q_intermediate[2] = min(q_curr[2], 1.8)
                waypoints.append(q_intermediate)

            waypoints.append(target_home)

            script_lines = ["def home_trajectory():"]
            for idx, pt in enumerate(waypoints):
                r = 0.03 if idx < len(waypoints) - 1 else 0.0
                pt_str = ", ".join(f"{x:.5f}" for x in pt)
                script_lines.append(f"  movej([{pt_str}], a=0.8, v=0.5, r={r})")
            script_lines.append("end")
            script_lines.append("home_trajectory()\n")
            script = "\n".join(script_lines)

            # 4. Gửi lệnh qua Secondary Port 30002
            print("[HomePlanner] Thực thi quỹ đạo chuyển động mượt qua bộ sinh quỹ đạo phần cứng URScript...")
            send_urscript(script, settle=0.1)

            # 5. Vòng lặp giám sát và cập nhật Viser 3D mượt mà theo góc khớp thực
            t_wait_start = time.time()
            while time.time() - t_wait_start < 8.0:
                q_now = ctrl.joints()
                if self.viz is not None and q_now is not None:
                    if hasattr(self.viz, "update_cfg"):
                        self.viz.update_cfg(np.array(q_now, dtype=float))
                    elif hasattr(self.viz, "update_robot_state"):
                        self.viz.update_robot_state(q_now)
                if quest_server is not None:
                    try:
                        fk_h = teleop._solver.fk(q_now)
                        tcp_h = ctrl.tcp()
                        quest_server.broadcast_robot_telemetry(
                            pos=fk_h["pos"],
                            quat=fk_h["quat"],
                            status="Returning Home",
                            gripper=False,
                            joints=q_now,
                            tcp_pose=tcp_h
                        )
                    except Exception:
                        pass
                max_dev = max(abs(q_now[i] - target_home[i]) for i in range(6))
                if max_dev < 0.02:
                    break
                time.sleep(0.04)

            # Cập nhật dáng robot lần cuối khi đã tới đích
            q_final = ctrl.joints()
            if self.viz is not None and q_final is not None:
                if hasattr(self.viz, "update_cfg"):
                    self.viz.update_cfg(np.array(q_final, dtype=float))
                elif hasattr(self.viz, "update_robot_state"):
                    self.viz.update_robot_state(q_final)

            ui_status.value = "Safely returned to Home!"
            print("[HomePlanner] Đã về Home an toàn!")

            # 6. Đồng bộ lại Target Widget tại vị trí mới bằng Forward Kinematics
            try:
                fk_res = teleop._solver.fk(target_home)
                target_widget.position = tuple(fk_res["pos"])
                target_widget.wxyz = tuple(fk_res["quat"])
            except Exception as ex:
                print(f"[HomePlanner] Cảnh báo cập nhật widget: {ex}")

            if widget_pose_source is not None:
                if hasattr(widget_pose_source, "sync_with_robot"):
                    widget_pose_source.sync_with_robot(target_home)
                else:
                    widget_pose_source._ema_pos = None
                    widget_pose_source._ema_quat_xyzw = None
            teleop._last_valid_q = list(target_home)

            # Bật lại cờ Teleop trên UI sau khi robot đã về đích hoàn tất
            cb_teleop.value = True
            return True

        except Exception as e:
            print(f"[HomePlanner] Lỗi khi thực hiện Go Home: {e}")
            ui_status.value = "Lỗi Go Home!"
            return False
        finally:
            if is_navigating_home is not None:
                is_navigating_home.clear()
