#!/usr/bin/env python3
"""
Script reset robot UR3e / URSim ve vi tri Safe Home chuan:
  SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]

Chuc nang:
  1. Tu dong ket noi Dashboard Port 29999 de mo khoa Protective Stop / Safety Popup
  2. Doc trang thai goc khop hien tai qua RTDE Port 30004
  3. Gui lenh movej qua Secondary Port 30002 de bo noi suy phan cung 500Hz cua UR
     dieu phoi robot ve Home em ai, chong rung giat va tranh loi van xoan khop
  4. Giam sat thoi gian thuc den khi robot ve dung dich
"""

import os
import sys
import time
import socket
import argparse
import math

SAFE_HOME_JOINTS = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
DASHBOARD_PORT = 29999
SECONDARY_PORT = 30002


def unlock_dashboard(robot_ip: str) -> bool:
    """Mo khoa Protective Stop va tiep tuc chuong trinh qua Dashboard Port 29999."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((robot_ip, DASHBOARD_PORT))
        time.sleep(0.1)
        try:
            s.recv(1024)
        except Exception:
            pass

        # Gui chuoi lenh giai phong an toan cua Universal Robots
        s.sendall(b"unlock protective stop\n")
        time.sleep(0.2)
        s.sendall(b"close safety popup\n")
        time.sleep(0.2)
        s.sendall(b"play\n")
        time.sleep(0.2)
        s.close()
        print(f"[Dashboard] Da gui lenh mo khoa Protective Stop toi {robot_ip}:{DASHBOARD_PORT}")
        return True
    except Exception as e:
        print(f"[Dashboard] Canh bao ket noi Port {DASHBOARD_PORT}: {e}")
        return False


def send_movej_script(robot_ip: str, target_joints: list, accel: float = 0.8, speed: float = 0.5) -> bool:
    """Gui doan lenh URScript movej qua Secondary Port 30002."""
    try:
        target_str = ", ".join(f"{x:.6f}" for x in target_joints)
        script = (
            f"def reset_home_prog():\n"
            f"  movej([{target_str}], a={accel:.3f}, v={speed:.3f}, r=0.0)\n"
            f"end\n"
            f"reset_home_prog()\n"
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((robot_ip, SECONDARY_PORT))
        time.sleep(0.1)
        try:
            s.recv(4096)
        except Exception:
            pass
        s.sendall(script.encode("utf-8"))
        time.sleep(0.2)
        s.close()
        print(f"[URScript] Da nạp lenh movej toi {robot_ip}:{SECONDARY_PORT}")
        return True
    except Exception as e:
        print(f"[URScript] Loi gui URScript qua Port {SECONDARY_PORT}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="UR3e Safe Home Reset Utility")
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=os.environ.get("ROBOT_IP", "127.0.0.1"),
        help="Dia chi IP cua Robot UR3e hoac URSim (mac dinh: 127.0.0.1)"
    )
    parser.add_argument(
        "--accel", "-a",
        type=float,
        default=0.8,
        help="Gia toc khop [rad/s^2] (mac dinh: 0.8)"
    )
    parser.add_argument(
        "--speed", "-v",
        type=float,
        default=0.5,
        help="Van toc khop [rad/s] (mac dinh: 0.5)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=12.0,
        help="Thoi gian cho toi da [giay] (mac dinh: 12.0)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" DUC DONG BO VA DUA ROBOT VE SAFE HOME (MOVEJ 500HZ) ")
    print("=" * 60)
    print(f"Robot IP       : {args.robot_ip}")
    print(f"Muc tieu khop  : {SAFE_HOME_JOINTS}")
    print(f"Thong so van toc: v={args.speed} rad/s, a={args.accel} rad/s^2")
    print("-" * 60)

    # 1. Mo khoa Dashboard
    unlock_dashboard(args.robot_ip)

    # 2. Ket noi RTDE de giam sat khop
    rr = None
    try:
        import rtde_receive
        rr = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        q_start = rr.getActualQ()
        deg_start = [round(math.degrees(x), 1) for x in q_start]
        print(f"[RTDE] Goc khop ban dau (deg): {deg_start}")
        print(f"[RTDE] Goc khop ban dau (rad): {[round(x, 4) for x in q_start]}")

        dist_start = sum(abs(q_start[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
        if dist_start < 0.02:
            print("[Thong bao] Robot da o vi tri Safe Home chuan (do lech < 0.02 rad). Hoan tat.")
            if rr:
                rr.disconnect()
            return 0
    except ImportError:
        print("[Canh bao] Khong tim thay thu vien rtde_receive trong moi truong Python hien tai.")
        print("[Thong bao] Tiep tuc gui lenh movej truc tiep qua Port 30002...")
    except Exception as e:
        print(f"[Canh bao] Khong the ket noi RTDE Port 30004: {e}")

    # 3. Gui lenh movej qua Port 30002
    success = send_movej_script(args.robot_ip, SAFE_HOME_JOINTS, accel=args.accel, speed=args.speed)
    if not success:
        print("[Loi] Khong the gui lenh dieu khien ve Home.")
        if rr:
            rr.disconnect()
        return 1

    # 4. Cho va giam sat den khi robot ve dich
    print("[Tien trinh] Dang di chuyen ve Safe Home...")
    t_start = time.time()
    reached = False

    while time.time() - t_start < args.timeout:
        if rr is not None:
            try:
                q_curr = rr.getActualQ()
                err = sum(abs(q_curr[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
                if err < 0.025:
                    reached = True
                    break
            except Exception:
                pass
        time.sleep(0.1)

    elapsed = time.time() - t_start
    if rr is not None:
        try:
            q_final = rr.getActualQ()
            deg_final = [round(math.degrees(x), 1) for x in q_final]
            print(f"[RTDE] Goc khop sau khi chay (deg): {deg_final}")
            err_final = sum(abs(q_final[i] - SAFE_HOME_JOINTS[i]) for i in range(6))
            if err_final < 0.03:
                print(f"[Thanh cong] Robot da ve Safe Home chuan (sai so: {err_final:.4f} rad, thoi gian: {elapsed:.2f}s).")
            else:
                print(f"[Canh bao] Robot chua dat muc tieu sau {elapsed:.2f}s (sai so con lai: {err_final:.4f} rad).")
        except Exception as e:
            print(f"[RTDE] Loi doc khop cuoi: {e}")
        finally:
            rr.disconnect()
    else:
        # Neu khong co RTDE de do, cho 3 giay mac dinh
        time.sleep(2.5)
        print(f"[Thanh cong] Da phat lenh movej toi robot thanh cong (thoi gian: {elapsed:.2f}s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
