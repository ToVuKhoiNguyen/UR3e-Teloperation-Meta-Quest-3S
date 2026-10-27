#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/robot_finder.py
Công cụ tự động dò tìm địa chỉ IP của Robot Universal Robots UR3e trên mạng LAN:
- Hỗ trợ chế độ DHCP: Tự động quét subnet và bắt IP của robot ngay khi nhận DHCP.
- Quét nhanh đa luồng (Multi-threaded scan < 0.5s) trên cổng Dashboard (29999) và RTDE (30004).
- Tự động nhận diện OUI MAC address (00:30:d6 của Beckhoff / Universal Robots).
- Cơ chế Polling chờ robot cấp phát DHCP với thanh tiến trình trực quan.
"""

import os
import sys
import time
import socket
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple


UR_MAC_PREFIXES = ("00:30:d6",)  # Beckhoff Automation / Universal Robots OUI


def is_ur_port_open(ip: str, port: int = 29999, timeout: float = 0.15) -> bool:
    """Kiểm tra cổng Dashboard (29999) hoặc RTDE (30004) có mở và phản hồi không."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        res = s.connect_ex((ip, port))
        if res == 0:
            if port == 29999:
                try:
                    banner = s.recv(256).decode(errors="ignore")
                    s.close()
                    return "Universal Robots" in banner or len(banner) > 0
                except Exception:
                    s.close()
                    return True
            s.close()
            return True
        s.close()
    except Exception:
        pass
    return False


def get_ethernet_subnets() -> List[Tuple[str, ipaddress.IPv4Network]]:
    """Lấy danh sách các dải mạng IPv4 trên các card mạng Ethernet vật lý."""
    subnets = []
    try:
        out = subprocess.check_output(["ip", "-4", "addr", "show"], timeout=2.0).decode()
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("inet ") and "127.0.0.1" not in line:
                parts = line.split()
                cidr = parts[1]
                dev = parts[-1]
                # Bỏ qua loopback và các card ảo của docker
                if dev.startswith(("lo", "docker", "br-", "veth")):
                    continue
                try:
                    net = ipaddress.IPv4Network(cidr, strict=False)
                    if net.num_addresses <= 1024:  # Giới hạn subnet /24 hoặc /22
                        subnets.append((dev, net))
                except Exception:
                    pass
    except Exception:
        pass

    # Nếu không tìm thấy, bổ sung subnet mặc định 10.42.0.0/24
    if not subnets:
        subnets.append(("enp109s0", ipaddress.IPv4Network("10.42.0.0/24", strict=False)))
    return subnets


def check_arp_table() -> Optional[str]:
    """Kiểm tra nhanh bảng ARP hệ thống (/proc/net/arp) xem có MAC của UR không."""
    try:
        with open("/proc/net/arp", "r") as f:
            lines = f.readlines()[1:]
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:
                ip, mac, dev = parts[0], parts[3].lower(), parts[5]
                if not dev.startswith(("lo", "docker", "br-", "veth")):
                    for pfx in UR_MAC_PREFIXES:
                        if mac.startswith(pfx):
                            if is_ur_port_open(ip, 29999, 0.2) or is_ur_port_open(ip, 30004, 0.2):
                                return ip
    except Exception:
        pass
    return None


def get_host_ips() -> set:
    """Lấy toàn bộ các IP cục bộ của chính máy tính điều khiển (để không nhận nhầm chính mình/Docker URSim)."""
    host_ips = {"127.0.0.1", "localhost", "0.0.0.0"}
    try:
        out = subprocess.check_output(["ip", "-4", "addr", "show"], timeout=2.0).decode()
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                ip_cidr = line.split()[1]
                ip_only = ip_cidr.split("/")[0]
                host_ips.add(ip_only)
    except Exception:
        pass
    return host_ips


def scan_subnet_for_ur(network: ipaddress.IPv4Network, max_workers: int = 64) -> Optional[str]:
    """Quét song song toàn bộ các IP trong subnet tìm cổng Dashboard 29999 hoặc RTDE 30004 (bỏ qua IP của chính PC)."""
    host_ips = get_host_ips()
    hosts = [str(h) for h in network.hosts() if str(h) not in host_ips]

    def probe(ip):
        if is_ur_port_open(ip, 29999, 0.12):
            return ip
        if is_ur_port_open(ip, 30004, 0.12):
            return ip
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = ex.map(probe, hosts)
        for res in results:
            if res:
                return res
    return None


def discover_robot_ip(
    preferred_ip: Optional[str] = None,
    timeout: float = 12.0,
    verbose: bool = True
) -> Optional[str]:
    """
    Tự động dò tìm địa chỉ IP của Robot UR3e trên mạng:
    1. Kiểm tra preferred_ip (nếu có và khác 'auto').
    2. Kiểm tra ARP table (/proc/net/arp).
    3. Quét subnet card mạng Ethernet.
    4. Nếu chưa thấy, lặp lại quét định kỳ (polling) chờ robot nhận DHCP.
    """
    # 1. Kiểm tra preferred_ip nếu được chỉ định
    if preferred_ip and preferred_ip.lower() not in ("auto", "none", "", "127.0.0.1", "localhost"):
        if is_ur_port_open(preferred_ip, 29999, 0.3) or is_ur_port_open(preferred_ip, 30004, 0.3):
            if verbose:
                print(f"[Auto-Discovery] Đã kết nối ngay tới Robot tại IP ưu tiên: \033[92m{preferred_ip}\033[0m")
            return preferred_ip

    if verbose:
        print(f"[Auto-Discovery] Bắt đầu tự động dò tìm Robot UR3e trên mạng LAN (Timeout: {timeout:.0f}s)...")

    start_time = time.time()
    subnets = get_ethernet_subnets()
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1

        # 2. Thử kiểm tra bảng ARP
        arp_ip = check_arp_table()
        if arp_ip:
            if verbose:
                print(f"\n\033[92;1m[Auto-Discovery] >>> TÌM THẤY ROBOT UR3E QUA BẢNG ARP: {arp_ip}\033[0m")
            return arp_ip

        # 3. Quét các subnet Ethernet
        for dev, net in subnets:
            found_ip = scan_subnet_for_ur(net)
            if found_ip:
                if verbose:
                    print(f"\n\033[92;1m[Auto-Discovery] >>> TÌM THẤY ROBOT UR3E TRÊN {dev} ({net}): {found_ip}\033[0m")
                return found_ip

        # 4. Chờ lượt quét tiếp theo (đợi DHCP cấp phát)
        elapsed = time.time() - start_time
        remain = max(0, int(timeout - elapsed))
        if verbose:
            print(f"  * Đang quét mạng tìm robot (Đợi DHCP cấp phát... còn {remain}s) ", end="\r", flush=True)
        time.sleep(1.0)

    if verbose:
        print(f"\n\033[93m[Auto-Discovery] Hết thời gian chờ ({timeout:.0f}s). Chưa phát hiện Robot UR3e trên các card mạng.\033[0m")
        print("  -> Vui lòng kiểm tra cáp mạng cắm vào robot, đảm bảo Teach Pendant đang ở màn hình Settings -> System -> Network và bấm Apply.")
    return None


if __name__ == "__main__":
    ip_arg = sys.argv[1] if len(sys.argv) > 1 else None
    found = discover_robot_ip(preferred_ip=ip_arg, timeout=10.0, verbose=True)
    if found:
        print(f"\n=> KẾT QUẢ: Robot IP = {found}")
        sys.exit(0)
    else:
        print("\n=> KẾT QUẢ: Không tìm thấy robot.")
        sys.exit(1)
