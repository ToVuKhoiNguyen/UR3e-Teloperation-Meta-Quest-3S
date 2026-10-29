import os
import re
import socket
import subprocess

def get_all_local_ips():
    """
    Lấy danh sách tất cả địa chỉ IP mạng IPv4 vật lý trên máy (Wi-Fi, Ethernet),
    loại trừ loopback và docker bridge.
    """
    ips = []
    try:
        output = subprocess.check_output(["ip", "-4", "addr", "show"], stderr=subprocess.DEVNULL, timeout=1.0).decode()
        for line in output.splitlines():
            m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/\d+.*scope global\s+(.*)", line)
            if m:
                ip, iface = m.group(1), m.group(2).strip()
                if not any(iface.startswith(prefix) for prefix in ("docker", "br-", "veth", "lo")):
                    if ip not in ips:
                        ips.append(ip)
    except Exception:
        pass
    return ips

def get_local_ip():
    """
    Tự động dò tìm địa chỉ IP mạng LAN thật của máy (ưu tiên Wi-Fi để Meta Quest kết nối, sau đó Ethernet),
    hoạt động 100% tin cậy ngay cả khi máy HOÀN TOÀN OFFLINE.
    """
    try:
        output = subprocess.check_output(["ip", "-4", "addr", "show"], stderr=subprocess.DEVNULL, timeout=1.0).decode()
        candidates = []
        for line in output.splitlines():
            m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/\d+.*scope global\s+(.*)", line)
            if m:
                ip, iface = m.group(1), m.group(2).strip()
                if not any(iface.startswith(prefix) for prefix in ("docker", "br-", "veth", "lo")):
                    candidates.append((iface, ip))
        if candidates:
            # Ưu tiên card Wi-Fi (wl*) để Quest kết nối không dây, tránh nhầm sang subnet robot 10.42.0.x
            for iface, ip in candidates:
                if iface.startswith("wl") and not ip.startswith("10.42.0."):
                    return ip
            # Sau đó đến card Ethernet không phải robot subnet
            for iface, ip in candidates:
                if any(iface.startswith(p) for p in ("en", "eth")) and not ip.startswith("10.42.0."):
                    return ip
            # Cuối cùng là bất kỳ IP nào
            return candidates[0][1]
    except Exception:
        pass

    # 2. Dự phòng bằng kết nối UDP socket không chặn
    for target in [("8.8.8.8", 80), ("1.1.1.1", 80), ("192.168.255.255", 1), ("10.255.255.255", 1), ("172.31.255.255", 1)]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.2)
            s.connect(target)
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127.") and not ip.startswith("172.17."):
                return ip
        except Exception:
            pass

    return "127.0.0.1"

def setup_ssl(base_dir, local_ip):
    import ssl
    certs_dir = os.path.join(base_dir, "certs")
    os.makedirs(certs_dir, exist_ok=True)
    try:
        os.chmod(certs_dir, 0o777)
    except Exception:
        pass

    cert_path = os.path.join(certs_dir, "cert.pem")
    key_path = os.path.join(certs_dir, "key.pem")
    ip_cache_path = os.path.join(certs_dir, "ip_cache.txt")

    cached_ip = ""
    if os.path.exists(ip_cache_path):
        try:
            with open(ip_cache_path, "r") as f:
                cached_ip = f.read().strip()
        except Exception:
            pass

    # Kiểm tra tính hợp lệ thực tế của cert bằng cách nạp thử vào ssl.SSLContext
    def _is_valid(c, k):
        if not (os.path.exists(c) and os.path.exists(k)):
            return False
        if os.path.getsize(c) < 100 or os.path.getsize(k) < 100:
            return False
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(c, k)
            return True
        except Exception:
            return False

    cert_valid = _is_valid(cert_path, key_path)

    if not cert_valid or (cached_ip and cached_ip != local_ip):
        all_ips = get_all_local_ips()
        if local_ip not in all_ips and local_ip not in ("127.0.0.1", "localhost"):
            all_ips.append(local_ip)

        san_entries = ["DNS:localhost", "IP:127.0.0.1"]
        for ip in all_ips:
            if ip not in ("127.0.0.1", "localhost"):
                san_entries.append(f"IP:{ip}")
        san_val = f"subjectAltName={','.join(san_entries)}"

        print(f"[SSL] Thiết lập chứng chỉ SSL đa SAN ({', '.join(san_entries)})...")
        for f in (cert_path, key_path, ip_cache_path):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        cmd = [
            "openssl", "req", "-x509", "-nodes", "-days", "365",
            "-newkey", "rsa:2048",
            "-keyout", key_path,
            "-out", cert_path,
            "-subj", f"/CN={local_ip}",
            "-addext", san_val
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            # Fallback nếu openssl phiên bản cũ không hỗ trợ -addext
            subprocess.run(
                f"openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout {key_path} -out {cert_path} -subj '/CN={local_ip}'",
                shell=True, stderr=subprocess.DEVNULL
            )
        try:
            os.chmod(cert_path, 0o666)
            os.chmod(key_path, 0o666)
        except Exception:
            pass
        try:
            with open(ip_cache_path, "w") as f:
                f.write(local_ip)
            os.chmod(ip_cache_path, 0o666)
        except Exception:
            pass

    return cert_path, key_path

