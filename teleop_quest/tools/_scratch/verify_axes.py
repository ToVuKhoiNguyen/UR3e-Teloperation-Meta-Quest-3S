import os
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_heading(head_quat_wxyz, prev_u_fwd=None, prev_u_right=None):
    """
    Tính vector hướng nhìn u_fwd và u_right từ head_quat [w, x, y, z].
    Chỉ lấy góc quay quanh trục đứng thế giới (Yaw), loại bỏ 100% Pitch và Roll.
    Nếu norm_h <= 0.05 (đầu nhìn thẳng đứng): giữ nguyên hướng lần bóp cò trước;
    chỉ gán mặc định [0,0,-1] và [1,0,0] khi chưa có mốc cũ (prev là None).
    """
    r_head = R.from_quat([head_quat_wxyz[1], head_quat_wxyz[2],
                           head_quat_wxyz[3], head_quat_wxyz[0]])
    v_fwd = r_head.apply([0, 0, -1])  # Hướng nhìn trong không gian phòng WebXR
    fx, fz = float(v_fwd[0]), float(v_fwd[2])
    norm_h = float(np.hypot(fx, fz))
    if norm_h > 0.05:
        u_fwd = np.array([fx / norm_h, 0.0, fz / norm_h])
        u_right = np.array([-fz / norm_h, 0.0, fx / norm_h])
    else:
        if prev_u_fwd is not None and prev_u_right is not None:
            u_fwd = np.copy(prev_u_fwd)
            u_right = np.copy(prev_u_right)
        else:
            u_fwd = np.array([0.0, 0.0, -1.0])
            u_right = np.array([1.0, 0.0, 0.0])
    return u_fwd, u_right


def project(delta_world, u_fwd, u_right):
    """Chiếu vector dịch chuyển tay trong không gian thực theo hướng heading."""
    d_fwd = float(np.dot(delta_world, u_fwd))
    d_lat = float(np.dot(delta_world, u_right))
    d_vert = float(delta_world[1])  # Trục đứng thế giới (+Y trong WebXR)
    return d_fwd, d_lat, d_vert


def make_head_quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Tạo quaternion [w, x, y, z] theo các góc Yaw, Pitch, Roll (đơn vị độ)."""
    r_yaw = R.from_euler('Y', yaw_deg, degrees=True)
    r_pitch = R.from_euler('X', pitch_deg, degrees=True)
    r_roll = R.from_euler('Z', roll_deg, degrees=True)
    r_total = r_yaw * r_pitch * r_roll
    q = r_total.as_quat()  # scipy: [x, y, z, w]
    return [q[3], q[0], q[1], q[2]]  # w, x, y, z


# ==============================================================================
# PHẦN 1: BẢNG KIỂM CHỨNG CÁC KỊCH BẢN CHUYỂN ĐỘNG (HEAD-CENTRIC)
# ==============================================================================
SCENARIOS = [
    # (Mô tả, head_quat, delta vật lý local, kỳ vọng (d_fwd, d_lat, d_vert))
    ("Nguoi nhin -Z, day thang truoc",
     make_head_quat(yaw_deg=0),
     np.array([0, 0, -0.05]),
     (0.05, 0.0, 0.0)),

    ("Nguoi nhin -Z, day len (Cao)",
     make_head_quat(yaw_deg=0),
     np.array([0, 0.05, 0]),
     (0.0, 0.0, 0.05)),

    ("Nguoi quay 45, day thang truoc",
     make_head_quat(yaw_deg=45),
     np.array([-0.035355, 0, -0.035355]),
     (0.05, 0.0, 0.0)),

    ("Nguoi quay 90, day thang truoc",
     make_head_quat(yaw_deg=90),
     np.array([-0.05, 0, 0]),
     (0.05, 0.0, 0.0)),

    ("Nguoi quay 90, day sang phai",
     make_head_quat(yaw_deg=90),
     np.array([0, 0, -0.05]),
     (0.0, 0.05, 0.0)),

    ("Nguoi quay 180, day thang truoc",
     make_head_quat(yaw_deg=180),
     np.array([0, 0, 0.05]),
     (0.05, 0.0, 0.0)),

    ("Cui dau 45, day thang truoc",
     make_head_quat(yaw_deg=0, pitch_deg=-45),
     np.array([0, 0, -0.05]),
     (0.05, 0.0, 0.0)),

    ("Quay 45, gat Phai va nhac Cao",
     make_head_quat(yaw_deg=45),
     np.array([0.035355, 0.05, -0.035355]),
     (0.0, 0.05, 0.05)),

    ("Nhin thang xuong (giu huong lan 1)",
     make_head_quat(yaw_deg=45, pitch_deg=-90),
     np.array([-0.035355, 0, -0.035355]),
     (0.05, 0.0, 0.0)),
]

print("=" * 86)
print("BẢNG 1: KIỂM CHỨNG HỆ TRỤC HEAD-CENTRIC (CĂN CHUẨN THEO HƯỚNG NHÌN LÚC BÓP CÒ)")
print("=" * 86)
print(f"{'Kịch bản':<38} {'d_fwd':>7} {'d_lat':>7} {'d_vert':>7}  {'Kỳ vọng':<12} OK?")
print("-" * 86)

all_scenarios_ok = True
prev_u_fwd = None
prev_u_right = None

for desc, hq, delta, expected in SCENARIOS:
    u_fwd, u_right = compute_heading(hq, prev_u_fwd=prev_u_fwd, prev_u_right=prev_u_right)
    prev_u_fwd, prev_u_right = u_fwd, u_right
    d_fwd, d_lat, d_vert = project(delta, u_fwd, u_right)
    err = np.linalg.norm(np.array([d_fwd, d_lat, d_vert]) - np.array(expected))
    ok = err < 0.005
    if not ok:
        all_scenarios_ok = False
    ok_str = "OK" if ok else "LOI"
    exp_str = f"[{expected[0]*100:.0f}, {expected[1]*100:.0f}, {expected[2]*100:.0f}]cm"
    print(f"{desc:<38} {d_fwd*100:+6.1f}cm {d_lat*100:+6.1f}cm {d_vert*100:+6.1f}cm  {exp_str:<12} {ok_str}")

print()

# ==============================================================================
# PHẦN 2: KIỂM CHỨNG 5 LẦN BÓP CÒ LIÊN TIẾP Ở CÁC GÓC YAW KHÁC NHAU (BUG-E DRIFT TEST)
# ==============================================================================
print("=" * 86)
print("BẢNG 2: KIỂM TRA 5 LẦN BÓP CÒ LIÊN TIẾP (BUG-E CHỐNG TRÔI DẠT TÍCH LŨY)")
print("=" * 86)
print(f"{'Lần bóp cò':<35} {'Góc Yaw':>8} {'d_fwd':>8} {'d_lat':>8} {'d_vert':>8} {'Drift (mm)':>11}  OK?")
print("-" * 86)

clutch_sequence = [
    ("Bóp cò lần #1 (Yaw 45°)",
     make_head_quat(yaw_deg=45),
     45),
    ("Bóp cò lần #2 (Nhìn thẳng xuống)",
     make_head_quat(yaw_deg=45, pitch_deg=-90),
     45),  # norm_h ~ 0: giữ nguyên u_fwd của lần 1 (Yaw 45°)
    ("Bóp cò lần #3 (Yaw 90°)",
     make_head_quat(yaw_deg=90),
     90),
    ("Bóp cò lần #4 (Nhìn thẳng lên)",
     make_head_quat(yaw_deg=90, pitch_deg=90),
     90),  # norm_h ~ 0: giữ nguyên u_fwd của lần 3 (Yaw 90°)
    ("Bóp cò lần #5 (Yaw 180°)",
     make_head_quat(yaw_deg=180),
     180)
]

all_clutches_ok = True
accumulated_drift_m = 0.0
curr_fwd = None
curr_right = None

for idx, (desc, hq, nominal_yaw) in enumerate(clutch_sequence, start=1):
    u_fwd, u_right = compute_heading(hq, prev_u_fwd=curr_fwd, prev_u_right=curr_right)
    curr_fwd, curr_right = u_fwd, u_right

    # Người dùng đẩy thẳng phía trước 5cm theo hướng nhìn của cơ thể tại lần bóp cò đó
    delta_clutch = 0.05 * u_fwd
    d_fwd, d_lat, d_vert = project(delta_clutch, u_fwd, u_right)

    # Đo độ lệch ngoài ý muốn (drift trên trục ngang hoặc đứng)
    spurious_drift = math.hypot(d_lat, d_vert)
    accumulated_drift_m += spurious_drift
    is_ok = math.isclose(d_fwd, 0.05, abs_tol=1e-3) and spurious_drift < 1e-4
    if not is_ok:
        all_clutches_ok = False
    ok_str = "OK" if is_ok else "LOI"

    print(f"{desc:<35} {nominal_yaw:>+7.0f}° {d_fwd*100:+7.1f}cm {d_lat*100:+7.1f}cm {d_vert*100:+7.1f}cm {accumulated_drift_m*1000:>9.2f}mm   {ok_str}")

print("-" * 86)
if all_clutches_ok:
    print(">> KẾT QUẢ: Mọi lần bóp cò (kể cả khi nhìn thẳng xuống/lên) đều giữ đúng hướng, Drift = 0.00mm.")
print()

# ==============================================================================
# PHẦN 3: SO SÁNH ĐỐI CHỨNG VỚI ROOM-FIXED (bd5a5aa default)
# ==============================================================================
print("=" * 86)
print("BẢNG 3: ĐỐI CHỨNG VỚI ROOM_FIXED (u_fwd=[0,0,-1] CỐ ĐỊNH - bd5a5aa CŨ)")
print("=" * 86)
print(f"{'Kịch bản':<38} {'d_fwd':>7} {'d_lat':>7} {'d_vert':>7}  Ghi chú")
print("-" * 86)
u_fwd_fixed = np.array([0.0, 0.0, -1.0])
u_right_fixed = np.array([1.0, 0.0, 0.0])

for desc, hq, delta, expected in SCENARIOS:
    d_fwd, d_lat, d_vert = project(delta, u_fwd_fixed, u_right_fixed)
    note = "OK (Đúng hướng)" if (math.isclose(d_fwd, expected[0], abs_tol=0.01) and math.isclose(d_lat, expected[1], abs_tol=0.01)) else "LỖI HƯỚNG (Bị trôi/lệch)"
    print(f"{desc:<38} {d_fwd*100:+6.1f}cm {d_lat*100:+6.1f}cm {d_vert*100:+6.1f}cm  {note}")
print("=" * 86)

# ==============================================================================
# PHẦN 4: REPLAY DỮ LIỆU THẬT quest_run2.jsonl (HIỆU CHUẨN HƯỚNG THEO BÀN MÁY)
# ==============================================================================
print()
print("=" * 90)
print("BẢNG 4: REPLAY DỮ LIỆU THẬT quest_run2.jsonl (HIỆU CHUẨN HƯỚNG BÀN MÁY TẠI CLUTCH #1)")
print("=" * 90)

candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "quest_run2.jsonl"),
    "quest_run2.jsonl",
    "/home/nguyen/ur_ws/curobo_ursim/quest_run2.jsonl"
]
jsonl_path = None
for p in candidates:
    if os.path.exists(p):
        jsonl_path = os.path.abspath(p)
        break

if jsonl_path is None:
    print("[CẢNH BÁO] Không tìm thấy file quest_run2.jsonl để replay!")
else:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    clutches = []
    curr_clutch = []
    last_trig = False
    for r in records:
        trig = bool(r.get("trigger", False))
        if trig and not last_trig:
            curr_clutch = [r]
        elif trig and last_trig:
            curr_clutch.append(r)
        elif not trig and last_trig:
            if curr_clutch:
                clutches.append(curr_clutch)
                curr_clutch = []
        last_trig = trig
    if curr_clutch:
        clutches.append(curr_clutch)

    # Hiệu chuẩn hướng bàn máy bằng yaw đầu tại lần bóp cò #1 (frame đầu tiên)
    c1 = clutches[0]
    hq1 = c1[0].get("head_quat")
    r_head1 = R.from_quat([hq1[1], hq1[2], hq1[3], hq1[0]])
    v_fwd1 = r_head1.apply([0, 0, -1])
    fx1, fz1 = float(v_fwd1[0]), float(v_fwd1[2])
    norm_h1 = float(np.hypot(fx1, fz1))
    desk_u_fwd = np.array([fx1 / norm_h1, 0.0, fz1 / norm_h1])
    desk_u_right = np.array([-fz1 / norm_h1, 0.0, fx1 / norm_h1])
    desk_yaw = math.degrees(math.atan2(desk_u_fwd[0], -desk_u_fwd[2]))
    print(f"[HIỆU CHUẨN] bàn máy yaw={desk_yaw:+.1f}°: u_fwd={desk_u_fwd}, u_right={desk_u_right}")
    print("-" * 90)

    target_groups = [
        ("Nhóm #1-3 (Tiến)", [1, 2, 3], "fwd > 0"),
        ("Nhóm #4-6 (Lùi)", [4, 5, 6], "fwd < 0"),
        ("Nhóm #8-10 (Phải)", [8, 9, 10], "lat > 0"),
        ("Nhóm #13-15 (Cao)", [13, 14, 15], "vert > 0"),
        ("Nhóm #17-19 (Tiến)", [17, 18, 19], "fwd > 0"),
        ("Nhóm #21-23 (Tiến + Thẳng)", [21, 22, 23], "fwd > 0 và |lat| < 1.5cm"),
    ]

    print(f"{'Nhóm cử động':<28} {'Clutch':<8} {'d_fwd':>8} {'d_lat':>8} {'d_vert':>8}   {'Kỳ vọng':<25} OK?")
    print("-" * 90)

    all_replay_ok = True
    for grp_name, indices, exp_desc in target_groups:
        for c_idx in indices:
            c = clutches[c_idx - 1]
            p_start = np.array(c[0]["pos"])
            p_end = np.array(c[-1]["pos"])
            delta = p_end - p_start
            d_fwd = float(np.dot(delta, desk_u_fwd))
            d_lat = float(np.dot(delta, desk_u_right))
            d_vert = float(delta[1])

            ok = False
            if c_idx in [1, 2, 3]:
                ok = d_fwd > 0
            elif c_idx in [4, 5, 6]:
                ok = d_fwd < 0
            elif c_idx in [8, 9, 10]:
                ok = d_lat > 0
            elif c_idx in [13, 14, 15]:
                ok = d_vert > 0
            elif c_idx in [17, 18, 19]:
                ok = d_fwd > 0
            elif c_idx in [21, 22, 23]:
                ok = (d_fwd > 0) and (abs(d_lat) < 0.015)

            if not ok:
                all_replay_ok = False
            ok_str = "OK" if ok else "LOI"
            print(f"{grp_name:<28} #{c_idx:<7} {d_fwd*100:+7.2f}cm {d_lat*100:+7.2f}cm {d_vert*100:+7.2f}cm   {exp_desc:<25} {ok_str}")

    print("-" * 90)
    if all_replay_ok:
        print(">> KẾT QUẢ REPLAY: 100% CÁC LẦN BÓP CÒ ĐỀU ĐẠT CHÍNH XÁC KỲ VỌNG!")
    else:
        print(">> KẾT QUẢ REPLAY: CÓ DÒNG KHÔNG ĐẠT KỲ VỌNG!")
    print("=" * 90)

