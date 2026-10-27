#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/visualize_data.py
Công cụ trực quan hóa (Data Visualization & Sanity Check) đa phương thức từ tệp HDF5:
- Top-Left: Đồ thị 2D động học 6 góc khớp (Joint Kinematics - qpos)
- Top-Right: Đồ thị 2D tín hiệu điều khiển ngàm kẹp (Gripper Actions - gripper_cmd & actual)
- Bottom-Left: Đồ thị 3D không gian quỹ đạo di chuyển của đầu búp kẹp (TCP Trajectory - tcp_pose)
- Bottom-Right: Khung hình camera màu RGB (Wrist Camera Vision) tại thời điểm giữa phiên ghi

Chế độ chỉ đọc (Read-only), an toàn 100% với tệp dữ liệu HDF5 gốc.
Tự động hỗ trợ môi trường không có màn hình (Headless/SSH) và xuất ảnh tĩnh PNG chất lượng cao.
"""

import os
import sys
import argparse
from typing import Optional, Tuple
import numpy as np

try:
    import h5py
except ImportError:
    print("[LỖI] Chưa cài đặt thư viện 'h5py'. Vui lòng chạy: pip install h5py", file=sys.stderr)
    sys.exit(1)

# Kiểm tra môi trường hiển thị trước khi import pyplot
has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

import matplotlib
if not has_display:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Danh xưng và màu sắc nhận diện cho 6 khớp Robot UR3e
JOINT_NAMES = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3",
]

JOINT_COLORS = [
    "#1f77b4",  # Xanh dương đậm
    "#ff7f0e",  # Cam
    "#2ca02c",  # Xanh lá
    "#d62728",  # Đỏ
    "#9467bd",  # Tím
    "#8c564b",  # Nâu
]


class HDF5Visualizer:
    """Bộ trực quan hóa và kiểm định chất lượng dữ liệu HDF5."""

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy tệp HDF5 tại: {self.file_path}")

    def load_data(self) -> dict:
        """Đọc an toàn các trường dữ liệu cần thiết từ tệp HDF5 ở chế độ chỉ đọc."""
        data = {}
        with h5py.File(self.file_path, "r") as f:
            # 1. Trích xuất Metadata
            data["attrs"] = dict(f.attrs)
            data["num_steps"] = int(f.attrs.get("num_steps", 0))
            data["hz"] = float(f.attrs.get("hz", 20.0))
            data["task_description"] = str(f.attrs.get("task_description", "N/A"))
            data["duration_sec"] = float(f.attrs.get("duration_sec", 0.0))
            data["success"] = bool(f.attrs.get("success", False))

            # 2. Đọc trục thời gian hoặc mốc thời gian
            if "observations/timestamps_ms" in f:
                ts_ms = np.array(f["observations/timestamps_ms"][:], dtype=np.float64)
                if len(ts_ms) > 0:
                    data["time_sec"] = (ts_ms - ts_ms[0]) / 1000.0
                else:
                    data["time_sec"] = np.array([])
            else:
                data["time_sec"] = np.arange(data["num_steps"]) / (data["hz"] if data["hz"] > 0 else 20.0)

            # Cập nhật số bước nếu metadata thiếu
            if data["num_steps"] == 0 and len(data["time_sec"]) > 0:
                data["num_steps"] = len(data["time_sec"])

            # 3. Góc khớp qpos [N, 6]
            if "observations/qpos" in f:
                data["qpos"] = np.array(f["observations/qpos"][:], dtype=np.float32)
            else:
                data["qpos"] = None

            # 4. Tín hiệu ngàm kẹp
            if "actions/gripper_cmd" in f:
                data["gripper_cmd"] = np.array(f["actions/gripper_cmd"][:], dtype=np.float32).flatten()
            elif "action" in f and f["action"].shape[-1] >= 7:
                # Cột cuối cùng của action 7-DoF là gripper
                data["gripper_cmd"] = np.array(f["action"][:, -1], dtype=np.float32).flatten()
            else:
                data["gripper_cmd"] = None

            if "observations/gripper" in f:
                data["gripper_actual"] = np.array(f["observations/gripper"][:], dtype=np.float32).flatten()
            else:
                data["gripper_actual"] = None

            # 5. Tọa độ đầu búp kẹp TCP [N, 6] (x, y, z, rx, ry, rz)
            if "observations/tcp_pose" in f:
                data["tcp_pose"] = np.array(f["observations/tcp_pose"][:], dtype=np.float32)
            else:
                data["tcp_pose"] = None

            # 6. Ảnh màu Camera cổ tay
            if "observations/images/wrist_camera" in f:
                data["has_wrist_camera"] = True
                data["camera_shape"] = f["observations/images/wrist_camera"].shape
            else:
                data["has_wrist_camera"] = False
                data["camera_shape"] = None

        return data

    def get_wrist_camera_frame(self, frame_idx: Optional[int] = None) -> Tuple[Optional[np.ndarray], int]:
        """Trích xuất 1 khung hình RGB duy nhất để tối ưu bộ nhớ RAM."""
        with h5py.File(self.file_path, "r") as f:
            if "observations/images/wrist_camera" not in f:
                return None, -1

            dset = f["observations/images/wrist_camera"]
            total_frames = dset.shape[0]
            if total_frames == 0:
                return None, -1

            if frame_idx is None or frame_idx < 0 or frame_idx >= total_frames:
                frame_idx = total_frames // 2

            frame = np.array(dset[frame_idx], dtype=np.uint8)
            # Chuẩn hóa nếu có kênh alpha RGBA -> RGB
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]

            return frame, frame_idx

    def plot(self, save_path: Optional[str] = None, show_gui: bool = True, frame_idx: Optional[int] = None) -> str:
        """Vẽ tổng hợp 4 biểu đồ trực quan hóa trên một Figure 2x2 chuyên nghiệp."""
        data = self.load_data()
        num_steps = data["num_steps"]
        time_arr = data["time_sec"] if len(data["time_sec"]) == num_steps else np.arange(num_steps)
        time_label = "Thời gian (giây)" if len(data["time_sec"]) == num_steps else "Bước (Steps)"

        # Tạo Figure 2x2
        fig = plt.figure(figsize=(15, 10), dpi=100)
        fig.patch.set_facecolor("#ffffff")

        # Tiêu đề tổng quát (Suptitle)
        ep_name = os.path.basename(self.file_path)
        task_str = data["task_description"]
        status_str = "Thành công (Success)" if data["success"] else "Chưa hoàn tất (Incomplete)"
        suptitle_text = (
            f"KIỂM ĐỊNH TỆP DỮ LIỆU: {ep_name}\n"
            f"Nhiệm vụ: {task_str} | Số bước: {num_steps} | Tần số: {data['hz']:.1f} Hz | Trạng thái: {status_str}"
        )
        fig.suptitle(suptitle_text, fontsize=13, fontweight="bold", y=0.98, color="#1a202c")

        # ---------------------------------------------------------
        # Subplot 1 (Top-Left): 2D Động học 6 góc khớp (Joint Kinematics)
        # ---------------------------------------------------------
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor("#f8fafc")
        if data["qpos"] is not None and data["qpos"].shape[0] > 0:
            for j_idx in range(min(6, data["qpos"].shape[1])):
                label = JOINT_NAMES[j_idx] if j_idx < len(JOINT_NAMES) else f"J{j_idx}"
                color = JOINT_COLORS[j_idx % len(JOINT_COLORS)]
                ax1.plot(time_arr, data["qpos"][:, j_idx], label=label, color=color, linewidth=1.8, alpha=0.9)
            ax1.set_title("1. Biến Thiên 6 Góc Khớp (Joint Kinematics)", fontsize=11, fontweight="bold", pad=8)
            ax1.set_xlabel(time_label, fontsize=9)
            ax1.set_ylabel("Góc khớp (radians)", fontsize=9)
            ax1.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
        else:
            ax1.text(0.5, 0.5, "Không có dữ liệu 'observations/qpos'", ha="center", va="center", color="#64748b")
            ax1.set_title("1. Biến Thiên 6 Góc Khớp (Trống)", fontsize=11, fontweight="bold")

        # ---------------------------------------------------------
        # Subplot 2 (Top-Right): 2D Tín hiệu đóng/mở ngàm kẹp (Gripper Actions)
        # ---------------------------------------------------------
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor("#f8fafc")
        has_cmd = data["gripper_cmd"] is not None and len(data["gripper_cmd"]) > 0
        has_act = data["gripper_actual"] is not None and len(data["gripper_actual"]) > 0

        if has_cmd or has_act:
            if has_cmd:
                t_cmd = time_arr if len(time_arr) == len(data["gripper_cmd"]) else np.arange(len(data["gripper_cmd"]))
                ax2.step(t_cmd, data["gripper_cmd"], where="post", label="Lệnh kẹp (gripper_cmd)", color="#d97706", linewidth=2.0)
            if has_act:
                t_act = time_arr if len(time_arr) == len(data["gripper_actual"]) else np.arange(len(data["gripper_actual"]))
                ax2.plot(t_act, data["gripper_actual"], label="Thực tế ngàm (gripper)", color="#059669", linestyle="--", linewidth=1.5, alpha=0.85)

            ax2.set_title("2. Tín Hiệu Điều Khiển Ngàm Kẹp (Gripper Actions)", fontsize=11, fontweight="bold", pad=8)
            ax2.set_xlabel(time_label, fontsize=9)
            ax2.set_ylabel("Giá trị ngàm (0.0 = Mở, 1.0 = Đóng)", fontsize=9)
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
            ax2.legend(loc="center right", fontsize=8, framealpha=0.9)
        else:
            ax2.text(0.5, 0.5, "Không có dữ liệu ngàm kẹp (gripper)", ha="center", va="center", color="#64748b")
            ax2.set_title("2. Tín Hiệu Ngàm Kẹp (Trống)", fontsize=11, fontweight="bold")

        # ---------------------------------------------------------
        # Subplot 3 (Bottom-Left): 3D Quỹ đạo đầu búp kẹp TCP (TCP Trajectory)
        # ---------------------------------------------------------
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        ax3.set_facecolor("#ffffff")
        if data["tcp_pose"] is not None and data["tcp_pose"].shape[0] > 0 and data["tcp_pose"].shape[1] >= 3:
            xs = data["tcp_pose"][:, 0]
            ys = data["tcp_pose"][:, 1]
            zs = data["tcp_pose"][:, 2]

            # Vẽ đường quỹ đạo
            ax3.plot(xs, ys, zs, color="#2563eb", linewidth=2.2, label="Quỹ đạo TCP", alpha=0.85)

            # Đánh dấu điểm bắt đầu (Start) và kết thúc (End)
            ax3.scatter(xs[0], ys[0], zs[0], color="#16a34a", s=90, marker="o", edgecolors="#0f172a", linewidth=1.5, label="Bắt đầu (Start)", zorder=5)
            ax3.scatter(xs[-1], ys[-1], zs[-1], color="#dc2626", s=110, marker="X", edgecolors="#0f172a", linewidth=1.5, label="Kết thúc (End)", zorder=5)

            ax3.set_title("3. Không Gian Quỹ Đạo TCP 3D (Base Frame)", fontsize=11, fontweight="bold", pad=8)
            ax3.set_xlabel("X (m)", fontsize=8, labelpad=4)
            ax3.set_ylabel("Y (m)", fontsize=8, labelpad=4)
            ax3.set_zlabel("Z (m)", fontsize=8, labelpad=4)
            ax3.tick_params(axis="both", which="major", labelsize=7)
            ax3.legend(loc="upper left", fontsize=8, framealpha=0.9)
        else:
            ax3.text(0.5, 0.5, 0.5, "Không có dữ liệu 'observations/tcp_pose'", ha="center", va="center", color="#64748b")
            ax3.set_title("3. Không Gian Quỹ Đạo TCP 3D (Trống)", fontsize=11, fontweight="bold")

        # ---------------------------------------------------------
        # Subplot 4 (Bottom-Right): Ảnh RGB Camera Cổ Tay (Wrist Camera Vision)
        # ---------------------------------------------------------
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor("#0f172a")
        img_frame, actual_idx = self.get_wrist_camera_frame(frame_idx)

        if img_frame is not None:
            ax4.imshow(img_frame)
            h, w = img_frame.shape[:2]
            ax4.set_title(f"4. Camera Cổ Tay (Wrist Camera - Frame {actual_idx}/{num_steps}, {w}x{h} RGB)", fontsize=11, fontweight="bold", pad=8)
            ax4.axis("off")
        else:
            ax4.set_facecolor("#f8fafc")
            ax4.text(0.5, 0.5, "Không có dữ liệu ảnh RGB 'observations/images/wrist_camera'", ha="center", va="center", color="#64748b")
            ax4.set_title("4. Thị Giác Camera (Trống)", fontsize=11, fontweight="bold")
            ax4.axis("off")

        # Căn chỉnh lề tự động
        plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

        # Xác định đường dẫn lưu ảnh PNG
        if save_path is None:
            base_dir = os.path.dirname(self.file_path)
            file_stem = os.path.splitext(os.path.basename(self.file_path))[0]
            save_path = os.path.join(base_dir, f"{file_stem}_visualization.png")

        # Lưu ảnh PNG độ nét cao
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"[Đã lưu ảnh trực quan] -> {save_path}")

        # Hiển thị GUI nếu khả dụng và được yêu cầu
        if show_gui and has_display:
            try:
                plt.show()
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể hiển thị cửa sổ Matplotlib GUI ({e}). Đã lưu file tĩnh tại {save_path}")
        else:
            plt.close(fig)

        return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Công cụ phân tích và trực quan hóa (Data Visualization & Sanity Check) đa phương thức từ tệp HDF5."
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        type=str,
        help="Đường dẫn đến tệp HDF5 cần kiểm tra (ví dụ: logs/ai_training_data/episode_0.hdf5)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh để lưu ảnh PNG kết quả (mặc định: cùng thư mục với file HDF5)"
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=None,
        help="Chỉ định chỉ số khung hình RGB muốn quan sát (mặc định: khung hình ở giữa tệp)"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Chỉ xuất và lưu tệp ảnh PNG mà không pop-up cửa sổ GUI tương tác"
    )

    args = parser.parse_args()

    try:
        visualizer = HDF5Visualizer(args.file)
        save_path = visualizer.plot(
            save_path=args.save,
            show_gui=not args.no_gui,
            frame_idx=args.frame_idx
        )
        print(f"[Thành công] Quá trình trực quan hóa tệp {args.file} hoàn tất!")
        print(f"            Tệp ảnh xuất: {save_path}")
    except Exception as e:
        print(f"[LỖI] Xử lý trực quan hóa thất bại: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
