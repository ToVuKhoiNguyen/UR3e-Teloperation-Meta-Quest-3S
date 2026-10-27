#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/extract_hdf5.py
Công cụ độc lập giải nén và trích xuất toàn bộ dữ liệu từ tệp HDF5:
- Video MP4 cho luồng ảnh màu (RGB)
- Chuỗi ảnh PNG 16-bit cho luồng độ sâu (Depth)
- Tệp CSV riêng biệt với header ngữ nghĩa cho toàn bộ dữ liệu số học (Kinematics, Dynamics, Wrench, Actions, Teleop, Timestamps)
- Tệp JSON và TXT tóm tắt toàn bộ Metadata và thông số phiên ghi
"""

import os
import sys
import argparse
import json
import csv
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import h5py
except ImportError:
    print("[LỖI] Chưa cài đặt thư viện 'h5py'. Vui lòng chạy: pip install h5py", file=sys.stderr)
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None
    print("[CẢNH BÁO] Không tìm thấy 'cv2' (opencv-python). Tính năng xử lý ảnh fallback sẽ bị tắt.", file=sys.stderr)

try:
    import imageio
except ImportError:
    imageio = None


# Bảng ánh xạ tiêu đề cột (Semantic Headers) theo cấu trúc vật lý của Robot UR3e và Viễn thao VR
SEMANTIC_HEADERS: Dict[str, List[str]] = {
    # 6 DoF Khớp Robot (Góc, Vận tốc, Dòng điện, Mô-men, Điện áp, Nhiệt độ)
    "qpos": ["base_rad", "shoulder_rad", "elbow_rad", "wrist_1_rad", "wrist_2_rad", "wrist_3_rad"],
    "qvel": ["base_rads", "shoulder_rads", "elbow_rads", "wrist_1_rads", "wrist_2_rads", "wrist_3_rads"],
    "target_q": ["target_base_rad", "target_shoulder_rad", "target_elbow_rad", "target_wrist_1_rad", "target_wrist_2_rad", "target_wrist_3_rad"],
    "target_qd": ["target_base_rads", "target_shoulder_rads", "target_elbow_rads", "target_wrist_1_rads", "target_wrist_2_rads", "target_wrist_3_rads"],
    "target_qdd": ["target_base_rads2", "target_shoulder_rads2", "target_elbow_rads2", "target_wrist_1_rads2", "target_wrist_2_rads2", "target_wrist_3_rads2"],
    "currents": ["base_A", "shoulder_A", "elbow_A", "wrist_1_A", "wrist_2_A", "wrist_3_A"],
    "target_currents": ["target_base_A", "target_shoulder_A", "target_elbow_A", "target_wrist_1_A", "target_wrist_2_A", "target_wrist_3_A"],
    "target_moments": ["target_base_Nm", "target_shoulder_Nm", "target_elbow_Nm", "target_wrist_1_Nm", "target_wrist_2_Nm", "target_wrist_3_Nm"],
    "joint_voltages": ["base_V", "shoulder_V", "elbow_V", "wrist_1_V", "wrist_2_V", "wrist_3_V"],
    "temperatures": ["base_degC", "shoulder_degC", "elbow_degC", "wrist_1_degC", "wrist_2_degC", "wrist_3_degC"],

    # 6D Descartes TCP Pose & Velocity
    "tcp_pose": ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"],
    "target_tcp_pose": ["target_x_m", "target_y_m", "target_z_m", "target_rx_rad", "target_ry_rad", "target_rz_rad"],
    "tcp_speed": ["vx_ms", "vy_ms", "vz_ms", "wx_rads", "wy_rads", "wz_rads"],

    # 6D Lực & Mô-men tiếp xúc (Wrench F/T)
    "tcp_force": ["fx_N", "fy_N", "fz_N", "tx_Nm", "ty_Nm", "tz_Nm"],
    "ft_raw_wrench": ["raw_fx_N", "raw_fy_N", "raw_fz_N", "raw_tx_Nm", "raw_ty_Nm", "raw_tz_Nm"],

    # Viễn thao VR & Hành động can thiệp (Teleoperation & AI Datasets)
    "vr_controller_pose": ["pos_x_m", "pos_y_m", "pos_z_m", "quat_x", "quat_y", "quat_z", "quat_w"],
    "action": ["target_q1_rad", "target_q2_rad", "target_q3_rad", "target_q4_rad", "target_q5_rad", "target_q6_rad", "gripper_cmd"],
    "gripper_cmd": ["gripper_cmd_norm"],
    "gripper": ["gripper_actual_norm"],

    # Tải trọng & Cảm biến gia tốc đầu công cụ
    "payload": ["mass_kg", "cog_x_m", "cog_y_m", "cog_z_m"],
    "tool_accelerometer": ["ax_ms2", "ay_ms2", "az_ms2"],

    # An toàn & Giám sát nguồn điện
    "safety_flags": ["safety_status_bits", "safety_status_word"],
    "robot_mode": ["robot_mode_code"],
    "safety_mode": ["safety_mode_code"],
    "speed_scaling": ["speed_scaling_factor"],
    "robot_current": ["robot_current_A"],
    "robot_voltage": ["robot_voltage_V"],

    # Timestamps đồng bộ
    "timestamps_ms": ["timestamp_ms"],
    "timestamps_ns": ["timestamp_ns"],
    "timestamps_robot": ["timestamp_robot_sec"],
}


class HDF5Extractor:
    """
    Trích xuất toàn diện dữ liệu HDF5 từ các phiên ghi Teleoperation và AI Training.
    """
    def __init__(
        self,
        filepath: str,
        output_dir: Optional[str] = None,
        fps_override: Optional[float] = None,
        verbose: bool = True
    ):
        self.filepath = os.path.abspath(filepath)
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Không tìm thấy tệp HDF5 tại: {self.filepath}")

        # Tự động xác định thư mục đầu ra nếu không chỉ định
        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
        else:
            base_name = os.path.splitext(os.path.basename(self.filepath))[0]
            parent_dir = os.path.dirname(self.filepath)
            self.output_dir = os.path.join(parent_dir, f"{base_name}_extracted")

        self.fps_override = fps_override
        self.verbose = verbose

        self.metadata: Dict[str, Any] = {}
        self.extracted_files: List[str] = []
        self.used_csv_names: Dict[str, str] = {}  # base_name -> full_path

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def extract_all(self) -> str:
        """
        Thực hiện toàn bộ quy trình duyệt cây đệ quy và trích xuất dữ liệu.
        """
        start_time = time.time()
        self._log(f"\n=======================================================")
        self._log(f" BẮT ĐẦU TRÍCH XUẤT DỮ LIỆU TỪ HDF5")
        self._log(f" Nguồn: {self.filepath}")
        self._log(f" Đích : {self.output_dir}")
        self._log(f"=======================================================")

        os.makedirs(self.output_dir, exist_ok=True)

        with h5py.File(self.filepath, "r") as h5_file:
            # 1. Trích xuất Root Metadata & Attributes
            self._extract_metadata(h5_file)

            # Xác định FPS mặc định (ưu tiên fps_override -> attrs['hz'] -> 20.0)
            fps = self.fps_override or float(self.metadata.get("hz", 20.0))
            if fps <= 0:
                fps = 20.0

            # 2. Duyệt đệ quy toàn bộ Dataset trong tệp
            datasets_to_process: List[Tuple[str, h5py.Dataset]] = []

            def visitor(name: str, obj: Any):
                if isinstance(obj, h5py.Dataset):
                    datasets_to_process.append((name, obj))

            h5_file.visititems(visitor)
            self._log(f"\n[Quét cấu trúc] Tìm thấy {len(datasets_to_process)} dataset trong tệp.")

            # 3. Phân loại và xử lý từng Dataset
            for dset_path, dset_obj in datasets_to_process:
                self._dispatch_dataset(dset_path, dset_obj, fps)

            # 4. Ghi báo cáo tóm tắt (summary.txt)
            self._write_summary(time.time() - start_time)

        self._log(f"\n>>> HOÀN TẤT TRÍCH XUẤT DỮ LIỆU TRONG {time.time() - start_time:.2f}s!")
        self._log(f">>> Thư mục kết quả: {self.output_dir}\n")
        return self.output_dir

    def _extract_metadata(self, h5_file: h5py.File):
        """Trích xuất toàn bộ Root Attributes sang file episode_metadata.json."""
        self.metadata = {}
        for k, v in h5_file.attrs.items():
            if isinstance(v, bytes):
                self.metadata[k] = v.decode("utf-8", errors="replace")
            elif isinstance(v, np.ndarray):
                self.metadata[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating, np.bool_)):
                self.metadata[k] = v.item()
            else:
                self.metadata[k] = v

        meta_path = os.path.join(self.output_dir, "episode_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        self.extracted_files.append(meta_path)
        self._log(f"[Metadata] Đã lưu thông số phiên ghi -> episode_metadata.json")

    def _dispatch_dataset(self, dset_path: str, dset: h5py.Dataset, fps: float):
        """Phân loại dataset theo số chiều (ndim) và kiểu dữ liệu (dtype)."""
        shape = dset.shape
        dtype = dset.dtype

        # Kiểm tra nếu dataset rỗng
        if len(shape) == 0 or shape[0] == 0:
            self._log(f"[Bỏ qua] Dataset rỗng: {dset_path} (shape={shape})")
            return

        # A. Nhóm Video RGB: Mảng 4D [T, H, W, 3] hoặc [T, H, W, 4], dtype=uint8
        if len(shape) == 4 and shape[-1] in (3, 4) and dtype == np.uint8:
            self._export_rgb_video(dset_path, dset, fps)

        # B. Nhóm Ảnh Độ Sâu (Depth): Mảng 3D [T, H, W] (hoặc tên có chữ 'depth')
        elif len(shape) == 3 and ("depth" in dset_path.lower() or dtype in (np.uint16, np.float32)):
            self._export_depth_sequence(dset_path, dset)

        # C. Nhóm Dữ liệu Số học: 1D hoặc 2D [T, D]
        elif len(shape) in (1, 2) and (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            self._export_numerical_csv(dset_path, dset)

        else:
            self._log(f"[Không hỗ trợ] Bỏ qua {dset_path}: shape={shape}, dtype={dtype}")

    def _export_rgb_video(self, dset_path: str, dset: h5py.Dataset, fps: float):
        """Xuất chuỗi frame RGB [T, H, W, 3] thành file video MP4 chất lượng cao."""
        if cv2 is None and imageio is None:
            self._log(f"[Bỏ qua Video] Thiếu imageio và OpenCV, không thể xuất {dset_path}")
            return

        cam_name = os.path.basename(dset_path)
        out_filename = f"{cam_name}.mp4"
        out_path = os.path.join(self.output_dir, out_filename)

        num_frames, h, w, c = dset.shape
        self._log(f"[RGB Video] Đang xuất {dset_path} -> {out_filename} ({num_frames} frames, {w}x{h} @ {fps:.1f} FPS)...")

        # Ưu tiên 1: Dùng imageio với libx264 để video tương thích 100% với trình phát web/VS Code/Antigravity
        if imageio is not None:
            try:
                with imageio.get_writer(
                    out_path,
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=None,
                    ffmpeg_params=["-movflags", "+faststart"],
                ) as writer:
                    chunk_size = 64
                    for start_idx in range(0, num_frames, chunk_size):
                        end_idx = min(start_idx + chunk_size, num_frames)
                        frames_chunk = dset[start_idx:end_idx]
                        for frame in frames_chunk:
                            frame_rgb = frame[..., :3] if c == 4 else frame
                            writer.append_data(frame_rgb)
                self.extracted_files.append(out_path)
                self._log(f"          -> Đã xuất video H.264 hoàn tất: {out_filename}")
                return
            except Exception as e:
                self._log(f"[CẢNH BÁO] Không thể xuất bằng imageio ({e}), chuyển sang OpenCV VideoWriter...")

        # Ưu tiên 2: Fallback sang OpenCV VideoWriter
        if cv2 is not None:
            writer = None
            for codec in ["avc1", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                if writer.isOpened():
                    break

            if writer is None or not writer.isOpened():
                self._log(f"[LỖI] Không thể khởi tạo VideoWriter cho {out_path}")
                return

            try:
                chunk_size = 64
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    frames_chunk = dset[start_idx:end_idx]
                    for frame in frames_chunk:
                        if c == 4:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        else:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame_bgr)
            finally:
                writer.release()

            self.extracted_files.append(out_path)
            self._log(f"          -> Đã xuất video hoàn tất: {out_filename}")

    def _export_depth_sequence(self, dset_path: str, dset: h5py.Dataset):
        """Xuất chuỗi frame Depth [T, H, W] thành thư mục chứa các ảnh PNG 16-bit bảo toàn độ chính xác mm."""
        cam_name = os.path.basename(dset_path)
        folder_name = f"{cam_name}_depth" if not cam_name.endswith("_depth") else cam_name
        depth_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(depth_dir, exist_ok=True)

        num_frames, h, w = dset.shape
        self._log(f"[Depth Image] Đang xuất {dset_path} -> {folder_name}/ ({num_frames} frames, {w}x{h}, dtype={dset.dtype})...")

        min_val, max_val = float("inf"), float("-inf")
        sum_val = 0.0
        total_pixels = 0

        # Đọc theo mẻ nhỏ và ghi ảnh PNG 16-bit
        chunk_size = 64
        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            frames_chunk = dset[start_idx:end_idx]

            for idx, frame in enumerate(frames_chunk, start=start_idx):
                frame_filename = f"frame_{idx:05d}.png"
                frame_path = os.path.join(depth_dir, frame_filename)

                # Thu thập thống kê khoảng cách
                valid_mask = frame > 0
                if np.any(valid_mask):
                    f_min = float(np.min(frame[valid_mask]))
                    f_max = float(np.max(frame[valid_mask]))
                    min_val = min(min_val, f_min)
                    max_val = max(max_val, f_max)
                    sum_val += float(np.sum(frame[valid_mask]))
                    total_pixels += int(np.sum(valid_mask))

                if cv2 is not None:
                    if dset.dtype == np.uint16:
                        cv2.imwrite(frame_path, frame)
                    elif np.issubdtype(dset.dtype, np.floating):
                        # Nếu là float mét, chuyển sang uint16 mm
                        frame_uint16 = (frame * 1000.0).clip(0, 65535).astype(np.uint16)
                        cv2.imwrite(frame_path, frame_uint16)
                    else:
                        cv2.imwrite(frame_path, frame.astype(np.uint16))

        # Lưu metadata thông số độ sâu
        info_path = os.path.join(depth_dir, "depth_info.json")
        mean_val = (sum_val / total_pixels) if total_pixels > 0 else 0.0
        depth_info = {
            "dataset_path": dset_path,
            "num_frames": num_frames,
            "resolution": [w, h],
            "dtype": str(dset.dtype),
            "unit": "millimeters (mm)",
            "min_depth_mm": round(min_val, 2) if min_val != float("inf") else 0.0,
            "max_depth_mm": round(max_val, 2) if max_val != float("-inf") else 0.0,
            "mean_depth_mm": round(mean_val, 2),
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(depth_info, f, indent=2)

        self.extracted_files.append(depth_dir)
        self._log(f"          -> Đã xuất {num_frames} frame depth 16-bit vào thư mục: {folder_name}/")

    def _export_numerical_csv(self, dset_path: str, dset: h5py.Dataset):
        """Xuất dataset số học 1D hoặc 2D sang tệp CSV với tiêu đề ngữ nghĩa chuẩn xác."""
        shape = dset.shape
        num_rows = shape[0]
        num_cols = shape[1] if len(shape) == 2 else 1

        # Xác định tên file CSV ngắn gọn nhưng tránh xung đột (name collision)
        raw_name = os.path.basename(dset_path)
        csv_filename = f"{raw_name}.csv"
        if csv_filename in self.used_csv_names and self.used_csv_names[csv_filename] != dset_path:
            # Xung đột tên (ví dụ observations/gripper vs actions/gripper) -> dùng đường dẫn gạch dưới
            safe_prefix = dset_path.replace("/", "_")
            csv_filename = f"{safe_prefix}.csv"

        self.used_csv_names[csv_filename] = dset_path
        csv_path = os.path.join(self.output_dir, csv_filename)

        # 1. Xác định Header các cột
        headers = ["step"]
        semantic = SEMANTIC_HEADERS.get(raw_name)
        if semantic and len(semantic) == num_cols:
            headers.extend(semantic)
        elif num_cols == 1:
            headers.append(raw_name)
        else:
            headers.extend([f"{raw_name}_{i}" for i in range(num_cols)])

        # 2. Đọc toàn bộ dữ liệu (Số học dung lượng rất nhẹ, nạp thẳng mảng)
        data = dset[:]
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # 3. Ghi file CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for step_idx in range(num_rows):
                row_vals = data[step_idx]
                row_formatted = [step_idx]
                for v in row_vals:
                    if isinstance(v, (np.floating, float)):
                        row_formatted.append(f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.6e}")
                    elif isinstance(v, (np.bool_, bool)):
                        row_formatted.append("1" if v else "0")
                    else:
                        row_formatted.append(str(v))
                writer.writerow(row_formatted)

        self.extracted_files.append(csv_path)
        self._log(f"[CSV Numeric] {dset_path} ({num_rows} steps, {num_cols} cols) -> {csv_filename}")

    def _write_summary(self, elapsed_sec: float):
        """Ghi báo cáo tóm tắt các tệp tin đã tạo ra."""
        summary_path = os.path.join(self.output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=======================================================\n")
            f.write(" BÁO CÁO TRÍCH XUẤT DỮ LIỆU TỆP HDF5\n")
            f.write("=======================================================\n")
            f.write(f"Tệp gốc        : {self.filepath}\n")
            f.write(f"Thư mục xuất   : {self.output_dir}\n")
            f.write(f"Thời gian xử lý: {elapsed_sec:.2f} giây\n")
            f.write(f"Ngày trích xuất: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("--- THÔNG TIN PHIÊN GHI (EPISODE ATTRIBUTES) ---\n")
            for k, v in self.metadata.items():
                f.write(f"  * {k}: {v}\n")

            f.write("\n--- DANH SÁCH TỆP TIN ĐÃ XUẤT ---\n")
            for p in sorted(self.extracted_files):
                rel_p = os.path.relpath(p, self.output_dir)
                size_str = ""
                if os.path.isfile(p):
                    size_kb = os.path.getsize(p) / 1024.0
                    size_str = f"({size_kb:.1f} KB)"
                elif os.path.isdir(p):
                    count = len(os.listdir(p))
                    size_str = f"({count} files)"
                f.write(f"  - {rel_p} {size_str}\n")
            f.write("=======================================================\n")

        self.extracted_files.append(summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Giải nén và trích xuất dữ liệu HDF5 (UR3e Teleoperation & AI Datasets) sang MP4, PNG và CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        type=str,
        help="Đường dẫn đến tệp .hdf5 cần giải nén (Ví dụ: logs/ai_training_data/episode_0.hdf5)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        type=str,
        help="Đường dẫn thư mục lưu kết quả (Mặc định: tạo thư mục '<tên_file>_extracted' tại cùng thư mục chứa tệp HDF5)"
    )
    parser.add_argument(
        "--fps",
        default=None,
        type=float,
        help="Ghi đè tốc độ khung hình (FPS) cho video MP4 (Mặc định đọc từ thuộc tính 'hz' trong tệp HDF5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Tắt thông báo chi tiết trong quá trình xuất"
    )

    args = parser.parse_args()

    try:
        extractor = HDF5Extractor(
            filepath=args.file,
            output_dir=args.output_dir,
            fps_override=args.fps,
            verbose=not args.quiet
        )
        out_dir = extractor.extract_all()
        print(f"\n[Thành công] Toàn bộ dữ liệu đã được xuất tại: {out_dir}")
    except Exception as e:
        print(f"\n[LỖI] Quá trình trích xuất thất bại: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
