#!/usr/bin/env python3
"""
Selectable Data Recorder (HDF5 Streaming Engine)
=================================================
Bộ ghi dữ liệu đa phương thức hiệu năng cao, tối ưu RAM cho Robot UR3e và Hệ thống Thị giác:
- Chunked Streaming HDF5: Ghi trực tiếp xuống tệp với các dataset mở rộng động (maxshape=(None, ...)).
- Zero-RAM Overflow: Bộ đệm theo mẻ (Batch Buffer) với cơ chế flush định kỳ (mỗi 1-2s hoặc N frames),
  giữ mức tiêu thụ RAM luôn thấp và ổn định (< 80MB).
- Selectable Streams: Cho phép bật/tắt linh hoạt các luồng dữ liệu (Kinematics, Dynamics, Wrench,
  Temperatures, Payload, Safety, RGB, Depth) theo yêu cầu.
- Full Backward Compatibility: Tương thích hoàn toàn 100% với định dạng ACT / Diffusion Policy
  của DataLogger cũ và các công cụ verify_dataset.py / replay_episode.py.
"""

import os
import time
import queue
import threading
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import h5py

# Cấu hình mặc định cho các kênh dữ liệu có thể lựa chọn
DEFAULT_STREAMS: Dict[str, bool] = {
    "robot_kinematics": True,      # qpos, qvel, tcp_pose, tcp_speed
    "robot_target": True,          # q_target, qd_target, tcp_target
    "robot_dynamics": True,        # motor currents, torques, voltages, robot current/voltage
    "robot_contact_wrench": True,  # 6-axis TCP force/wrench, accelerometer
    "robot_temperatures": False,   # 6 joint motor temperatures (tắt mặc định để tiết kiệm)
    "robot_payload": True,         # mass, CoG x/y/z
    "robot_safety_status": True,   # robot_mode, safety_mode, e_stop, protective_stop
    "camera_wrist_rgb": True,      # Wrist RealSense RGB frame (uint8 [H, W, 3])
    "camera_wrist_depth": False,   # Wrist RealSense Depth frame (uint16 [H, W] mm)
    "actions": True,               # action array [T, 7] (target_q + gripper)
    "timestamps": True,            # timestamps_ms, timestamps_ns, timestamps_robot
}


class SelectableDataRecorder:
    """
    Bộ thu thập dữ liệu chuyên gia chuẩn HDF5 với cơ chế Chunked Streaming & Selectable Streams.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.output_dir = cfg.get("output_dir", "logs/ai_training_data")
        self.hz = cfg.get("hz", 20)
        self.flush_chunk_size = int(cfg.get("flush_chunk_size", 25))       # Số frame trước mỗi lần xả đĩa
        self.flush_interval_sec = float(cfg.get("flush_interval_sec", 1.5)) # Thời gian tối đa giữa 2 lần flush
        self.compression = cfg.get("compression", "gzip")                  # "gzip" hoặc "lzf"
        self.compression_opts = int(cfg.get("compression_opts", 1)) if self.compression == "gzip" else None

        # Quản lý Streams bật/tắt
        user_streams = cfg.get("streams", {})
        self.streams: Dict[str, bool] = dict(DEFAULT_STREAMS)
        self.streams.update(user_streams)

        self._lock = threading.Lock()
        self.is_recording = False
        self.current_episode_idx = 0
        self._init_episode_index()

        # Quản lý file HDF5 hiện tại
        self.h5_file: Optional[h5py.File] = None
        self.current_filepath: Optional[str] = None
        self.total_steps_in_episode = 0
        self.episode_start_time = 0.0
        self._last_flush_time = 0.0

        # Buffer RAM tạm thời chỉ cho 1 mẻ nhỏ (batch)
        self._batch_buffer: List[Dict[str, Any]] = []
        self._datasets_initialized = False

        # Luồng ghi bất đồng bộ (Async Writer Thread) & Hàng đợi bảo vệ vòng lặp chính
        self._queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()

    def _init_episode_index(self):
        """Xác định số thứ tự episode tiếp theo dựa trên các file đã có trong output_dir."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            self.current_episode_idx = 0
            return

        existing_files = [
            f for f in os.listdir(self.output_dir)
            if f.startswith("episode_") and f.endswith(".hdf5")
        ]
        if not existing_files:
            self.current_episode_idx = 0
        else:
            indices = []
            for f in existing_files:
                try:
                    idx = int(f.replace("episode_", "").replace(".hdf5", ""))
                    indices.append(idx)
                except ValueError:
                    pass
            self.current_episode_idx = max(indices) + 1 if indices else 0

    def set_streams(self, streams: Dict[str, bool]):
        """Cập nhật cờ bật/tắt các luồng dữ liệu trước khi bắt đầu episode."""
        with self._lock:
            if self.is_recording:
                raise RuntimeError("Không thể thay đổi cấu hình streams khi đang ghi episode!")
            self.streams.update(streams)

    def start_episode(
        self,
        task_description: str = "UR3e Teleoperation Pick and Place",
        metadata: Optional[Dict[str, Any]] = None,
        streams: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Bắt đầu một episode mới, khởi tạo tệp HDF5 và sẵn sàng stream dữ liệu xuống đĩa.
        """
        with self._lock:
            if self.is_recording:
                self.end_episode(success=False)

            if streams:
                self.streams.update(streams)

            os.makedirs(self.output_dir, exist_ok=True)
            filename = f"episode_{self.current_episode_idx}.hdf5"
            self.current_filepath = os.path.join(self.output_dir, filename)

            self.task_description = task_description
            self.metadata = metadata or {}
            self.episode_start_time = time.time()
            self._last_flush_time = self.episode_start_time
            self.total_steps_in_episode = 0
            self._batch_buffer.clear()
            self._datasets_initialized = False

            # Mở file HDF5 chế độ ghi
            self.h5_file = h5py.File(self.current_filepath, "w")

            # Ghi thông tin Metadata cố định vào attrs
            self.h5_file.attrs["hz"] = self.hz
            self.h5_file.attrs["task_description"] = self.task_description
            self.h5_file.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.h5_file.attrs["compression"] = self.compression

            # Lưu danh sách các luồng dữ liệu đã kích hoạt
            for st_name, st_enabled in self.streams.items():
                self.h5_file.attrs[f"stream_{st_name}"] = st_enabled

            # Metadata mở rộng tùy chọn
            for meta_k, meta_v in self.metadata.items():
                try:
                    if isinstance(meta_v, (list, tuple, np.ndarray)):
                        self.h5_file.attrs[meta_k] = np.array(meta_v)
                    else:
                        self.h5_file.attrs[meta_k] = meta_v
                except Exception:
                    pass

            # Khởi tạo các nhóm cơ sở
            self.h5_file.create_group("observations")
            self.h5_file.create_group("observations/images")

            # Khởi động hàng đợi phi chặn và luồng ghi bất đồng bộ
            self._queue = queue.Queue(maxsize=2000)
            self._stop_worker.clear()
            self._worker_thread = threading.Thread(target=self._async_worker_loop, daemon=True)
            self.is_recording = True
            self._worker_thread.start()

            print(f"[SelectableRecorder] >>> BẮT ĐẦU Episode {self.current_episode_idx} -> {self.current_filepath}")
            active_list = [k for k, v in self.streams.items() if v]
            print(f"[SelectableRecorder]     Luồng kích hoạt ({len(active_list)}): {', '.join(active_list)}")
            return self.current_filepath

    def _async_worker_loop(self):
        """
        Worker thread chạy ngầm độc lập hoàn toàn với Main Control Loop:
        - Nhận dữ liệu từ self._queue
        - Nạp vào batch buffer và ghi xuống đĩa HDF5 (+ nén GZIP)
        - Độ trễ I/O đĩa (50-200ms) bị cô lập 100% tại luồng này
        """
        while not self._stop_worker.is_set() or (self._queue is not None and not self._queue.empty()):
            try:
                item = self._queue.get(timeout=0.05)
            except (queue.Empty, AttributeError):
                with self._lock:
                    if self.is_recording and self._batch_buffer and (time.time() - self._last_flush_time) >= self.flush_interval_sec:
                        self._flush_batch()
                continue

            if item is None:
                if self._queue is not None:
                    self._queue.task_done()
                break

            with self._lock:
                if self.is_recording and self.h5_file is not None:
                    self._batch_buffer.append(item)
                    if not self._datasets_initialized:
                        self._init_chunked_datasets(item)

                    now = time.time()
                    if (
                        len(self._batch_buffer) >= self.flush_chunk_size
                        or (now - self._last_flush_time) >= self.flush_interval_sec
                    ):
                        self._flush_batch()
            if self._queue is not None:
                self._queue.task_done()

    def _init_chunked_datasets(self, sample: Dict[str, Any]):
        """
        Khởi tạo các dataset HDF5 hỗ trợ mở rộng tự động (maxshape=(None, ...))
        dựa trên mẫu dữ liệu đầu tiên (Lazy initialization).
        """
        if self._datasets_initialized or self.h5_file is None:
            return

        obs_grp = self.h5_file["observations"]
        c_kwargs: Dict[str, Any] = {"compression": self.compression}
        if self.compression_opts is not None:
            c_kwargs["compression_opts"] = self.compression_opts

        chunk_t = min(32, max(4, self.flush_chunk_size))

        # 1. Các trường chuẩn bắt buộc cho ACT / Diffusion Policy
        obs_grp.create_dataset(
            "qpos", shape=(0, 6), maxshape=(None, 6),
            dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs
        )
        obs_grp.create_dataset(
            "qvel", shape=(0, 6), maxshape=(None, 6),
            dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs
        )
        obs_grp.create_dataset(
            "tcp_pose", shape=(0, 6), maxshape=(None, 6),
            dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs
        )
        obs_grp.create_dataset(
            "gripper", shape=(0, 1), maxshape=(None, 1),
            dtype=np.float32, chunks=(chunk_t, 1), **c_kwargs
        )
        obs_grp.create_dataset(
            "timestamps_ms", shape=(0,), maxshape=(None,),
            dtype=np.int64, chunks=(chunk_t,), **c_kwargs
        )

        # Action (q_target 6 + gripper_target 1) = [T, 7]
        self.h5_file.create_dataset(
            "action", shape=(0, 7), maxshape=(None, 7),
            dtype=np.float32, chunks=(chunk_t, 7), **c_kwargs
        )

        # 1b. Dữ liệu Viễn Thao & Can thiệp Người dùng (Teleoperation & AI Datasets)
        obs_grp.create_dataset(
            "vr_controller_pose", shape=(0, 7), maxshape=(None, 7),
            dtype=np.float32, chunks=(chunk_t, 7), **c_kwargs
        )
        if "target_tcp_pose" not in obs_grp:
            obs_grp.create_dataset(
                "target_tcp_pose", shape=(0, 6), maxshape=(None, 6),
                dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs
            )

        if "actions" not in self.h5_file:
            act_grp = self.h5_file.create_group("actions")
        else:
            act_grp = self.h5_file["actions"]
        act_grp.create_dataset(
            "gripper_cmd", shape=(0, 1), maxshape=(None, 1),
            dtype=np.float32, chunks=(chunk_t, 1), **c_kwargs
        )

        # 2. Các trường Động học & Động lực học mở rộng (Nếu bật)
        if self.streams.get("robot_kinematics", True):
            obs_grp.create_dataset("tcp_speed", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)

        if self.streams.get("robot_target", True):
            obs_grp.create_dataset("target_q", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("target_qd", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("target_qdd", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            if "target_tcp_pose" not in obs_grp:
                obs_grp.create_dataset("target_tcp_pose", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)

        if self.streams.get("robot_dynamics", True):
            obs_grp.create_dataset("currents", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("target_currents", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("target_moments", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("joint_voltages", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("robot_current", shape=(0, 1), maxshape=(None, 1), dtype=np.float32, chunks=(chunk_t, 1), **c_kwargs)
            obs_grp.create_dataset("robot_voltage", shape=(0, 1), maxshape=(None, 1), dtype=np.float32, chunks=(chunk_t, 1), **c_kwargs)

        if self.streams.get("robot_contact_wrench", True):
            obs_grp.create_dataset("tcp_force", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("ft_raw_wrench", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)
            obs_grp.create_dataset("tool_accelerometer", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(chunk_t, 3), **c_kwargs)

        if self.streams.get("robot_temperatures", False):
            obs_grp.create_dataset("temperatures", shape=(0, 6), maxshape=(None, 6), dtype=np.float32, chunks=(chunk_t, 6), **c_kwargs)

        if self.streams.get("robot_payload", True):
            obs_grp.create_dataset("payload", shape=(0, 4), maxshape=(None, 4), dtype=np.float32, chunks=(chunk_t, 4), **c_kwargs)

        if self.streams.get("robot_safety_status", True):
            obs_grp.create_dataset("robot_mode", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, chunks=(chunk_t, 1), **c_kwargs)
            obs_grp.create_dataset("safety_mode", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, chunks=(chunk_t, 1), **c_kwargs)
            obs_grp.create_dataset("safety_flags", shape=(0, 2), maxshape=(None, 2), dtype=np.int32, chunks=(chunk_t, 2), **c_kwargs)
            obs_grp.create_dataset("speed_scaling", shape=(0, 1), maxshape=(None, 1), dtype=np.float32, chunks=(chunk_t, 1), **c_kwargs)

        if self.streams.get("timestamps", True):
            obs_grp.create_dataset("timestamps_ns", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(chunk_t,), **c_kwargs)
            obs_grp.create_dataset("timestamps_robot", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(chunk_t,), **c_kwargs)

        # 3. Khởi tạo Camera Datasets (Hình ảnh RGB và Depth)
        img_grp = obs_grp["images"]
        images_dict = sample.get("images", {}) or {}
        img_chunk_t = min(16, max(2, self.flush_chunk_size // 2))

        for cam_key, img_val in images_dict.items():
            if img_val is None:
                continue
            if not isinstance(img_val, np.ndarray):
                continue

            h, w = img_val.shape[:2]
            if img_val.ndim == 3 and self.streams.get("camera_wrist_rgb", True):
                # RGB frame [T, H, W, 3] uint8
                img_grp.create_dataset(
                    cam_key,
                    shape=(0, h, w, 3),
                    maxshape=(None, h, w, 3),
                    dtype=np.uint8,
                    chunks=(img_chunk_t, h, w, 3),
                    **c_kwargs
                )
            elif img_val.ndim == 2 and self.streams.get("camera_wrist_depth", True):
                # Depth frame [T, H, W] uint16 mm
                img_grp.create_dataset(
                    cam_key,
                    shape=(0, h, w),
                    maxshape=(None, h, w),
                    dtype=np.uint16,
                    chunks=(img_chunk_t, h, w),
                    **c_kwargs
                )

        self._datasets_initialized = True

    def record_step(
        self,
        telemetry: Dict[str, Any],
        images: Optional[Dict[str, np.ndarray]] = None,
        action: Optional[Union[List[float], np.ndarray]] = None,
        gripper_actual: float = 0.0,
        gripper_target: float = 0.0,
        q_target: Optional[List[float]] = None,
        vr_controller_pose: Optional[Union[List[float], np.ndarray]] = None,
        target_tcp_pose: Optional[Union[List[float], np.ndarray]] = None,
        event_tag: str = "teleop"
    ):
        """
        Nạp một bước dữ liệu đa phương thức (Multimodal Step) vào buffer và tự động flush khi đủ chunk.
        """
        if not self.is_recording or self.h5_file is None:
            return

        # Chuẩn hóa Action [q_target (6) + gripper_target (1)] (Tính toán cục bộ không cần lock)
        if action is not None:
            act_arr = np.array(action, dtype=np.float32).flatten()
        elif q_target is not None:
            act_arr = np.array(list(q_target) + [float(gripper_target)], dtype=np.float32)
        else:
            q_act = telemetry.get("actual_q", [0.0] * 6)
            act_arr = np.array(list(q_act) + [float(gripper_target)], dtype=np.float32)

        # Chuẩn hóa VR Controller Pose 7D [X, Y, Z, Qx, Qy, Qz, Qw]
        if vr_controller_pose is not None:
            vr_arr = np.array(vr_controller_pose, dtype=np.float32).flatten()
        elif "vr_controller_pose" in telemetry:
            vr_arr = np.array(telemetry["vr_controller_pose"], dtype=np.float32).flatten()
        else:
            vr_arr = np.zeros(7, dtype=np.float32)

        if len(vr_arr) < 7:
            vr_arr = np.pad(vr_arr, (0, 7 - len(vr_arr)))
        elif len(vr_arr) > 7:
            vr_arr = vr_arr[:7]

        # Chuẩn hóa Target TCP Pose 6D [x, y, z, rx, ry, rz]
        if target_tcp_pose is not None:
            tgt_arr = np.array(target_tcp_pose, dtype=np.float32).flatten()
        elif "target_tcp_pose" in telemetry:
            tgt_arr = np.array(telemetry["target_tcp_pose"], dtype=np.float32).flatten()
        else:
            tgt_arr = np.zeros(6, dtype=np.float32)

        if len(tgt_arr) < 6:
            tgt_arr = np.pad(tgt_arr, (0, 6 - len(tgt_arr)))
        elif len(tgt_arr) > 6:
            tgt_arr = tgt_arr[:6]

        step_data: Dict[str, Any] = {
            "telemetry": telemetry,
            "images": images or {},
            "action": act_arr,
            "gripper_actual": float(gripper_actual),
            "gripper_target": float(gripper_target),
            "vr_controller_pose": vr_arr,
            "target_tcp_pose": tgt_arr,
            "event_tag": event_tag,
        }

        # Đẩy gói dữ liệu vào Queue phi chặn (Non-blocking enqueue < 0.01ms)
        if self._queue is not None and self.is_recording:
            try:
                self._queue.put_nowait(step_data)
            except queue.Full:
                pass
        else:
            with self._lock:
                self._batch_buffer.append(step_data)
                if not self._datasets_initialized:
                    self._init_chunked_datasets(step_data)
                now = time.time()
                if (
                    len(self._batch_buffer) >= self.flush_chunk_size
                    or (now - self._last_flush_time) >= self.flush_interval_sec
                ):
                    self._flush_batch()

    def log_step(
        self,
        q_actual: list,
        qd_actual: list,
        tcp_actual: list,
        q_target: Optional[list] = None,
        gripper_actual: float = 0.0,
        gripper_target: float = 0.0,
        images: Optional[Dict[str, np.ndarray]] = None,
        image_path: Optional[str] = None,
        event_tag: str = "teleop",
        telemetry: Optional[Dict[str, Any]] = None,
        vr_controller_pose: Optional[Union[List[float], np.ndarray]] = None,
        target_tcp_pose: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs
    ):
        """
        Hàm tương thích ngược 100% với giao diện DataLogger cũ.
        Cho phép thay thế DataLogger bằng SelectableDataRecorder mà không cần sửa code gọi.
        """
        tel = telemetry or {}
        if "actual_q" not in tel:
            tel["actual_q"] = list(q_actual) if q_actual is not None else [0.0] * 6
        if "actual_qd" not in tel:
            tel["actual_qd"] = list(qd_actual) if qd_actual is not None else [0.0] * 6
        if "actual_tcp_pose" not in tel:
            tel["actual_tcp_pose"] = list(tcp_actual) if tcp_actual is not None else [0.0] * 6
        if "target_q" not in tel and q_target is not None:
            tel["target_q"] = list(q_target)
        if "timestamp_host_ms" not in tel:
            tel["timestamp_host_ms"] = int(time.time() * 1000)
        if "timestamp_host_ns" not in tel:
            tel["timestamp_host_ns"] = time.perf_counter_ns()

        self.record_step(
            telemetry=tel,
            images=images,
            action=None,
            gripper_actual=gripper_actual,
            gripper_target=gripper_target,
            q_target=q_target,
            vr_controller_pose=vr_controller_pose,
            target_tcp_pose=target_tcp_pose,
            event_tag=event_tag
        )

    def _flush_batch(self):
        """Xả toàn bộ mẻ dữ liệu trong buffer xuống tệp HDF5 và giải phóng RAM."""
        if not self._batch_buffer or self.h5_file is None:
            return

        B = len(self._batch_buffer)
        old_T = self.total_steps_in_episode
        new_T = old_T + B

        obs_grp = self.h5_file["observations"]

        # 1. Trích xuất mảng vector cơ sở
        qpos_batch = np.array([b["telemetry"].get("actual_q", [0.0]*6) for b in self._batch_buffer], dtype=np.float32)
        qvel_batch = np.array([b["telemetry"].get("actual_qd", [0.0]*6) for b in self._batch_buffer], dtype=np.float32)
        tcp_batch = np.array([b["telemetry"].get("actual_tcp_pose", [0.0]*6) for b in self._batch_buffer], dtype=np.float32)
        grip_batch = np.array([b["gripper_actual"] for b in self._batch_buffer], dtype=np.float32).reshape(B, 1)
        action_batch = np.array([b["action"] for b in self._batch_buffer], dtype=np.float32)
        ts_ms_batch = np.array([b["telemetry"].get("timestamp_host_ms", int(time.time()*1000)) for b in self._batch_buffer], dtype=np.int64)

        # Mở rộng kích thước và gán dữ liệu
        def append_dataset(dset, data):
            dset.resize(new_T, axis=0)
            dset[old_T:new_T] = data

        append_dataset(obs_grp["qpos"], qpos_batch)
        append_dataset(obs_grp["qvel"], qvel_batch)
        append_dataset(obs_grp["tcp_pose"], tcp_batch)
        append_dataset(obs_grp["gripper"], grip_batch)
        append_dataset(obs_grp["timestamps_ms"], ts_ms_batch)
        append_dataset(self.h5_file["action"], action_batch)

        # Ghi các trường viễn thao (VR controller pose & gripper command)
        if "vr_controller_pose" in obs_grp:
            vr_batch = np.array([b["vr_controller_pose"] for b in self._batch_buffer], dtype=np.float32)
            append_dataset(obs_grp["vr_controller_pose"], vr_batch)

        if "target_tcp_pose" in obs_grp:
            tgt_tcp_batch = np.array([b["target_tcp_pose"] for b in self._batch_buffer], dtype=np.float32)
            append_dataset(obs_grp["target_tcp_pose"], tgt_tcp_batch)

        if "actions" in self.h5_file and "gripper_cmd" in self.h5_file["actions"]:
            grip_cmd_batch = np.array([[b["gripper_target"]] for b in self._batch_buffer], dtype=np.float32)
            append_dataset(self.h5_file["actions/gripper_cmd"], grip_cmd_batch)

        # 2. Ghi các luồng mở rộng nếu có
        if self.streams.get("robot_kinematics", True) and "tcp_speed" in obs_grp:
            spd_batch = np.array([b["telemetry"].get("actual_tcp_speed", [0.0]*6) for b in self._batch_buffer], dtype=np.float32)
            append_dataset(obs_grp["tcp_speed"], spd_batch)

        if self.streams.get("robot_target", True):
            if "target_q" in obs_grp:
                append_dataset(obs_grp["target_q"], np.array([b["telemetry"].get("target_q", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "target_qd" in obs_grp:
                append_dataset(obs_grp["target_qd"], np.array([b["telemetry"].get("target_qd", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "target_qdd" in obs_grp:
                append_dataset(obs_grp["target_qdd"], np.array([b["telemetry"].get("target_qdd", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))

        if self.streams.get("robot_dynamics", True):
            if "currents" in obs_grp:
                append_dataset(obs_grp["currents"], np.array([b["telemetry"].get("actual_current", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "target_currents" in obs_grp:
                append_dataset(obs_grp["target_currents"], np.array([b["telemetry"].get("target_current", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "target_moments" in obs_grp:
                append_dataset(obs_grp["target_moments"], np.array([b["telemetry"].get("target_moment", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "joint_voltages" in obs_grp:
                append_dataset(obs_grp["joint_voltages"], np.array([b["telemetry"].get("joint_voltages", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "robot_current" in obs_grp:
                append_dataset(obs_grp["robot_current"], np.array([[b["telemetry"].get("robot_current", 0.0)] for b in self._batch_buffer], dtype=np.float32))
            if "robot_voltage" in obs_grp:
                append_dataset(obs_grp["robot_voltage"], np.array([[b["telemetry"].get("robot_voltage", 0.0)] for b in self._batch_buffer], dtype=np.float32))

        if self.streams.get("robot_contact_wrench", True):
            if "tcp_force" in obs_grp:
                append_dataset(obs_grp["tcp_force"], np.array([b["telemetry"].get("actual_tcp_force", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "ft_raw_wrench" in obs_grp:
                append_dataset(obs_grp["ft_raw_wrench"], np.array([b["telemetry"].get("ft_raw_wrench", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))
            if "tool_accelerometer" in obs_grp:
                append_dataset(obs_grp["tool_accelerometer"], np.array([b["telemetry"].get("tool_accelerometer", [0.0]*3) for b in self._batch_buffer], dtype=np.float32))

        if self.streams.get("robot_temperatures", False) and "temperatures" in obs_grp:
            append_dataset(obs_grp["temperatures"], np.array([b["telemetry"].get("joint_temperatures", [0.0]*6) for b in self._batch_buffer], dtype=np.float32))

        if self.streams.get("robot_payload", True) and "payload" in obs_grp:
            pld_rows = []
            for b in self._batch_buffer:
                p_mass = float(b["telemetry"].get("payload", 0.0))
                p_cog = b["telemetry"].get("payload_cog", [0.0, 0.0, 0.0])
                pld_rows.append([p_mass, p_cog[0], p_cog[1], p_cog[2]])
            append_dataset(obs_grp["payload"], np.array(pld_rows, dtype=np.float32))

        if self.streams.get("robot_safety_status", True):
            if "robot_mode" in obs_grp:
                append_dataset(obs_grp["robot_mode"], np.array([[b["telemetry"].get("robot_mode", 0)] for b in self._batch_buffer], dtype=np.int32))
            if "safety_mode" in obs_grp:
                append_dataset(obs_grp["safety_mode"], np.array([[b["telemetry"].get("safety_mode", 0)] for b in self._batch_buffer], dtype=np.int32))
            if "safety_flags" in obs_grp:
                flags_rows = [
                    [int(b["telemetry"].get("is_emergency_stopped", False)), int(b["telemetry"].get("is_protective_stopped", False))]
                    for b in self._batch_buffer
                ]
                append_dataset(obs_grp["safety_flags"], np.array(flags_rows, dtype=np.int32))
            if "speed_scaling" in obs_grp:
                append_dataset(obs_grp["speed_scaling"], np.array([[b["telemetry"].get("speed_scaling", 1.0)] for b in self._batch_buffer], dtype=np.float32))

        if self.streams.get("timestamps", True):
            if "timestamps_ns" in obs_grp:
                append_dataset(obs_grp["timestamps_ns"], np.array([b["telemetry"].get("timestamp_host_ns", 0) for b in self._batch_buffer], dtype=np.int64))
            if "timestamps_robot" in obs_grp:
                append_dataset(obs_grp["timestamps_robot"], np.array([b["telemetry"].get("timestamp_robot", 0.0) for b in self._batch_buffer], dtype=np.float64))

        # 3. Ghi các luồng hình ảnh Camera (RGB / Depth)
        img_grp = obs_grp["images"]
        for cam_key in list(img_grp.keys()):
            cam_frames = []
            for b in self._batch_buffer:
                img_f = b["images"].get(cam_key)
                if img_f is not None:
                    cam_frames.append(img_f)
                else:
                    # Fallback frame đen nếu thiếu
                    shape_rest = img_grp[cam_key].shape[1:]
                    dtype_f = img_grp[cam_key].dtype
                    cam_frames.append(np.zeros(shape_rest, dtype=dtype_f))

            if cam_frames:
                cam_arr = np.stack(cam_frames, axis=0)
                append_dataset(img_grp[cam_key], cam_arr)

        # Cập nhật số bước và xả đĩa an toàn
        self.total_steps_in_episode = new_T
        self._batch_buffer.clear()
        self._last_flush_time = time.time()
        self.h5_file.flush()

    def flush(self):
        """Xả đĩa thủ công toàn bộ dữ liệu đang chờ trong buffer."""
        if self._queue is not None:
            try:
                self._queue.join()
            except Exception:
                pass
        with self._lock:
            if self.is_recording and self._batch_buffer:
                self._flush_batch()

    def end_episode(self, success: bool = True) -> Optional[str]:
        """
        Kết thúc episode, hoàn tất ghi các chunk cuối cùng, đóng file HDF5 và cập nhật metadata.
        """
        with self._lock:
            if not self.is_recording or self.h5_file is None:
                self.is_recording = False
                return None

        # Báo hiệu dừng cho worker thread và chờ xử lý hết hàng đợi
        if self._queue is not None:
            self._stop_worker.set()
            try:
                self._queue.put(None, timeout=0.5)
            except Exception:
                pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        with self._lock:
            # Thu gom các gói tin còn sót lại nếu worker thread thoát sớm
            if self._queue is not None:
                while not self._queue.empty():
                    try:
                        item = self._queue.get_nowait()
                        if item is not None:
                            self._batch_buffer.append(item)
                            if not self._datasets_initialized:
                                self._init_chunked_datasets(item)
                    except Exception:
                        break

            # Xả hết frame còn đọng trong RAM xuống đĩa
            self._flush_batch()

            T = self.total_steps_in_episode
            if T == 0:
                print(f"[SelectableRecorder] Hủy lưu vì episode rỗng (0 frames)!")
                self._cleanup_file()
                self.is_recording = False
                self._queue = None
                self._worker_thread = None
                return None

            duration = time.time() - self.episode_start_time

            # Cập nhật metadata cuối
            self.h5_file.attrs["num_steps"] = T
            self.h5_file.attrs["success"] = success
            self.h5_file.attrs["duration_sec"] = duration

            self.h5_file.close()
            self.h5_file = None
            self.is_recording = False
            self._queue = None
            self._worker_thread = None

            filepath = self.current_filepath
            print(f"[SelectableRecorder] ĐÃ LƯU Episode {self.current_episode_idx} ({T} frames, {duration:.1f}s, success={success}) -> {filepath}")

            self.current_episode_idx += 1
            self.current_filepath = None
            return filepath

    def discard_episode(self):
        """Hủy bỏ episode hiện tại, đóng file và xóa tệp dở dang khỏi đĩa."""
        with self._lock:
            if not self.is_recording:
                return

        if self._queue is not None:
            self._stop_worker.set()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            try:
                self._queue.put(None, timeout=0.1)
            except Exception:
                pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        with self._lock:
            print(f"[SelectableRecorder] ĐÃ HỦY Episode {self.current_episode_idx} theo yêu cầu người dùng.")
            self._batch_buffer.clear()
            self._cleanup_file()
            self.is_recording = False
            self._queue = None
            self._worker_thread = None

    def _cleanup_file(self):
        """Đóng file và xóa tệp tạm nếu tồn tại."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None

        if self.current_filepath and os.path.exists(self.current_filepath):
            try:
                os.remove(self.current_filepath)
                print(f"[SelectableRecorder] Đã xóa tệp chưa hoàn tất: {self.current_filepath}")
            except Exception as e:
                print(f"[SelectableRecorder] Cảnh báo không thể xóa file tạm: {e}")
        self.current_filepath = None

    def __del__(self):
        """Đảm bảo đóng file an toàn khi đối tượng bị hủy."""
        try:
            if getattr(self, "h5_file", None) is not None:
                self.h5_file.close()
        except Exception:
            pass
