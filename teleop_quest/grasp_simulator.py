import numpy as np
from typing import Tuple, Optional

class GraspSimulator:
    """
    Mô phỏng gắp nhặt vật thể ảo 3D (Quả cà chua) trong Viser:
      - Spawns quả cà chua và khay chứa trong không gian 3D.
      - Cơ chế Kinematic/Magnetic Grasping: hút dính quả cà chua khi đầu gắp chạm gần và bóp cò Grip.
      - Thả quả cà chua vào khay chứa khi nhả Grip.
    """
    def __init__(self, server, pick_pos=(0.20, 0.20, 0.042), place_pos=(0.15, -0.20, 0.025)):
        self.server = server
        self.pick_init_pos = np.array(pick_pos, dtype=np.float32)
        self.place_basket_pos = np.array(place_pos, dtype=np.float32)
        
        self.tomato_pos = self.pick_init_pos.copy()
        self.is_grasped = False
        self.placed_count = 0
        # Ngưỡng khoảng cách gắp 9.5cm (kết hợp với điều kiện hạ thấp Z để gắp chính xác)
        self.grasp_distance_threshold = 0.095  # 9.5cm
        
        # 1. Sinh quả cà chua 3D (Khối cầu đỏ)
        self.tomato_mesh = self.server.scene.add_icosphere(
            "/world/tomato",
            radius=0.032,
            color=(235, 45, 45),
            position=tuple(self.tomato_pos)
        )
        # Cuống lá xanh nhỏ trên đỉnh quả cà chua
        self.stem_mesh = self.server.scene.add_icosphere(
            "/world/tomato_stem",
            radius=0.008,
            color=(40, 180, 50),
            position=(self.tomato_pos[0], self.tomato_pos[1], self.tomato_pos[2] + 0.032)
        )

        # 2. Sinh chiếc khay/giỏ đựng (Khối hộp xanh dương)
        self.basket_mesh = self.server.scene.add_box(
            "/world/basket",
            dimensions=(0.15, 0.15, 0.03),
            color=(60, 150, 230),
            position=tuple(self.place_basket_pos)
        )

        # 3. Sinh mặt bàn làm việc 3D (Gỗ xám nhạt) nằm trong tầm với của tay máy (Mặt bàn Z = +0.01m)
        self.table_mesh = self.server.scene.add_box(
            "/world/table",
            dimensions=(0.75, 0.80, 0.04),
            color=(185, 175, 160),
            position=(0.20, 0.0, -0.01)
        )
        self.wall_mesh = None

    def set_position(self, x: float, y: float, z: float = 0.042):
        """Đặt vị trí quả cà chua tại tọa độ bất kỳ để thử nghiệm"""
        self.tomato_pos = np.array([x, y, z], dtype=np.float32)
        self.pick_init_pos = self.tomato_pos.copy()
        self._update_mesh_positions()

    def update(self, tool_pos: list, grip_pressed: bool) -> Tuple[bool, Optional[str]]:
        """
        Cập nhật logic gắp thả theo vị trí đầu gắp TCP (robot world) và trạng thái nút Grip.
        Trả về: (is_grasped, event_message)
        """
        tool_p = np.array(tool_pos[:3], dtype=np.float32)
        event = None

        if not self.is_grasped:
            # Chưa gắp: Kiểm tra khoảng cách tới quả cà chua
            dist = np.linalg.norm(tool_p - self.tomato_pos)
            dist_xy = np.linalg.norm(tool_p[:2] - self.tomato_pos[:2])
            # Cho phép gắp khi khoảng cách 3D < 9.5cm HOẶC căn chuẩn tâm XY < 5.5cm và đã hạ thấp Z < 18cm
            can_grasp = (dist < self.grasp_distance_threshold) or (dist_xy < 0.055 and tool_p[2] < 0.18)
            if can_grasp and grip_pressed:
                self.is_grasped = True
                event = "GRASP_SUCCESS"
                print(f"[GraspSim] ĐÃ GẮP QUẢ CÀ CHUA! (Khoảng cách: {dist*100:.1f}cm, lệch XY: {dist_xy*100:.1f}cm)")
        else:
            # Đang gắp:
            if grip_pressed:
                # Quả cà chua di chuyển dính theo đầu gắp robot
                self.tomato_pos = tool_p.copy()
                self._update_mesh_positions()
            else:
                # Người dùng nhả nút Grip -> Thả quả cà chua
                self.is_grasped = False
                dist_to_basket = np.linalg.norm(self.tomato_pos[:2] - self.place_basket_pos[:2])
                if dist_to_basket < 0.10:
                    # Rơi trúng vào khay chứa!
                    self.tomato_pos[2] = self.place_basket_pos[2] + 0.032
                    self.placed_count += 1
                    event = "PLACE_SUCCESS"
                    print(f"[GraspSim] THÀNH CÔNG! Đã đặt quả cà chua vào khay (Tổng: {self.placed_count})")
                else:
                    # Rơi xuống mặt bàn
                    self.tomato_pos[2] = 0.035
                    event = "DROPPED"
                    print("[GraspSim] Quả cà chua rơi xuống mặt bàn.")
                self._update_mesh_positions()

        return self.is_grasped, event

    def _update_mesh_positions(self):
        """Cập nhật vị trí hiển thị 3D trên Viser"""
        self.tomato_mesh.position = tuple(self.tomato_pos)
        self.stem_mesh.position = (self.tomato_pos[0], self.tomato_pos[1], self.tomato_pos[2] + 0.032)

    def reset(self):
        """Đặt lại quả cà chua về vị trí gắp ban đầu"""
        self.is_grasped = False
        self.tomato_pos = self.pick_init_pos.copy()
        self._update_mesh_positions()
        print(f"[GraspSim] Đã reset quả cà chua về vị trí ban đầu {self.pick_init_pos}")
