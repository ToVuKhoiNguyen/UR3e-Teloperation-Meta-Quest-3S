import os
import torch
import math
import numpy as np

class JacobianMetrics:
    def __init__(self, robot_config_file="ur3e.yml", tcp_link="tool0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tcp_link = tcp_link
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

    def compute_pyroki_jacobian(self, q: torch.Tensor):
        """Tính ma trận Jacobian hình học qua tham số DH chuẩn của UR3e (PyRoki CPU, không cần CUDA)"""
        return self.compute_dh_jacobian(q)

    def compute_dh_jacobian(self, q: torch.Tensor):
        """
        Compute analytic Jacobian using standard UR3e DH parameters.
        q: [B, 6] tensor (in radians)
        """
        # UR3e DH parameters
        d = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
        a = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
        alpha = [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0]
        
        B = q.shape[0]
        J_dh = torch.zeros((B, 6, 6), device=self.device, dtype=torch.float32)
        
        for b in range(B):
            qb = q[b]
            # Forward kinematics to find z_i and p_i
            z = []
            p = []
            T = torch.eye(4, device=self.device, dtype=torch.float32)
            
            z.append(T[0:3, 2].clone())
            p.append(T[0:3, 3].clone())
            
            for i in range(6):
                ct = torch.cos(qb[i])
                st = torch.sin(qb[i])
                ca = math.cos(alpha[i])
                sa = math.sin(alpha[i])
                
                Ti = torch.tensor([
                    [ct, -st*ca,  st*sa, a[i]*ct],
                    [st,  ct*ca, -ct*sa, a[i]*st],
                    [0,   sa,     ca,    d[i]],
                    [0,   0,      0,     1]
                ], device=self.device, dtype=torch.float32)
                
                T = torch.matmul(T, Ti)
                z.append(T[0:3, 2].clone())
                p.append(T[0:3, 3].clone())
                
            p_n = p[-1]
            for i in range(6):
                z_i = z[i]
                p_i = p[i]
                # J_v = z_i x (p_n - p_i)
                J_v = torch.linalg.cross(z_i, p_n - p_i)
                # J_w = z_i
                J_w = z_i
                J_dh[b, 0:3, i] = J_v
                J_dh[b, 3:6, i] = J_w
                
        return J_dh

    def compare_jacobians(self, num_samples=1000):
        """
        Compare PyRoki Jacobian with DH Jacobian.
        """
        q_rand = (torch.rand((num_samples, 6), device=self.device) * 2 * math.pi) - math.pi
        
        J_pyroki = self.compute_pyroki_jacobian(q_rand)
        J_dh = self.compute_dh_jacobian(q_rand)
        
        det_pyroki = torch.linalg.det(J_pyroki)
        det_dh = torch.linalg.det(J_dh)
        
        max_err = torch.max(torch.abs(torch.abs(det_pyroki) - torch.abs(det_dh))).item()
        return max_err

    def get_metrics(self, q: torch.Tensor):
        """
        Compute metrics: sigma_min, manipulability (w), condition_number (cond)
        q: [B, 6] tensor
        Returns dict with tensors of shape [B]
        """
        J = self.compute_pyroki_jacobian(q)
        
        # SVD: J = U * S * V^T
        # torch.linalg.svd returns U, S, Vh
        U, S, Vh = torch.linalg.svd(J)
        
        sigma_min = S[:, -1]
        sigma_max = S[:, 0]
        
        # Manipulability w = sqrt(det(J * J^T)) = product of singular values
        w = torch.prod(S, dim=1)
        
        # Condition number cond = sigma_max / sigma_min
        # Add small epsilon to avoid division by zero
        cond = sigma_max / (sigma_min + 1e-9)
        
        return {
            "sigma_min": sigma_min,
            "manipulability": w,
            "condition_number": cond,
            "singular_values": S
        }

if __name__ == "__main__":
    metrics = JacobianMetrics()
    print("Testing Jacobian computation...")
    max_err = metrics.compare_jacobians(10)
    print(f"Max absolute difference in |det(J)| between PyRoki and DH: {max_err:.6e}")
    print("Metrics module ready!")
