import torch
torch.set_default_dtype(torch.float64)
from geodesic_solver import Immersed_Manifold

class Torus:
    def __init__(self, a, c):
        self.a = a
        self.c = c
    def immersion(self, point):
        """Immersion."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = point[..., 0]
        y = point[..., 1]
        X = (self.c + self.a * torch.cos(y)) * torch.cos(x)
        Y = (self.c + self.a * torch.cos(y)) * torch.sin(x)
        Z = self.a * torch.sin(y)
        return torch.stack(
            [X, Y, Z],
            axis=-1,
        ).to(device)

    # def chart(X):
    #   x, y, z = X[:,0], X[:,1], X[:,2]
    #   x_coords = x/(1-z)
    #   y_coords = y/(1-z)
    #   return torch.stack((x_coords, y_coords), dim=1)


    def jacobian_matrix_batch(self, pts):
        """
        Calculates the Jacobian matrix for a batch of input points.

        Args:
        xy_tensor: A tensor of shape (batch_size, 2) containing (x, y) pairs.

        Returns:
        A tensor of shape (batch_size, 3, 2) containing the Jacobian matrices for each input point.
        """
        a,c = self.a, self.c

        x, y = pts[..., 0], pts[..., 1]
        # denominator = (x**2 + y**2 + 1)**2
        first_row = torch.stack([-(c + a * torch.cos(y)) * torch.sin(x) , -a * torch.sin(y) * torch.cos(x)],dim=-1)
        second_row = torch.stack([(c + a * torch.cos(y)) * torch.cos(x) , -a * torch.sin(y) * torch.sin(x)],dim=-1)
        third_row = torch.stack([torch.zeros_like(x) , a * torch.cos(y) ],dim=-1)

        jacobians = torch.stack([first_row, second_row, third_row], dim=1)
        return jacobians
    
    def exp(self, base_pts, velocities):
        "Computes exponential maps"
        immersed_manifold = Immersed_Manifold(self.immersion)
        return immersed_manifold.exp(base_pts, velocities)


    # def parallel_transport(x, v, angle=None):
    #     """
    #     Rotates a tangent vector v at point x on the unit sphere.
    #     This performs parallel transport of the tangent vector.

    #     Args:
    #         x (torch.Tensor): Unit vector on the sphere (|x| = 1)
    #         v (torch.Tensor): Tangent vector at x (so x⋅v = 0)
    #         angle (float, optional): Rotation angle. If None, uses |v| as the angle

    #     Returns:
    #         tuple: (rotated_x, rotated_v) - both the rotated point and its tangent vector
    #     """

    #     # Ensure inputs are PyTorch tensors
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.tensor(x, dtype=torch.float32)
    #     if not isinstance(v, torch.Tensor):
    #         v = torch.tensor(v, dtype=torch.float32)

    #     # Normalize x to ensure it's on the unit sphere
    #     x = x / torch.norm(x, dim=1, keepdim=True)

    #     # Project v onto the tangent space of x if it's not already tangent
    #     dot_product = (x*v).sum(dim=1, keepdim=True)
    #     v_tangent = v - dot_product * x

    #     # Calculate the rotation angle (norm of tangent vector)
    #     if angle is None:
    #         angle = torch.norm(v_tangent, dim=1,keepdim=True)


    #     # If tangent vector is zero (or nearly zero), no rotation needed
    #     if torch.norm(v_tangent) < 1e-8:
    #         return x, v_tangent

    #     # Normalize v_tangent to get direction (preserving magnitude for later)
    #     v_norm = torch.norm(v_tangent, dim=1, keepdim=True)
    #     v_dir = v_tangent / v_norm

    #     # Calculate the rotation axis a = x × v_dir (cross product)
    #     a = torch.cross(x, v_dir, dim=1)
    #     a = a / torch.norm(a, dim=1, keepdim=True)  # Normalize rotation axis

    #     # Use Rodrigues' rotation formula to rotate x
    #     cos_angle = torch.cos(angle)
    #     sin_angle = torch.sin(angle)

    #     # Rotate the position vector x
    #     cross = torch.cross(a, x, dim=1)
    #     rotated_x = cos_angle * x  + sin_angle * cross


    #     # Rotate the tangent vector v using the same rotation
    #     # For a tangent vector, we need to maintain tangency at the new point

    #     # First rotate v as if it were a free vector
    #     rotated_v_free = v_tangent * cos_angle + torch.cross(a, v_tangent, dim=1) * sin_angle

    #     # Then project back to the tangent space at the new point
    #     dot_product = (rotated_x*rotated_v_free).sum(dim=1, keepdim=True)
    #     rotated_v = rotated_v_free - dot_product * rotated_x

    #     # Preserve the original magnitude of v
    #     rotated_v = rotated_v * (v_norm / torch.norm(rotated_v, dim=1, keepdim=True))

    #     return rotated_x, rotated_v