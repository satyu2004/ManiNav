import torch
torch.set_default_dtype(torch.float64)
from geodesic_solver import Immersed_Manifold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def model_tanh(hidden_dims: list):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, hidden_dims[0]),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dims[1], 1)
    )
    return model

f = model_tanh([32, 32]).to(device)
# f.load_state_dict(torch.load('Recon_Surface\\neural_surface.pth'))
f.load_state_dict(torch.load('neural_surface.pth'))


class Surface:
    # def __init__(self, a, c):
    #     self.a = a
    #     self.c = c
    def immersion(self, point):
        """Immersion. Input is a batch of 2-tupes in the unit square."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = point[..., 0]
        Y = point[..., 1]
        Z = f(point).squeeze(-1)  

        return torch.stack(
            [X, Y, Z],
            axis=-1,
        ).to(device)

    # def chart(X):
    #   x, y, z = X[:,0], X[:,1], X[:,2]
    #   x_coords = x/(1-z)
    #   y_coords = y/(1-z)
    #   return torch.stack((x_coords, y_coords), dim=1)


    # def jacobian_matrix_batch(self, pts):
    #     """
    #     Calculates the Jacobian matrix for a batch of input points.

    #     Args:
    #     xy_tensor: A tensor of shape (batch_size, 2) containing (x, y) pairs.

    #     Returns:
    #     A tensor of shape (batch_size, 3, 2) containing the Jacobian matrices for each input point.
    #     # """

    #     pts.requires_grad_(True)
    #     immersed_pts = self.immersion(pts)
    #     # dimension = immersed_pts.shape[-1]
    #     jac = torch.stack([torch.autograd.grad(immersed_pts[:, i], pts, torch.ones_like(immersed_pts[:, i]), create_graph=True)[0] for i in range(3)], dim=1)
    #     # print(f"Jacobian shape: {jac.shape}")
    #     return jac

    def jacobian_matrix_batch(self, pts):
            """
            Efficiently calculates the Jacobian matrix for a batch of input points,
            leveraging the specific structure of our immersion function.
            
            Since we know that:
            - X = pts[..., 0] => ∂X/∂x = 1, ∂X/∂y = 0
            - Y = pts[..., 1] => ∂Y/∂x = 0, ∂Y/∂y = 1
            - Only Z needs gradient computation
            
            Args:
                pts: A tensor of shape (batch_size, 2) containing (x, y) pairs.
                
            Returns:
                A tensor of shape (batch_size, 3, 2) containing the Jacobian matrices.
            """
            # Ensure we're on the right device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pts = pts.to(device)
            batch_size = pts.shape[0]
            
            # Create the Jacobian tensor
            jacobian = torch.zeros(batch_size, 3, 2, device=device)
            
            # Set the known derivatives for X and Y components
            # For the X component (immersed_pts[:, 0]):
            jacobian[:, 0, 0] = 1.0  # ∂X/∂x = 1
            jacobian[:, 0, 1] = 0.0  # ∂X/∂y = 0
            
            # For the Y component (immersed_pts[:, 1]):
            jacobian[:, 1, 0] = 0.0  # ∂Y/∂x = 0
            jacobian[:, 1, 1] = 1.0  # ∂Y/∂y = 1
            
            # Only compute gradients for the Z component
            # Clone and set requires_grad to avoid modifying the input tensor
            pts_grad = pts.detach().clone().requires_grad_(True)
            
            # Compute only the Z component
            z_values = f(pts_grad).squeeze()
            
            # Compute gradients of Z with respect to inputs
            z_grad = torch.autograd.grad(
                outputs=z_values,
                inputs=pts_grad,
                grad_outputs=torch.ones_like(z_values, device=device),
                create_graph=True,
                only_inputs=True
            )[0]
            
            # Store Z gradients in the Jacobian tensor
            jacobian[:, 2, :] = z_grad
            
            return jacobian

    # def jacobian_matrix_batch(self, pts):
    #     """
    #     Calculates the Jacobian matrix for a batch of input points.

    #     Args:
    #     pts: A tensor of shape (batch_size, 2) containing (x, y) pairs.

    #     Returns:
    #     A tensor of shape (batch_size, 3, 2) containing the Jacobian matrices for each input point.
    #     """
    #     pts.requires_grad_(True)
    #     def immersion_fn(p):
    #         return self.immersion(p).view(-1)  # Flatten for jacobian computation

    #     jac = torch.autograd.functional.jacobian(immersion_fn, pts, create_graph=True)
    #     jac = jac.view(pts.shape[0], 3, 2)  # Reshape to (batch_size, 3, 2)
    #     print(f"Jacobian shape: {jac.shape}")
    #     return jac


    def exp(self, base_pts, velocities):
        "Computes exponential maps"
        immersed_manifold = Immersed_Manifold(f=f)
        return immersed_manifold.exp(base_pts, velocities)


 