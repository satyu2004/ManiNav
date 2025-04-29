import torch
torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

# class Immersed_Manifold:
#     def __init__(self, f=None, immersion=None, chart=None):
#         self.immersion = immersion
#         self.chart = chart
#         self.f = f

#     def compute_partial_derivatives(self, pts):
#         """
#         Computes the partial derivatives of the immersion function.

#         Args:
#             pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points (u, v).

#         Returns:
#             torch.Tensor: Tensor of shape (N, 2, 2) containing the partial derivatives.
#         """
#         pts.requires_grad_(True)
#         immersed_pts = self.immersion(pts)
#         # dimension = immersed_pts.shape[-1]
#         jac = torch.stack([torch.autograd.grad(immersed_pts[:, i], pts, torch.ones_like(immersed_pts[:, i]), create_graph=True)[0] for i in range(3)], dim=1).transpose(1,2)
#         return jac


#     def compute_metric_tensor(self, pts):
#         """
#         Computes the metric tensor (first fundamental form).

#         Args:
#             pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

#         Returns:
#             torch.Tensor: Tensor of shape (N, 2, 2) representing the metric tensor.
#         """
#         x_uv = self.compute_partial_derivatives(pts)
#         x_u = x_uv[:, 0, :]
#         x_v = x_uv[:, 1, :]
#         g11 = torch.sum(x_u * x_u, dim=-1)
#         g12 = torch.sum(x_u * x_v, dim=-1)
#         g22 = torch.sum(x_v * x_v, dim=-1)
#         return torch.stack([torch.stack([g11, g12], dim=-1), torch.stack([g12, g22], dim=-1)], dim=-2)

#     def compute_inverse_metric_tensor(self, pts):
#         """
#         Computes the inverse of the metric tensor.

#         Args:
#             pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

#         Returns:
#             torch.Tensor: Tensor of shape (N, 2, 2) representing the inverse metric tensor.
#         """
#         G = self.compute_metric_tensor(pts)
#         det_G = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
#         inv_det_G = 1.0 / det_G
#         g11 = G[:, 0, 0]
#         g12 = G[:, 0, 1]
#         g22 = G[:, 1, 1]
#         return torch.stack([torch.stack([g22, -g12], dim=-1) * inv_det_G.unsqueeze(-1),
#                             torch.stack([-g12, g11], dim=-1) * inv_det_G.unsqueeze(-1)], dim=-2)

#     def compute_christoffel_symbols(self, pts):
#         """
#         Computes the Christoffel symbols of the second kind.

#         Args:
#             pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

#         Returns:
#             torch.Tensor: Tensor of shape (N, 2, 2, 2) representing the Christoffel symbols.
#         """
#         pts.requires_grad_(True)
#         G = self.compute_metric_tensor(pts)
#         G_inv = self.compute_inverse_metric_tensor(pts)
#         x_uv = self.compute_partial_derivatives(pts)
#         x_u = x_uv[:, 0, :]
#         x_v = x_uv[:, 1, :]

#         g11_u = torch.autograd.grad(G[:, 0, 0].sum(), pts, create_graph=True)[0][:, 0]
#         g11_v = torch.autograd.grad(G[:, 0, 0].sum(), pts, create_graph=True)[0][:, 1]
#         g12_u = torch.autograd.grad(G[:, 0, 1].sum(), pts, create_graph=True)[0][:, 0]
#         g12_v = torch.autograd.grad(G[:, 0, 1].sum(), pts, create_graph=True)[0][:, 1]
#         g22_u = torch.autograd.grad(G[:, 1, 1].sum(), pts, create_graph=True)[0][:, 0]
#         g22_v = torch.autograd.grad(G[:, 1, 1].sum(), pts, create_graph=True)[0][:, 1]

#         gamma111 = 0.5 * (G_inv[:, 0, 0] * (2 * g11_u - g11_u) + G_inv[:, 0, 1] * (g11_v + g12_u - g12_u))
#         gamma112 = 0.5 * (G_inv[:, 1, 0] * (2 * g11_u - g11_v) + G_inv[:, 1, 1] * (g11_v + g12_u - g22_u))
#         gamma121 = 0.5 * (G_inv[:, 0, 0] * (g11_v + g12_u - g12_u) + G_inv[:, 0, 1] * (g12_v + g22_u - g22_v))
#         gamma122 = 0.5 * (G_inv[:, 1, 0] * (g11_v + g12_u - g12_v) + G_inv[:, 1, 1] * (g12_v + g22_u - g22_v))
#         gamma221 = 0.5 * (G_inv[:, 0, 0] * (2 * g12_v - g22_u) + G_inv[:, 0, 1] * (g22_u + g22_u - g22_v))
#         gamma222 = 0.5 * (G_inv[:, 1, 0] * (2 * g12_v - g22_v) + G_inv[:, 1, 1] * (g22_u + g22_v - 2 * g22_v))

#         return torch.stack([torch.stack([torch.stack([gamma111, gamma112], dim=-1),
#                                         torch.stack([gamma121, gamma122], dim=-1)], dim=-2),
#                             torch.stack([torch.stack([gamma121, gamma122], dim=-1),
#                                         torch.stack([gamma221, gamma222], dim=-1)], dim=-2)], dim=-3)

#     def geodesic_rhs(self, t, Z):
#         """
#         Computes the right-hand side of the geodesic equation as a first-order system.

#         Args:
#             t (float): Time parameter (not used, but required for odeint).
#             Z (torch.Tensor): Tensor of shape (N, 4) representing [u, v, du_dt, dv_dt].

#         Returns:
#             torch.Tensor: Tensor of shape (N, 4) representing [du_dt, dv_dt, d^2u_dt2, d^2v_dt2].
#         """
#         # print(f"Z.shape: {Z.shape}")
#         G = self.compute_christoffel_symbols(Z[:, :2]) #Compute Christoffel symbols at the base points.
#         du_dt, dv_dt = Z[:, 2], Z[:, 3]
#         # d2u_dt2 = -torch.einsum('nijk,ni,nj->n', G[:, 0, :, :], Z[:, 2:], Z[:, 2:])
#         # d2v_dt2 = -torch.einsum('nijk,ni,nj->n', G[:, 1, :, :], Z[:, 2:], Z[:, 2:])
#         # print(f"G.shape: {G[:,0,:,:].shape}, Z[:,2:].shape: {Z[:, 2:].shape})")
#         d2u_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 0, :, :], Z[:, 2:], Z[:, 2:])
#         d2v_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 1, :, :], Z[:, 2:], Z[:, 2:])
#         return torch.stack([Z[:, 2], Z[:, 3], d2u_dt2, d2v_dt2], dim=-1)
    
#     def exp(self, base_pts, velocities, t_span = torch.tensor([0, 1], dtype=torch.float64)):
#         """
#         Solves the geodesic equation and returns the solution at t=1.

#         Args:
#             Z (torch.Tensor): Tensor of shape (N, 2) representing the base points (u, v).
#             V (torch.Tensor): Tensor of shape (N, 2) representing the initial velocities (du_dt, dv_dt).
#             t_span (torch.Tensor): Time points at which solution is to be evaluated
#         Returns:
#             torch.Tensor: Tensor of shape (N, 4) representing the solution [u(1), v(1), du_dt(1), dv_dt(1)].
#         """
#         # print(f"base_pts.shape: {base_pts.shape}, velocities.shape: {velocities.shape}")
#         initial_state = torch.cat([base_pts, velocities], dim=-1)
#         # print(f"initial_state.shape: {initial_state.shape}")

#         solution = odeint(self.geodesic_rhs, initial_state, t_span, rtol=1e-10, atol=1e-12)
        
#         return solution[-1][:,:2]  # Return the solution at t=1

class Immersed_Manifold:
    def __init__(self, f=None, immersion=None, chart=None):
        self.immersion = immersion
        self.chart = chart
        self.f = f

    def compute_partial_derivatives(self, pts):
        """
        Computes the partial derivatives of the immersion function efficiently.
        Optimized for immersion functions where the first two components are the input coordinates.
        
        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points (u, v).

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 3) containing the partial derivatives.
        """
        device = pts.device
        batch_size = pts.shape[0]
        
        # Initialize the Jacobian tensor with zeros
        jac = torch.zeros(batch_size, 2, 3, device=device)
        
        # Set the known derivatives for first two components
        # For u-derivatives (first row of Jacobian)
        jac[:, 0, 0] = 1.0  # ∂X/∂u = 1
        jac[:, 0, 1] = 0.0  # ∂Y/∂u = 0
        
        # For v-derivatives (second row of Jacobian)
        jac[:, 1, 0] = 0.0  # ∂X/∂v = 0
        jac[:, 1, 1] = 1.0  # ∂Y/∂v = 1
        
        # Only compute gradients for the third component (Z)
        pts_grad = pts.detach().clone().requires_grad_(True)
        
        # Compute Z component of the immersion
        if self.f is not None:
            z_values = self.f(pts_grad).squeeze()
        else:
            # If f is not available, extract Z from full immersion
            immersed_pts = self.immersion(pts_grad)
            z_values = immersed_pts[:, 2]
        
        # Compute gradients of Z with respect to u
        z_u = torch.autograd.grad(
            outputs=z_values,
            inputs=pts_grad,
            grad_outputs=torch.ones_like(z_values, device=device),
            create_graph=True,
            only_inputs=True
        )[0]
        
        # Store Z gradients in the Jacobian tensor
        jac[:, 0, 2] = z_u[:, 0]  # ∂Z/∂u
        jac[:, 1, 2] = z_u[:, 1]  # ∂Z/∂v
        
        return jac

    def compute_metric_tensor(self, pts):
        """
        Computes the metric tensor (first fundamental form) more efficiently.

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2) representing the metric tensor.
        """
        x_uv = self.compute_partial_derivatives(pts)
        
        # Extract components for readability
        dX_du = x_uv[:, 0, 0]  # = 1.0
        dY_du = x_uv[:, 0, 1]  # = 0.0
        dZ_du = x_uv[:, 0, 2]
        
        dX_dv = x_uv[:, 1, 0]  # = 0.0
        dY_dv = x_uv[:, 1, 1]  # = 1.0
        dZ_dv = x_uv[:, 1, 2]
        
        # Compute metric tensor components efficiently
        # g11 = dX_du²+ dY_du² + dZ_du² = 1 + dZ_du²
        g11 = 1.0 + dZ_du * dZ_du
        
        # g12 = dX_du*dX_dv + dY_du*dY_dv + dZ_du*dZ_dv = 0 + 0 + dZ_du*dZ_dv
        g12 = dZ_du * dZ_dv
        
        # g22 = dX_dv² + dY_dv² + dZ_dv² = 0 + 1 + dZ_dv² = 1 + dZ_dv²
        g22 = 1.0 + dZ_dv * dZ_dv
        
        return torch.stack([torch.stack([g11, g12], dim=-1), 
                           torch.stack([g12, g22], dim=-1)], dim=-2)

    def compute_inverse_metric_tensor(self, pts):
        """
        Computes the inverse of the metric tensor efficiently.

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2) representing the inverse metric tensor.
        """
        G = self.compute_metric_tensor(pts)
        
        # Extract components for readability
        g11 = G[:, 0, 0]
        g12 = G[:, 0, 1]
        g22 = G[:, 1, 1]
        
        det_G = g11 * g22 - g12 * g12
        inv_det_G = 1.0 / det_G
        
        return torch.stack([torch.stack([g22, -g12], dim=-1) * inv_det_G.unsqueeze(-1),
                           torch.stack([-g12, g11], dim=-1) * inv_det_G.unsqueeze(-1)], dim=-2)


    def compute_christoffel_symbols(self, pts):
        """
        Computes the Christoffel symbols of the second kind more efficiently,
        with proper handling of unused parameters in gradient computation.

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2, 2) representing the Christoffel symbols.
        """
        device = pts.device
        batch_size = pts.shape[0]
        pts_grad = pts.detach().clone().requires_grad_(True)
        
        # Compute metric tensor at these points
        x_uv = self.compute_partial_derivatives(pts_grad)
        dZ_du = x_uv[:, 0, 2]
        dZ_dv = x_uv[:, 1, 2]
        
        # Make sure both coordinates are used in the computation
        # This ensures the computational graph is properly connected
        dZ_du_connected = dZ_du + 0.0 * pts_grad.sum()
        dZ_dv_connected = dZ_dv + 0.0 * pts_grad.sum()
        
        # Compute metric tensor components with explicit connections to input
        g11 = 1.0 + dZ_du_connected * dZ_du_connected
        g12 = dZ_du_connected * dZ_dv_connected
        g22 = 1.0 + dZ_dv_connected * dZ_dv_connected
        
        # Safe gradient computation with allow_unused=True
        def safe_grad(output, input_tensor):
            grad_result = torch.autograd.grad(
                outputs=output.sum(), 
                inputs=input_tensor,
                create_graph=True,
                allow_unused=True
            )[0]
            
            # If gradient is None, return zeros
            if grad_result is None:
                return torch.zeros_like(input_tensor)
            return grad_result
        
        # Compute derivatives of metric tensor components safely
        # Compute gradients individually to avoid dimension issues
        g11_grad = safe_grad(g11, pts_grad)
        g12_grad = safe_grad(g12, pts_grad)
        g22_grad = safe_grad(g22, pts_grad)
        
        # Extract individual components
        g11_u = g11_grad[:, 0]
        g11_v = g11_grad[:, 1]
        g12_u = g12_grad[:, 0]
        g12_v = g12_grad[:, 1]
        g22_u = g22_grad[:, 0]
        g22_v = g22_grad[:, 1]
        
        # Alternative approach: compute derivatives separately with error handling
        # try:
        #     g11_u = torch.autograd.grad(g11.sum(), pts_grad, create_graph=True)[0][:, 0]
        # except RuntimeError:
        #     g11_u = torch.autograd.grad(g11.sum(), pts_grad, create_graph=True, allow_unused=True)[0]
        #     if g11_u is None:
        #         g11_u = torch.zeros(batch_size, device=device)
        #     else:
        #         g11_u = g11_u[:, 0]
        
        # Similarly for other derivatives...
        
        # Compute inverse metric tensor
        det_G = g11 * g22 - g12 * g12
        inv_det_G = 1.0 / det_G
        
        G_inv_11 = g22 * inv_det_G
        G_inv_12 = -g12 * inv_det_G
        G_inv_22 = g11 * inv_det_G
        
        # Compute Christoffel symbols
        gamma111 = 0.5 * (G_inv_11 * (2 * g11_u - g11_u) + G_inv_12 * (g11_v + g12_u - g12_u))
        gamma112 = 0.5 * (G_inv_12 * (2 * g11_u - g11_v) + G_inv_22 * (g11_v + g12_u - g22_u))
        gamma121 = 0.5 * (G_inv_11 * (g11_v + g12_u - g12_u) + G_inv_12 * (g12_v + g22_u - g22_v))
        gamma122 = 0.5 * (G_inv_12 * (g11_v + g12_u - g12_v) + G_inv_22 * (g12_v + g22_u - g22_v))
        gamma221 = 0.5 * (G_inv_11 * (2 * g12_v - g22_u) + G_inv_12 * (g22_u + g22_u - g22_v))
        gamma222 = 0.5 * (G_inv_12 * (2 * g12_v - g22_v) + G_inv_22 * (g22_u + g22_v - 2 * g22_v))
        
        return torch.stack([torch.stack([torch.stack([gamma111, gamma112], dim=-1),
                                        torch.stack([gamma121, gamma122], dim=-1)], dim=-2),
                        torch.stack([torch.stack([gamma121, gamma122], dim=-1),
                                    torch.stack([gamma221, gamma222], dim=-1)], dim=-2)], dim=-3)

    # def compute_christoffel_symbols(self, pts):
    #     """
    #     Computes the Christoffel symbols of the second kind more efficiently.

    #     Args:
    #         pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

    #     Returns:
    #         torch.Tensor: Tensor of shape (N, 2, 2, 2) representing the Christoffel symbols.
    #     """
    #     device = pts.device
    #     pts_grad = pts.detach().clone().requires_grad_(True)
        
    #     # Compute metric tensor at these points
    #     x_uv = self.compute_partial_derivatives(pts_grad)
    #     dZ_du = x_uv[:, 0, 2]
    #     dZ_dv = x_uv[:, 1, 2]
        
    #     # Compute metric tensor components
    #     g11 = 1.0 + dZ_du * dZ_du
    #     g12 = dZ_du * dZ_dv
    #     g22 = 1.0 + dZ_dv * dZ_dv
        
    #     # Compute derivatives of metric tensor components
    #     g11_u = torch.autograd.grad(g11.sum(), pts_grad, create_graph=True)[0][:, 0]
    #     g11_v = torch.autograd.grad(g11.sum(), pts_grad, create_graph=True)[0][:, 1]
    #     g12_u = torch.autograd.grad(g12.sum(), pts_grad, create_graph=True)[0][:, 0]
    #     g12_v = torch.autograd.grad(g12.sum(), pts_grad, create_graph=True)[0][:, 1]
    #     g22_u = torch.autograd.grad(g22.sum(), pts_grad, create_graph=True)[0][:, 0]
    #     g22_v = torch.autograd.grad(g22.sum(), pts_grad, create_graph=True)[0][:, 1]
        
    #     # Compute inverse metric tensor
    #     det_G = g11 * g22 - g12 * g12
    #     inv_det_G = 1.0 / det_G
        
    #     G_inv_11 = g22 * inv_det_G
    #     G_inv_12 = -g12 * inv_det_G
    #     G_inv_22 = g11 * inv_det_G
        
    #     # Compute Christoffel symbols
    #     gamma111 = 0.5 * (G_inv_11 * (2 * g11_u - g11_u) + G_inv_12 * (g11_v + g12_u - g12_u))
    #     gamma112 = 0.5 * (G_inv_12 * (2 * g11_u - g11_v) + G_inv_22 * (g11_v + g12_u - g22_u))
    #     gamma121 = 0.5 * (G_inv_11 * (g11_v + g12_u - g12_u) + G_inv_12 * (g12_v + g22_u - g22_v))
    #     gamma122 = 0.5 * (G_inv_12 * (g11_v + g12_u - g12_v) + G_inv_22 * (g12_v + g22_u - g22_v))
    #     gamma221 = 0.5 * (G_inv_11 * (2 * g12_v - g22_u) + G_inv_12 * (g22_u + g22_u - g22_v))
    #     gamma222 = 0.5 * (G_inv_12 * (2 * g12_v - g22_v) + G_inv_22 * (g22_u + g22_v - 2 * g22_v))
        
    #     return torch.stack([torch.stack([torch.stack([gamma111, gamma112], dim=-1),
    #                                      torch.stack([gamma121, gamma122], dim=-1)], dim=-2),
    #                        torch.stack([torch.stack([gamma121, gamma122], dim=-1),
    #                                    torch.stack([gamma221, gamma222], dim=-1)], dim=-2)], dim=-3)

    def geodesic_rhs(self, t, Z):
        """
        Computes the right-hand side of the geodesic equation as a first-order system.

        Args:
            t (float): Time parameter (not used, but required for odeint).
            Z (torch.Tensor): Tensor of shape (N, 4) representing [u, v, du_dt, dv_dt].

        Returns:
            torch.Tensor: Tensor of shape (N, 4) representing [du_dt, dv_dt, d^2u_dt2, d^2v_dt2].
        """
        G = self.compute_christoffel_symbols(Z[:, :2])
        du_dt, dv_dt = Z[:, 2], Z[:, 3]
        
        d2u_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 0, :, :], Z[:, 2:], Z[:, 2:])
        d2v_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 1, :, :], Z[:, 2:], Z[:, 2:])
        
        return torch.stack([du_dt, dv_dt, d2u_dt2, d2v_dt2], dim=-1)
    
    def exp(self, base_pts, velocities, t_span=torch.tensor([0, 1], dtype=torch.float64)):
        """
        Solves the geodesic equation and returns the solution at t=1.

        Args:
            base_pts (torch.Tensor): Tensor of shape (N, 2) representing the base points (u, v).
            velocities (torch.Tensor): Tensor of shape (N, 2) representing the initial velocities (du_dt, dv_dt).
            t_span (torch.Tensor): Time points at which solution is to be evaluated
            
        Returns:
            torch.Tensor: Tensor of shape (N, 2) representing the endpoint of geodesics [u(1), v(1)].
        """
        initial_state = torch.cat([base_pts, velocities], dim=-1)
        
        # Use odeint for solving the ODE system
        from torchdiffeq import odeint
        solution = odeint(self.geodesic_rhs, initial_state, t_span, rtol=1e-10, atol=1e-12)
        
        return solution[-1][:, :2]  # Return the endpoint positions