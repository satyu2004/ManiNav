import torch
torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

class Immersed_Manifold:
    def __init__(self, immersion, chart=None):
        self.immersion = immersion
        self.chart = chart

    def compute_partial_derivatives(self, pts):
        """
        Computes the partial derivatives of the immersion function.

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points (u, v).

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2) containing the partial derivatives.
        """
        pts.requires_grad_(True)
        immersed_pts = self.immersion(pts)
        # dimension = immersed_pts.shape[-1]
        jac = torch.stack([torch.autograd.grad(immersed_pts[:, i], pts, torch.ones_like(immersed_pts[:, i]), create_graph=True)[0] for i in range(3)], dim=1).transpose(1,2)
        return jac

    def compute_metric_tensor(self, pts):
        """
        Computes the metric tensor (first fundamental form).

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2) representing the metric tensor.
        """
        x_uv = self.compute_partial_derivatives(pts)
        x_u = x_uv[:, 0, :]
        x_v = x_uv[:, 1, :]
        g11 = torch.sum(x_u * x_u, dim=-1)
        g12 = torch.sum(x_u * x_v, dim=-1)
        g22 = torch.sum(x_v * x_v, dim=-1)
        return torch.stack([torch.stack([g11, g12], dim=-1), torch.stack([g12, g22], dim=-1)], dim=-2)

    def compute_inverse_metric_tensor(self, pts):
        """
        Computes the inverse of the metric tensor.

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2) representing the inverse metric tensor.
        """
        G = self.compute_metric_tensor(pts)
        det_G = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        inv_det_G = 1.0 / det_G
        g11 = G[:, 0, 0]
        g12 = G[:, 0, 1]
        g22 = G[:, 1, 1]
        return torch.stack([torch.stack([g22, -g12], dim=-1) * inv_det_G.unsqueeze(-1),
                            torch.stack([-g12, g11], dim=-1) * inv_det_G.unsqueeze(-1)], dim=-2)

    def compute_christoffel_symbols(self, pts):
        """
        Computes the Christoffel symbols of the second kind.

        Args:
            pts (torch.Tensor): Tensor of shape (N, 2) representing the parameter space points.

        Returns:
            torch.Tensor: Tensor of shape (N, 2, 2, 2) representing the Christoffel symbols.
        """
        pts.requires_grad_(True)
        G = self.compute_metric_tensor(pts)
        G_inv = self.compute_inverse_metric_tensor(pts)
        x_uv = self.compute_partial_derivatives(pts)
        x_u = x_uv[:, 0, :]
        x_v = x_uv[:, 1, :]

        g11_u = torch.autograd.grad(G[:, 0, 0].sum(), pts, create_graph=True)[0][:, 0]
        g11_v = torch.autograd.grad(G[:, 0, 0].sum(), pts, create_graph=True)[0][:, 1]
        g12_u = torch.autograd.grad(G[:, 0, 1].sum(), pts, create_graph=True)[0][:, 0]
        g12_v = torch.autograd.grad(G[:, 0, 1].sum(), pts, create_graph=True)[0][:, 1]
        g22_u = torch.autograd.grad(G[:, 1, 1].sum(), pts, create_graph=True)[0][:, 0]
        g22_v = torch.autograd.grad(G[:, 1, 1].sum(), pts, create_graph=True)[0][:, 1]

        gamma111 = 0.5 * (G_inv[:, 0, 0] * (2 * g11_u - g11_u) + G_inv[:, 0, 1] * (g11_v + g12_u - g12_u))
        gamma112 = 0.5 * (G_inv[:, 1, 0] * (2 * g11_u - g11_v) + G_inv[:, 1, 1] * (g11_v + g12_u - g22_u))
        gamma121 = 0.5 * (G_inv[:, 0, 0] * (g11_v + g12_u - g12_u) + G_inv[:, 0, 1] * (g12_v + g22_u - g22_v))
        gamma122 = 0.5 * (G_inv[:, 1, 0] * (g11_v + g12_u - g12_v) + G_inv[:, 1, 1] * (g12_v + g22_u - g22_v))
        gamma221 = 0.5 * (G_inv[:, 0, 0] * (2 * g12_v - g22_u) + G_inv[:, 0, 1] * (g22_u + g22_u - g22_v))
        gamma222 = 0.5 * (G_inv[:, 1, 0] * (2 * g12_v - g22_v) + G_inv[:, 1, 1] * (g22_u + g22_v - 2 * g22_v))

        return torch.stack([torch.stack([torch.stack([gamma111, gamma112], dim=-1),
                                        torch.stack([gamma121, gamma122], dim=-1)], dim=-2),
                            torch.stack([torch.stack([gamma121, gamma122], dim=-1),
                                        torch.stack([gamma221, gamma222], dim=-1)], dim=-2)], dim=-3)

    def geodesic_rhs(self, t, Z):
        """
        Computes the right-hand side of the geodesic equation as a first-order system.

        Args:
            t (float): Time parameter (not used, but required for odeint).
            Z (torch.Tensor): Tensor of shape (N, 4) representing [u, v, du_dt, dv_dt].

        Returns:
            torch.Tensor: Tensor of shape (N, 4) representing [du_dt, dv_dt, d^2u_dt2, d^2v_dt2].
        """
        # print(f"Z.shape: {Z.shape}")
        G = self.compute_christoffel_symbols(Z[:, :2]) #Compute Christoffel symbols at the base points.
        du_dt, dv_dt = Z[:, 2], Z[:, 3]
        # d2u_dt2 = -torch.einsum('nijk,ni,nj->n', G[:, 0, :, :], Z[:, 2:], Z[:, 2:])
        # d2v_dt2 = -torch.einsum('nijk,ni,nj->n', G[:, 1, :, :], Z[:, 2:], Z[:, 2:])
        # print(f"G.shape: {G[:,0,:,:].shape}, Z[:,2:].shape: {Z[:, 2:].shape})")
        d2u_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 0, :, :], Z[:, 2:], Z[:, 2:])
        d2v_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 1, :, :], Z[:, 2:], Z[:, 2:])
        return torch.stack([Z[:, 2], Z[:, 3], d2u_dt2, d2v_dt2], dim=-1)
    
    def exp(self, base_pts, velocities, t_span = torch.tensor([0, 1], dtype=torch.float64)):
        """
        Solves the geodesic equation and returns the solution at t=1.

        Args:
            Z (torch.Tensor): Tensor of shape (N, 2) representing the base points (u, v).
            V (torch.Tensor): Tensor of shape (N, 2) representing the initial velocities (du_dt, dv_dt).
            t_span (torch.Tensor): Time points at which solution is to be evaluated
        Returns:
            torch.Tensor: Tensor of shape (N, 4) representing the solution [u(1), v(1), du_dt(1), dv_dt(1)].
        """
        # print(f"base_pts.shape: {base_pts.shape}, velocities.shape: {velocities.shape}")
        initial_state = torch.cat([base_pts, velocities], dim=-1)
        # print(f"initial_state.shape: {initial_state.shape}")

        solution = odeint(self.geodesic_rhs, initial_state, t_span, rtol=1e-10, atol=1e-12)
        
        return solution[-1][:,:2]  # Return the solution at t=1