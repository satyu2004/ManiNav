import torch
torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Immersed_Manifold:
    def __init__(self, f=None, immersion=None, chart=None):
        self.immersion = immersion
        self.chart = chart
        self.f = f.to(device) if f is not None else None

    def compute_christoffel_symbols(self, pts):
        """
        Compute the Christoffel symbols of the second kind for a surface z=f(x,y)
        at the given points.
        
        Args:
            f: A function that takes a batch of 2D points and returns corresponding z values
            pts: Tensor of shape (batch_size, 2) containing the (x,y) points
        
        Returns:
            Tensor of shape (batch_size, 2, 2, 2) containing the Christoffel symbols
            The indices are [batch, i, j, k] where i is the upper index and j,k are lower indices
            This corresponds to Γⁱⱼₖ in standard notation
        """
        # Ensure pts requires gradient for automatic differentiation
        pts = pts.clone().detach().requires_grad_(True).to(device)
        batch_size = pts.shape[0]
        
        # Compute function values and first derivatives
        z = self.f(pts)
        
        # Compute first derivatives (gradients)
        ones = torch.ones_like(z)
        grad_z = torch.autograd.grad(z, pts, grad_outputs=ones, create_graph=True)[0]
        fx = grad_z[:, 0]  # ∂f/∂x
        fy = grad_z[:, 1]  # ∂f/∂y
        
        # Compute second derivatives
        grad_fx = torch.autograd.grad(fx, pts, grad_outputs=torch.ones_like(fx), create_graph=True)[0]
        grad_fy = torch.autograd.grad(fy, pts, grad_outputs=torch.ones_like(fy), create_graph=True)[0]
        with torch.no_grad():
            # Extract second derivatives
            fxx = grad_fx[:, 0]  # ∂²f/∂x²
            fxy = grad_fx[:, 1]  # ∂²f/∂x∂y
            
        
            fyx = grad_fy[:, 0]  # ∂²f/∂y∂x (should equal fxy)
            fyy = grad_fy[:, 1]  # ∂²f/∂y²
            
            # Compute metric tensor components
            g11 = 1 + fx**2
            g12 = fx * fy
            g22 = 1 + fy**2
            
            # Compute determinant of the metric tensor
            det_g = g11 * g22 - g12**2
            # Simplifies to: 1 + fx**2 + fy**2
            
            # Compute inverse metric tensor components
            g_inv_11 = g22 / det_g
            g_inv_12 = -g12 / det_g
            g_inv_22 = g11 / det_g
            
            # Compute Christoffel symbols
            # Γ¹₁₁ = g¹¹fx·fxx + g¹²fy·fxx
            gamma_111 = g_inv_11 * fx * fxx + g_inv_12 * fy * fxx
            
            # Γ¹₁₂ = g¹¹fx·fxy + g¹²fy·fxy
            gamma_112 = g_inv_11 * fx * fxy + g_inv_12 * fy * fxy
            
            # Γ¹₂₂ = g¹¹fx·fyy + g¹²fy·fyy
            gamma_122 = g_inv_11 * fx * fyy + g_inv_12 * fy * fyy
            
            # Γ²₁₁ = g²¹fx·fxx + g²²fy·fxx
            gamma_211 = g_inv_12 * fx * fxx + g_inv_22 * fy * fxx
            
            # Γ²₁₂ = g²¹fx·fxy + g²²fy·fxy
            gamma_212 = g_inv_12 * fx * fxy + g_inv_22 * fy * fxy
            
            # Γ²₂₂ = g²¹fx·fyy + g²²fy·fyy
            gamma_222 = g_inv_12 * fx * fyy + g_inv_22 * fy * fyy
            
            # Package into a rank-3 tensor of shape (batch_size, 2, 2, 2)
            # The indices are [point, upper index, lower index 1, lower index 2]
            # i.e., Gamma^i_jk where i is the upper index and j,k are lower indices
            christoffel = torch.zeros(batch_size, 2, 2, 2, device=pts.device)
            
            # Assign values to appropriate positions
            christoffel[:, 0, 0, 0] = gamma_111  # Γ¹₁₁
            christoffel[:, 0, 0, 1] = gamma_112  # Γ¹₁₂
            christoffel[:, 0, 1, 0] = gamma_112  # Γ¹₂₁ (equal to Γ¹₁₂ by symmetry)
            christoffel[:, 0, 1, 1] = gamma_122  # Γ¹₂₂
            christoffel[:, 1, 0, 0] = gamma_211  # Γ²₁₁
            christoffel[:, 1, 0, 1] = gamma_212  # Γ²₁₂
            christoffel[:, 1, 1, 0] = gamma_212  # Γ²₂₁ (equal to Γ²₁₂ by symmetry)
            christoffel[:, 1, 1, 1] = gamma_222  # Γ²₂₂
            
            del fx, fy, fxx, fxy, fyx, fyy  # Free up memory
            del g11, g12, g22, det_g, g_inv_11, g_inv_12, g_inv_22  # Free up memory
            del grad_z, grad_fx, grad_fy  # Free up memory
            del ones, z  # Free up memory
            del gamma_111, gamma_112, gamma_122, gamma_211, gamma_212, gamma_222  # Free up memory
            torch.cuda.empty_cache()  # Clear CUDA cache
        return christoffel

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
        with torch.no_grad():
            du_dt, dv_dt = Z[:, 2], Z[:, 3]
            
            d2u_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 0, :, :], Z[:, 2:], Z[:, 2:])
            d2v_dt2 = -torch.einsum('nij,ni,nj->n', G[:, 1, :, :], Z[:, 2:], Z[:, 2:])
        
        return torch.stack([du_dt, dv_dt, d2u_dt2, d2v_dt2], dim=-1)
    
    def exp(self, base_pts, velocities, t_span=torch.tensor([0, 1], dtype=torch.float64).to(device)):
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