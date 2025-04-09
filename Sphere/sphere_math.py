import torch
torch.set_default_dtype(torch.float64)


def immersion(X):
      " Maps points in the plane to points on the sphere via the inverse of stereographic projection"
      x = X[..., 0]
      y = X[..., 1]
      R = x**2 + y**2 + 1
      return torch.stack([2*x/R, 2*y/R, 1-2/R], dim=-1)

def chart(X):
  x, y, z = X[:,0], X[:,1], X[:,2]
  x_coords = x/(1-z)
  y_coords = y/(1-z)
  return torch.stack((x_coords, y_coords), dim=1)


def jacobian_matrix_batch(xy_tensor):
      """
      Calculates the Jacobian matrix for a batch of input points.

      Args:
        xy_tensor: A tensor of shape (batch_size, 2) containing (x, y) pairs.

      Returns:
        A tensor of shape (batch_size, 3, 2) containing the Jacobian matrices for each input point.
      """

      x, y = xy_tensor[:, 0], xy_tensor[:, 1]
      denominator = (x**2 + y**2 + 1)**2
      first_row = torch.stack([(2*y**2 - 2*x**2 + 2) / denominator, -4*x*y / denominator],dim=1)
      second_row = torch.stack([-4*x*y / denominator, (2*x**2 - 2*y**2 + 2) / denominator],dim=1)
      third_row = torch.stack([4*x / denominator, 4*y / denominator],dim=1)

      jacobians = torch.stack([first_row, second_row, third_row], dim=1)
      return jacobians


def parallel_transport(x, v, angle=None):
    """
    Rotates a tangent vector v at point x on the unit sphere.
    This performs parallel transport of the tangent vector.

    Args:
        x (torch.Tensor): Unit vector on the sphere (|x| = 1)
        v (torch.Tensor): Tangent vector at x (so x⋅v = 0)
        angle (float, optional): Rotation angle. If None, uses |v| as the angle

    Returns:
        tuple: (rotated_x, rotated_v) - both the rotated point and its tangent vector
    """

    # Ensure inputs are PyTorch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)

    # Normalize x to ensure it's on the unit sphere
    x = x / torch.norm(x, dim=1, keepdim=True)

    # Project v onto the tangent space of x if it's not already tangent
    dot_product = (x*v).sum(dim=1, keepdim=True)
    v_tangent = v - dot_product * x

    # Calculate the rotation angle (norm of tangent vector)
    if angle is None:
        angle = torch.norm(v_tangent, dim=1,keepdim=True)


    # If tangent vector is zero (or nearly zero), no rotation needed
    if torch.norm(v_tangent) < 1e-8:
        return x, v_tangent

    # Normalize v_tangent to get direction (preserving magnitude for later)
    v_norm = torch.norm(v_tangent, dim=1, keepdim=True)
    v_dir = v_tangent / v_norm

    # Calculate the rotation axis a = x × v_dir (cross product)
    a = torch.cross(x, v_dir, dim=1)
    a = a / torch.norm(a, dim=1, keepdim=True)  # Normalize rotation axis

    # Use Rodrigues' rotation formula to rotate x
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # Rotate the position vector x
    cross = torch.cross(a, x, dim=1)
    rotated_x = cos_angle * x  + sin_angle * cross


    # Rotate the tangent vector v using the same rotation
    # For a tangent vector, we need to maintain tangency at the new point

    # First rotate v as if it were a free vector
    rotated_v_free = v_tangent * cos_angle + torch.cross(a, v_tangent, dim=1) * sin_angle

    # Then project back to the tangent space at the new point
    dot_product = (rotated_x*rotated_v_free).sum(dim=1, keepdim=True)
    rotated_v = rotated_v_free - dot_product * rotated_x

    # Preserve the original magnitude of v
    rotated_v = rotated_v * (v_norm / torch.norm(rotated_v, dim=1, keepdim=True))

    return rotated_x, rotated_v