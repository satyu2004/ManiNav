import torch
torch.set_default_dtype(torch.float64)
import sphere_math
import numpy as np




def generate_uniform_disk_vectors_torch(num_vectors, radius=0.1):
  """
  Generates a batch of vectors distributed uniformly in a disk of radius 1 using PyTorch.

  Args:
    num_vectors: The number of vectors to generate.

  Returns:
    A PyTorch tensor of shape (num_vectors, 2) containing the generated vectors.
  """

  radii = radius * torch.sqrt(torch.rand(num_vectors, dtype=torch.float64))  # Square root for uniform area distribution
  angles = 2 * torch.pi * torch.rand(num_vectors)
  x = radii * torch.cos(angles)
  y = radii * torch.sin(angles)

  return torch.stack((x, y), dim=1)

def random_trajectories(X0, n_steps=20, max_hop=0.1):
      """
        Args:
          X0: initial points (tensor of shape (N,2))
          n_steps: number of steps
          max_hop: maximum distance by which you can jump

        Returns:
          V: velocities in chart (tensor of shape (N, n_steps, 2))
          pos: positions in chart (tensor of shape (N, n_steps, 2))
          V_3d: velocities in 3d space (tensor of shape (N, n_steps, 3))
          pos_3d: positions in 3d space (tensor of shape (N, n_steps, 3))
      """

      N = X0.shape[0]
      V, pos, V_3d, pos_3d =  torch.zeros((N, n_steps, 2), dtype=torch.float64), torch.zeros((N, n_steps, 2), dtype=torch.float64), torch.zeros((N, n_steps, 3), dtype=torch.float64), torch.zeros((N, n_steps, 3), dtype=torch.float64)
      start_pts = X0
      X0_3d = sphere_math.immersion(start_pts)

      for i in range(n_steps):
        # Compute Jacobian at this point
        jacobians = sphere_math.jacobian_matrix_batch(start_pts)
        pinvs = torch.linalg.pinv(jacobians)

        #Find its QR Decomposition
        Q,R = torch.linalg.qr(jacobians)


        # Sample small vector in the plane
        small_vectors = generate_uniform_disk_vectors_torch(N, radius=max_hop).unsqueeze(-1)


        # Map it into the tangent space and pull back to chart
        random_tangents_3d = torch.bmm(Q, small_vectors)
        random_tangents = torch.bmm(pinvs, random_tangents_3d).squeeze()
        V[:,i,:] = random_tangents
        random_tangents_3d = random_tangents_3d.squeeze()
        V_3d[:,i,:] = random_tangents_3d

        # Compute the exponential map of this vector
        # AXES = torch.cross(X0_3d,random_tangents_3d, dim=1)
        # AXES = AXES/torch.linalg.norm(AXES,dim=1).unsqueeze(-1)
        # ANGLE = torch.linalg.norm(random_tangents_3d,dim=1)
        # exponentials_3d = np.array([Quaternion(axis=AXES[i], angle=ANGLE[i]).rotate(X0_3d[i]) for i in range(AXES.shape[0])])
        exponentials_3d = sphere_math.parallel_transport(sphere_math.immersion(start_pts), random_tangents_3d)[0] # retain only rotated_x from parallel transport
        # exponentials_3d = torch.tensor(exponentials_3d)
        pos_3d[:,i,:] = exponentials_3d
        exponentials = sphere_math.chart(exponentials_3d)
        pos[:,i,:] = exponentials
        start_pts = exponentials
      return V, pos, V_3d, pos_3d