import torch




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