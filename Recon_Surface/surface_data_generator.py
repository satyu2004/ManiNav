
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random_generators #local

# from scipy.stats import uniform_direction
import surface_math #local

from tqdm import tqdm

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
n_steps = 100 # Sequence Length
N = 100 # Batch Size
max_hop = 0.1 # Scale parameter for velocities



# Generate Initial Points
X0 = 0.5 * torch.rand(N,2) + 0.25 # Uniformly distributed in [0,1]x[0,1]

V, pos = torch.zeros((N, n_steps, 2)), torch.zeros((N, n_steps, 2))

minibatch_size = 100
for i in tqdm(range(N // minibatch_size)):
    V[minibatch_size*i:minibatch_size*(i+1)], pos[minibatch_size*i:minibatch_size*(i+1)] = random_generators.random_trajectories(X0=X0[minibatch_size*i:minibatch_size*(i+1)], n_steps=n_steps, max_hop=max_hop)

print(f"X0 shape: {X0.shape}, V shape: {V.shape}, pos shape: {pos.shape}")
path = 'Recon_Surface\\data'

torch.save(pos, f'{path}\pos.pt')
torch.save(V, f'{path}\V.pt')
torch.save(X0, f'{path}\X0.pt')

# print(torch.load('Torus\data\X0.pt').shape)
print(torch.load(f'{path}\X0.pt').shape)





# # Initial Pts
# a = 1 # Initial disk radius
# X0 = random_generators.generate_uniform_disk_vectors_torch(num_vectors=N, radius=a)

# # Velocities
# epsilon = 0.1
# V = torch.zeros((N, n, 2))
# for i in range(n):
#     V[:,i] = random_generators.generate_uniform_disk_vectors_torch(num_vectors=N, radius=epsilon)


# pos_rel = torch.cumsum(V, dim=1)
# pos = X0.unsqueeze(1) + pos_rel # Ground Truth Positions

# path = 'data'
# torch.save(pos, f'{path}\pos.pt')
# torch.save(V, f'{path}\V.pt')
# torch.save(X0, f'{path}\X0.pt')

# print(torch.load('data\X0.pt').shape)