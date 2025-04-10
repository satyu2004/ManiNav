
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random_generators #local

from scipy.stats import uniform_direction
import torus_math

from tqdm import tqdm

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
n_steps = 100 # Sequence Length
N = 10000 # Batch Size
max_hop = 0.1 # Scale parameter for velocities



# Generate Initial Points
X0 = 2 * torch.pi * torch.rand(N,2)

V, pos = torch.zeros((N, n_steps, 2)), torch.zeros((N, n_steps, 2))

for i in tqdm(range(N // 1000)):
    V[1000*i:1000*(i+1)], pos[1000*i:1000*(i+1)] = random_generators.random_trajectories(X0=X0[1000*i:1000*(i+1)], n_steps=n_steps, max_hop=max_hop)


path = 'Torus\data'

torch.save(pos, f'{path}\pos.pt')
torch.save(V, f'{path}\V.pt')
torch.save(X0, f'{path}\X0.pt')

print(torch.load('Torus\data\X0.pt').shape)






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