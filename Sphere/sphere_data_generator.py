
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random_generators #local

from scipy.stats import uniform_direction
import sphere_math

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
n_steps = 100 # Sequence Length
N = 175000 # Batch Size
max_hop = 0.1 # Scale parameter for velocities




# Pick initial point
X0_3d = torch.tensor(uniform_direction.rvs(3, size=N))
X0_3d[:,2] = -1 * torch.abs(X0_3d[:,2])
indices = X0_3d[:,2]<-1/np.sqrt(2) # only retain points close to south pole
X0_3d = X0_3d[indices]
X0 = sphere_math.chart(X0_3d)
N = X0.shape[0]

# print(N)
# Generate Trajectories
V, pos, V_3d, pos_3d = random_generators.random_trajectories(X0=X0, n_steps=n_steps, max_hop=max_hop)

path = 'Sphere\data'
torch.save(pos, f'{path}\pos.pt')
torch.save(V, f'{path}\V.pt')
torch.save(X0, f'{path}\X0.pt')

print(torch.load('Sphere\data\X0.pt').shape)






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