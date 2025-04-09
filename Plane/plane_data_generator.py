
import torch
torch.set_default_dtype(torch.float64)
import random_generators #local


n = 100 # Sequence Length
N = 50000 # Batch Size
max_v = 0.1 # Scale parameter for velocities



# Initial Pts
a = 1 # Initial disk radius
X0 = random_generators.generate_uniform_disk_vectors_torch(num_vectors=N, radius=a)

# Velocities
epsilon = 0.1
V = torch.zeros((N, n, 2))
for i in range(n):
    V[:,i] = random_generators.generate_uniform_disk_vectors_torch(num_vectors=N, radius=epsilon)


pos_rel = torch.cumsum(V, dim=1)
pos = X0.unsqueeze(1) + pos_rel # Ground Truth Positions

path = 'data'
torch.save(pos, f'{path}\pos.pt')
torch.save(V, f'{path}\V.pt')
torch.save(X0, f'{path}\X0.pt')

print(torch.load('data\X0.pt').shape)