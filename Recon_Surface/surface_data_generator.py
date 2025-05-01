
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random_generators #local

# from scipy.stats import uniform_direction
import surface_math #local

from tqdm import tqdm

import pdb
import gc


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
n_steps = 40 # Sequence Length
N = 10000 # Batch Size
max_hop = 0.1 # Scale parameter for velocities



# Generate Initial Points
X0 = 0.5 * torch.rand(N,2) + 0.25 # Uniformly distributed in [0,1]x[0,1]

V, pos = torch.zeros((N, n_steps, 2), requires_grad=False), torch.zeros((N, n_steps, 2), requires_grad=False)

mbs = 10000 # mini-batch size

start_pts = X0.clone().requires_grad_(True) # Start points for the first iteration


for k in tqdm(range(n_steps)):
    for i in (range(N // mbs)):
        V[mbs*i:mbs*(i+1), k], pos[mbs*i:mbs*(i+1), k] = random_generators.random_hops(start_pts=start_pts[mbs*i:mbs*(i+1)], max_hop=max_hop)
        # del V_intermediate, pos_intermediate # Free up memory
    start_pts = pos[:, k].clone().requires_grad_(True) # Update start points for the next iteration
    torch.cuda.empty_cache()

print(f"X0 shape: {X0.shape}, V shape: {V.shape}, pos shape: {pos.shape}")
# path = 'Recon_Surface\\data'
# path = 'data'

torch.save(pos, f'pos.pt')
torch.save(V, f'V.pt')
torch.save(X0, f'X0.pt')

# print(torch.load('Torus\data\X0.pt').shape)
print(torch.load(f'X0.pt').shape)

