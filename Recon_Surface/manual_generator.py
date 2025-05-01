import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random_generators #local

# from scipy.stats import uniform_direction
import surface_math #local

from tqdm import tqdm


V = torch.load("V.pt")
pos = torch.load("pos.pt")
counter = torch.load("counter.pt")

if counter == 0:
    start_pts = torch.load("X0.pt")

else:
    start_pts = pos[:,-1]

batch_size = 1000
V_new_list = []
pos_new_list = []

# Split start_pts into minibatches
start_pts_batches = torch.split(start_pts, batch_size)

for start_pts_batch in tqdm(start_pts_batches):
    V_new_batch, pos_new_batch = random_generators.random_hops(start_pts_batch)
    V_new_list.append(V_new_batch)
    pos_new_list.append(pos_new_batch)
    torch.cuda.empty_cache()  # Wipe torch's memory

V_new = torch.cat(V_new_list, dim=0)
pos_new = torch.cat(pos_new_list, dim=0)

V = torch.cat((V, V_new.unsqueeze(dim=1)), dim=0)
pos = torch.cat((pos, pos_new.unsqueeze(dim=1)), dim=0)

torch.save(V, "V.pt")
torch.save(pos, "pos.pt")
counter += 1
print("Counter:", counter)
torch.save(counter, "counter.pt")


