import torch
from torch.utils.data import Dataset

class MyTrajectoryDataset(Dataset):
    def __init__(self, X0, V, pos):
        self.X0 = X0  # (N, num_x_tokens)
        self.V = V    # (N, T, v_dim)
        self.pos = pos  # (N, T, y_dim)

    def __len__(self):
        return self.X0.size(0)

    def __getitem__(self, idx):
        return {
            "x_inputs": self.X0[idx].float(),    # (x_dim,)
            "v_inputs": self.V[idx].float(),       # (T, v_dim)
            "labels": self.pos[idx].float(),       # (T, y_dim)
        }
