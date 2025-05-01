import models
import torch
torch.set_default_dtype(torch.float64)
X0 = torch.load('Plane\\data\X0.pt')[:100]
V = torch.load('Plane\\data\V.pt')[:100]

net = models.LSTM_multilayer(hidden_size=8, num_layers=2)
print(net(X0, V))