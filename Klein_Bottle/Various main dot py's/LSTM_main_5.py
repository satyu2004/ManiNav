# from training_loops import RNN_train_aggregated as train

from Torus.torus_math import immersion, chart
import models

import torch
torch.set_default_dtype(torch.float64)
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import uniform_direction

import time
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

import pickle
def pickle_list(my_list, filename):
    """
    Pickles a list and saves it to a file.

    Args:
        my_list (list): The list to pickle.
        filename (str): The filename to save the pickled list to.
    """
    try:
        with open(filename, 'wb') as f:  # 'wb' for write binary
            pickle.dump(my_list, f)
        print(f"List pickled and saved to {filename}")
    except IOError as e:
        print(f"Error pickling list: {e}")

base_architecture = models.ConditionalLSTM
base_name = 'LSTM'
path = 'Sphere'

n_runs = 10
batch_size = 1024
num_epochs = 1000
hidden_dims = [128]
RNN_models = [[base_architecture(hidden_size=d)]*n_runs for d in hidden_dims]

X0, V, pos = torch.load(f'{path}\data\X0.pt').to(device), torch.load(f'{path}\data\V.pt').to(device), torch.load(f'{path}\data\pos.pt').to(device)

N = X0.shape[0]
n = V.shape[1]
# Data Preparation and Train-Test Splitting
train_test_split = 0.8
X0_train = X0[:int(train_test_split*N)]
X0_test = X0[int(train_test_split*N):]
V_train = V[:int(train_test_split*N)]
V_test = V[int(train_test_split*N):]
pos_train = pos[:int(train_test_split*N)]
pos_test = pos[int(train_test_split*N):]



def train(net, X0, V, pos, seq_length, indices_to_aggregate=[], lr=0.01, batch_size = 1024, num_epochs = 1000):
  k = seq_length
  # Define optimizer and scheduler
  optimizer = optim.Adam(net.parameters(), lr = lr)  # Example optimizer
  # Load your dataset
  train_loader = DataLoader(TensorDataset(X0, V, pos), batch_size=batch_size, shuffle=True)

  # Training loop
  num_epochs = num_epochs
  run_time = time.time()
  if len(indices_to_aggregate)>0:
    L = indices_to_aggregate
  else:
    L = range(1, k+1)
  # print(f"Training by aggregating on indices {L}")
  for epoch in tqdm(range(num_epochs)):
      running_loss = 0.0

      for minibatch in train_loader:

          # Forward pass
          X = minibatch[0].to(device)
          V = minibatch[1].to(device)
          Y = minibatch[2].to(device)
          loss = 0

          for i in L:
            Yhat = net(X,V[:,:i]).squeeze()
            criterion = nn.MSELoss()
            loss += criterion(immersion(Y[:,i-1]), immersion(Yhat))


          # Backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

      # if epoch%100==0:
      #   print(f"Epoch = {epoch+1}, Loss = {running_loss/len(train_loader) :.4e}")
  runtime = time.time()-run_time  
  # print(f"Training time = {runtime}")
  return runtime


# Training all models and computing predictions
runtimes = []
for dim, model_list in tqdm(zip(hidden_dims, RNN_models)):
    times = []
    for run, model in tqdm(enumerate(model_list)):
        t = train(model, X0=X0_train, V=V_train, pos=pos_train, seq_length=10, indices_to_aggregate=[], lr=0.01, batch_size = batch_size, num_epochs = num_epochs)
        times.append(t)
        torch.save(model.state_dict(), f'{path}\model_weights\{base_name}\hidden_dim_{dim}_{run}.pth')
        pos_pred = torch.zeros_like(pos_test)
        # for i in range(n):
        #     pos_pred[:,i] = model(x_0=X0_test, V=V_test[:,:i+1]).squeeze()
        # torch.save(pos_pred, f'results\{base_name}\hidden_dim_{dim}_{run}.pt')
    pickle_list(times, f'{path}\runtimes_{base_name}_dim_{dim}.pkl')    
    