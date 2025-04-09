import torch
torch.set_default_dtype(torch.float64)
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim


def RNN_train_aggregated(net, X0, V, pos, seq_length, indices_to_aggregate=[], lr=0.01, batch_size = 1024, num_epochs = 1000):
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
  print(f"Training by aggregating on indices {L}")
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

      if epoch%100==0:
        print(f"Epoch = {epoch+1}, Loss = {running_loss/len(train_loader) :.4e}")
  runtime = time.time()-run_time  
  print(f"Training time = {runtime}")
  return runtime
