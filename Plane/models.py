import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, hidden_size, input_size=2, output_size=2, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.encoder = nn.Linear(2*output_size, hidden_size)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity='relu', bias=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x_0, V):
        # Initialize hidden state
        h_0 = self.encoder(x_0).unsqueeze(0)

        # Forward pass through RNN
        out = self.rnn(V, h_0)
        out = self.decoder(out[1])

        return out




class ConditionalLSTM(nn.Module):
    def __init__(self, hidden_size, input_size_x=2, input_size_v=2, output_size=2):
        super(ConditionalLSTM, self).__init__()
        self.encoder_x = nn.Linear(input_size_x, hidden_size)
        self.lstm = nn.LSTM(input_size_v, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_0, V):
        # Encode x into initial hidden and cell states
        encoded_x = torch.tanh(self.encoder_x(x_0))  # Using tanh activation
        initial_hidden = encoded_x.unsqueeze(0)  # Shape (1, batch_size, hidden_size)
        initial_cell = torch.zeros_like(initial_hidden)  # Initialize cell state to zeros

        # Pass v through the LSTM with the initialized states
        lstm_out, _ = self.lstm(V, (initial_hidden, initial_cell))

        # Take the output from the last time step
        last_time_step_output = lstm_out[:, -1, :]
        output = self.fc(last_time_step_output)
        return output
    



class ConditionalGRU(nn.Module):
    def __init__(self, hidden_size, input_size_x=2, input_size_v=2, output_size=2):
        super(ConditionalGRU, self).__init__()
        self.encoder_x = nn.Linear(input_size_x, hidden_size)
        self.gru = nn.GRU(input_size_v, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_0, V):
        # Encode x into initial hidden state
        encoded_x = torch.tanh(self.encoder_x(x_0))
        initial_hidden = encoded_x.unsqueeze(0)

        # Pass v through the GRU with the initialized state
        gru_out, _ = self.gru(V, initial_hidden)

        # Take the output from the last time step
        last_time_step_output = gru_out[:, -1, :]
        output = self.fc(last_time_step_output)
        return output