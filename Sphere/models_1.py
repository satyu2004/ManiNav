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

# class RNN_multilayer(nn.Module):
#     def __init__(self, hidden_size, input_size=2, output_size=2, num_layers=2):
#         super(RNN_multilayer, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # self.encoder = nn.Linear(2*output_size, hidden_size)
#         self.encoder = [nn.Linear(input_size, hidden_size)]*num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity='relu', bias=False, batch_first=True)
#         self.decoder = nn.Linear(hidden_size, output_size)

#     def forward(self, x_0, V):
#         # Initialize hidden state
#         h_0 = torch.stack([self.encoder[i](x_0) for i in range(self.num_layers)], dim=0)

#         # Forward pass through RNN
#         out = self.rnn(V, h_0)
#         out = self.decoder(out[1][-1])

#         return out

class RNN_multilayer(nn.Module):
    def __init__(self, hidden_size, input_size=2, output_size=2, num_layers=2):
        super(RNN_multilayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.encoder = nn.Linear(2*output_size, hidden_size)
        # self.encoder = [nn.Linear(input_size, hidden_size)]*num_layers # original version

        self.encoder = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_layers)])
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity='relu', bias=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x_0, V):
        # Initialize hidden state
        h_0 = torch.stack([self.encoder[i](x_0) for i in range(self.num_layers)], dim=0)

        # Forward pass through RNN
        out = self.rnn(V, h_0)
        # out = self.decoder(out[1][-1])
        out = self.decoder(out[0])

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


class TransformerWrapper(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        from transformers import GPT2Config
        from safetensors.torch import load_file
        from transformer_code.src.model import ContinuousGPT2
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        config = GPT2Config.from_json_file(f"{checkpoint_path}/config.json")
        self.model = ContinuousGPT2(config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])
        state_dict = load_file(f"{checkpoint_path}/model.safetensors")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x_0, V):
        batch_size, steps, _ = V.shape
        attention_mask = torch.ones((batch_size, 3 + steps), device=V.device)
        with torch.no_grad():
            output = self.model(x_0, V, attention_mask, None)
        return output["logits"]





class RotaryTransformerWrapper(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        from transformers import GPT2Config
        from safetensors.torch import load_file
        from transformer_code.src.rope_model import ContinuousRotaryTransformer
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        hf_config = GPT2Config(
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            vocab_size=cfg["vocab_size"],
        )
        self.model = ContinuousRotaryTransformer(hf_config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])
        state_dict = load_file(f"{checkpoint_path}/model.safetensors")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x_0, V):
        batch_size, steps, _ = V.shape
        attention_mask = torch.ones((batch_size, 3 + steps), device=V.device)
        with torch.no_grad():
            output = self.model(x_0, V, attention_mask, None)
        return output["logits"]



class MambaWrapper(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        from mamba_code.src.mamba_model import ContinuousMamba
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model = ContinuousMamba(
            d_model=cfg["n_embd"],
            num_layers=cfg["n_layer"],
            num_x_tokens=cfg["num_x_tokens"],
            x_dim=cfg["x_dim"],
            v_dim=cfg["v_dim"],
            y_dim=cfg["y_dim"]
        )
        self.model.load_state_dict(torch.load(f"{checkpoint_path}/best_model.pt"))
        self.model.eval()

    def forward(self, x_0, V):
        batch_size, steps, _ = V.shape
        with torch.no_grad():
            out = self.model(x_0, V, None)
        return out["logits"]