from torch import nn
import torch
from src.transformer_with_rope import RotaryTransformer

from sphere_math import immersion

class ContinuousRotaryTransformer(nn.Module):
    def __init__(self, config, num_x_tokens=3, x_dim=3, v_dim=3, y_dim=3):
        super().__init__()
        self.config = config
        self.num_x_tokens = num_x_tokens
        self.n_embd = config.n_embd

        self.x_embedding = nn.Linear(x_dim, num_x_tokens * self.n_embd)
        self.v_projection = nn.Linear(v_dim, self.n_embd)

        self.transformer = RotaryTransformer(
            num_layers=config.n_layer,
            dim=self.n_embd,
            n_heads=config.n_head
        )

        self.output_head = nn.Linear(self.n_embd, y_dim)

    def forward(self, x_inputs, v_inputs, attention_mask=None, labels=None):
        x_embeds = self.x_embedding(x_inputs).view(-1, self.num_x_tokens, self.n_embd)
        v_embeds = self.v_projection(v_inputs)
        full_input = torch.cat([x_embeds, v_embeds], dim=1)

        hidden_states = self.transformer(full_input)
        v_outputs = hidden_states[:, self.num_x_tokens:, :]
        y_preds = self.output_head(v_outputs)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(immersion(y_preds), immersion(labels))

        return {"loss": loss, "logits": y_preds}
