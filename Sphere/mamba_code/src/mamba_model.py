# src/mamba_model.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

from sphere_math import immersion

class ContinuousMamba(nn.Module):
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, 
                 num_layers=2, num_x_tokens=1, x_dim=2, v_dim=2, y_dim=2):
        super().__init__()
        self.num_x_tokens = num_x_tokens
        self.n_embd = d_model

        self.x_embedding = nn.Linear(x_dim, num_x_tokens * d_model)
        self.v_projection = nn.Linear(v_dim, d_model)
        self.output_head = nn.Linear(d_model, y_dim)

        # Stack multiple Mamba blocks
        self.backbone = nn.Sequential(*[
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, x_inputs, v_inputs, labels=None):
        x_embeds = self.x_embedding(x_inputs).view(-1, self.num_x_tokens, self.n_embd)
        v_embeds = self.v_projection(v_inputs) 
        full_input = torch.cat([x_embeds, v_embeds], dim=1)

        hidden_states = self.backbone(full_input)
        v_outputs = hidden_states[:, self.num_x_tokens:, :]
        y_preds = self.output_head(v_outputs)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(immersion(y_preds), immersion(labels))

        return {"loss": loss, "logits": y_preds}
