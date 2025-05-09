from transformers import GPT2Model, GPT2PreTrainedModel
import torch
import torch.nn as nn

from sphere_math import immersion




class ContinuousGPT2(GPT2PreTrainedModel):
    def __init__(self, config, num_x_tokens=3, x_dim = 3, v_dim=3, y_dim=3):
        super().__init__(config)
        self.gpt2 = GPT2Model(config)
        self.num_x_tokens = num_x_tokens
        self.x_embedding = nn.Linear(x_dim, num_x_tokens*config.n_embd)
        self.v_projection = nn.Linear(v_dim, config.n_embd)
        self.output_head = nn.Linear(config.n_embd, y_dim)
        
        self.num_x_tokens = num_x_tokens
        self.n_embd = config.n_embd 
        
        # If you want to try fine tuning from an LLM, you can initialize weights. make sure to use the default GPT 2 config
        # self.init_weights()

    def forward(self, x_inputs, v_inputs, attention_mask=None, labels=None):
        x_embeds = self.x_embedding(x_inputs).view(-1, self.num_x_tokens, self.n_embd)
        v_embeds = self.v_projection(v_inputs)
        full_input = torch.cat([x_embeds, v_embeds], dim=1)

        position_ids = torch.arange(full_input.size(1), device=full_input.device).unsqueeze(0)
        hidden_states = self.gpt2(inputs_embeds=full_input,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids).last_hidden_state

        v_outputs = hidden_states[:, self.num_x_tokens:, :]
        y_preds = self.output_head(v_outputs)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(immersion(y_preds), immersion(labels))

        return {"loss": loss, "logits": y_preds}
