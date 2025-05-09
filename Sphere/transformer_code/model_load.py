from safetensors.torch import load_file  # Hugging Face recommends using this for .safetensors
import os
import random
import numpy as np
import torch
import yaml
from transformers import GPT2Config, TrainingArguments, Trainer
from src.model import ContinuousGPT2
from src.rope_model import ContinuousRotaryTransformer
from src.collator import collate_fn
from src.callbacks import SimpleLoggerCallback
from datetime import datetime
from transformers import set_seed
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.dataset import MyTrajectoryDataset




X0 = torch.load(f"{cfg['data_path']}/X0.pt")[:cfg["n_trajectories"]].float()
V = torch.load(f"{cfg['data_path']}/V.pt")[:cfg["n_trajectories"],:cfg["n_velocity_steps"],:].float()
pos = torch.load(f"{cfg['data_path']}/pos.pt")[:cfg["n_trajectories"], :cfg["n_velocity_steps"], :].float()

dataset = MyTrajectoryDataset(X0, V, pos)

# Path to a specific run
run_id = 0  # example
checkpoint_dir = f"checkpoints/run_{run_id}"

# Step 1: Load config
config = GPT2Config.from_json_file(f"{checkpoint_dir}/config.json")

# Step 2: Rebuild your model
rotary_model = False
if rotary_model:
    model = ContinuousRotaryTransformer(config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])
else:
    model = ContinuousGPT2(config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])

# Step 3: Load the weights
state_dict = load_file(f"{checkpoint_dir}/model.safetensors")
model.load_state_dict(state_dict)
model.eval()  # Don't forget this if you're testing!

# Step 4: Load a test sample (example: first one in your dataset)
test_sample = dataset[:20]
x_inputs = test_sample["x_inputs"]  # Add batch dim
v_inputs = test_sample["v_inputs"]
attention_mask = torch.ones((x_inputs.size(0), 3 + v_inputs.size(1)))
pos_inputs = test_sample["labels"]

# Step 5: Run forward pass (assuming your model has a forward(x_inputs, v_inputs))
with torch.no_grad():
    output = model(x_inputs, v_inputs, attention_mask, pos_inputs)

print("Model output:", output['logits'])
print("Model loss:", output['loss'])
