import yaml
import torch
from transformers import GPT2Config
from src.model import ContinuousGPT2
from src.dataset import MyCustomDataset
from src.collator import collate_fn
from torch.utils.data import DataLoader

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.dataset import MyTrajectoryDataset
import torch

# Load real data
X0 = torch.load(f"{cfg['data_path']}/X0.pt")[cfg["n_trajectories"]:].float()
V = torch.load(f"{cfg['data_path']}/V.pt")[cfg["n_trajectories"]:].float()
pos = torch.load(f"{cfg['data_path']}/pos.pt")[cfg["n_trajectories"]:].float()

eval_dataset = MyTrajectoryDataset(X0, V, pos)

hf_config = GPT2Config.from_pretrained(cfg["model_name"])
model = ContinuousGPT2(hf_config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])
model.load_state_dict(torch.load(f"{cfg['output_dir']}/pytorch_model.bin", map_location="cpu"))
model.eval()

eval_loader = DataLoader(eval_dataset, batch_size=cfg["batch_size"], collate_fn=collate_fn)

mse_total, mae_total, n = 0.0, 0.0, 0

with torch.no_grad():
    for batch in eval_loader:
        out = model(batch["x_inputs"], batch["v_inputs"], batch["attention_mask"])
        preds = out["logits"]
        targets = batch["labels"]

        mse = ((preds - targets) ** 2).mean().item()
        mae = (preds - targets).abs().mean().item()

        mse_total += mse * preds.size(0)
        mae_total += mae * preds.size(0)
        n += preds.size(0)

print(f"✅ Eval results over {n} samples:")
print(f"   MSE: {mse_total / n:.4f}")
print(f"   MAE: {mae_total / n:.4f}")
