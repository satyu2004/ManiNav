import torch
from torch.utils.data import DataLoader
from src.dataset import MyTrajectoryDataset
from src.mamba_model import ContinuousMamba
from src.collator import collate_fn
import yaml
import os

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-instantiate the model (must match training config)
model = ContinuousMamba(
    d_model=cfg["n_embd"],
    num_layers=cfg["n_layer"],
    num_x_tokens=cfg["num_x_tokens"],
    x_dim=cfg["x_dim"],
    v_dim=cfg["v_dim"],
    y_dim=cfg["y_dim"]
)
model.to(device)

run_ID = 0
# Load best model weights
best_model_dir = os.path.join(cfg["output_dir"], f"run_{run_ID}")
best_model_path = os.path.join(best_model_dir, "best_model.pt")
model.load_state_dict(torch.load(best_model_path))
model.eval()
print("✅ Loaded best model weights.")

# Load test data
X0_test = torch.load(f"{cfg['data_path']}/X0.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+100].float()
V_test = torch.load(f"{cfg['data_path']}/V.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+100,:cfg["n_v_steps"],:].float()
pos_test = torch.load(f"{cfg['data_path']}/pos.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+100,:cfg["n_v_steps"],:].float()
test_dataset = MyTrajectoryDataset(X0_test, V_test, pos_test)
test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], collate_fn=collate_fn)

# Run inference on test set
all_preds = []
all_labels = []

with torch.no_grad():
    first_batch = True
    for batch in test_dataloader:
        x = batch["x_inputs"].to(device)
        v = batch["v_inputs"].to(device)
        y = batch["labels"].to(device)

        out = model(x, v, y)
        preds = out["logits"]
        mse_internal = out["loss"]

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        if first_batch:
            print('MSE of first batch:', mse_internal)


# Stack all predictions and labels
all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# You can now compute custom metrics like MSE, MAE, R², etc.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(all_labels.numpy().reshape(-1, 2), all_preds.numpy().reshape(-1, 2))
#mae = mean_absolute_error(all_labels.numpy(), all_preds.numpy())
#r2 = r2_score(all_labels.numpy(), all_preds.numpy())

print(f"📊 Test MSE: {mse:.4f}")
#print(f"📊 Test MAE: {mae:.4f}")
#print(f"📊 Test R²: {r2:.4f}")
