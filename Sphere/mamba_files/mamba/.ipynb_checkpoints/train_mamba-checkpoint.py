# train_mamba.py
import torch
from torch.utils.data import DataLoader
from src.dataset import MyTrajectoryDataset
from src.mamba_model import ContinuousMamba
from src.collator import collate_fn
import yaml
import os
from src.callbacks import SimpleLogger


# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Dataset
X0 = torch.load(f"{cfg['data_path']}/X0.pt")[:cfg["n_trajectories"]].float()
V = torch.load(f"{cfg['data_path']}/V.pt")[:cfg["n_trajectories"], :cfg["n_v_steps"],:].float()
pos = torch.load(f"{cfg['data_path']}/pos.pt")[:cfg["n_trajectories"], :cfg["n_v_steps"],:].float()
dataset = MyTrajectoryDataset(X0, V, pos)


X0_test = torch.load(f"{cfg['data_path']}/X0.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+cfg["n_eval"]].float()
V_test = torch.load(f"{cfg['data_path']}/V.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+cfg["n_eval"], :cfg["n_v_steps"],:].float()
pos_test = torch.load(f"{cfg['data_path']}/pos.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+cfg["n_eval"], :cfg["n_v_steps"],:].float()
test_dataset = MyTrajectoryDataset(X0_test, V_test, pos_test)
test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], collate_fn=collate_fn)


dataloader = DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x_inputs"].to(device)
            v = batch["v_inputs"].to(device)
            y = batch["labels"].to(device)

            out = model(x, v, y)
            total_loss += out["loss"].item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model.train()
    return avg_loss



n_runs = 10
for run_idx in range(n_runs):
    # Model
    model = ContinuousMamba(
        d_model=cfg["n_embd"],
        num_layers=cfg["n_layer"],
        num_x_tokens=cfg["num_x_tokens"],
        x_dim=cfg["x_dim"],
        v_dim=cfg["v_dim"],
        y_dim=cfg["y_dim"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * cfg["num_epochs"])
    
    
    logger = SimpleLogger(log_every=cfg["logging_steps"])
    eval_every = cfg.get("eval_every", 2)
    early_stopping_patience = cfg.get("early_stopping_patience", 3)
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    
    
    
    # Training loop
    run_output_dir = os.path.join(cfg["output_dir"], f"sphere_80_run_{run_idx}")
    run_logging_dir = os.path.join(cfg["logging_dir"], f"sphere_80_run_{run_idx}")
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_logging_dir, exist_ok=True)

    best_model_path = os.path.join(run_output_dir, "best_model.pt")
    
    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            x = batch["x_inputs"].to(device)
            v = batch["v_inputs"].to(device)
            y = batch["labels"].to(device)
    
            optimizer.zero_grad()
            out = model(x, v, y)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            total_loss += loss.item()
    
            logger.log_step(epoch, step, loss.item())
            if step % cfg["logging_steps"] == 0:
                print(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}")
        logger.log_epoch(epoch, total_loss / len(dataloader))
        if (epoch + 1) % eval_every == 0:
            val_loss = evaluate(model, test_dataloader, device)
            logger.log_eval(epoch, val_loss, checkpoint_dir=run_logging_dir, model=model)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ New best val loss: {val_loss:.4f}. Model saved.")
            else:
                epochs_since_improvement += 1
                print(f"No improvement in val loss for {epochs_since_improvement} evals.")
    
            # Check early stopping
            if epochs_since_improvement >= early_stopping_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break
    
        # Save model
        #torch.save(model.state_dict(), os.path.join(run_output_dir, f"mamba_epoch_{epoch}.pt"))
        print(f"Epoch {epoch} completed. Avg Loss: {total_loss/len(dataloader):.4f}")
