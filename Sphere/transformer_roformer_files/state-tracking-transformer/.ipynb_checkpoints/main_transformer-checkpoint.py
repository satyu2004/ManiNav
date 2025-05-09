import os
import random
import numpy as np
import torch
import yaml
from transformers import GPT2Config, TrainingArguments, Trainer, EarlyStoppingCallback
from transformer_code.src.model import ContinuousGPT2
#from src.rope_model import ContinuousRotaryTransformer
from transformer_code.src.collator import collate_fn
from transformer_code.src.callbacks import SimpleLoggerCallback
from datetime import datetime
from transformers import set_seed
with open("transformer_code/config.yaml") as f:
    cfg = yaml.safe_load(f)

from transformer_code.src.dataset import MyTrajectoryDataset
from sklearn.metrics import mean_squared_error
import numpy as np

def compute_mse(eval_pred):
    # Unpack the predictions and labels from eval_pred
    predictions, labels = eval_pred

    dm = predictions.shape[2]
    
    # Convert them to numpy arrays for MSE calculation
    predictions = predictions.reshape(-1, dm)
    labels = labels.reshape(-1, dm)
    
    # Compute the Mean Squared Error (MSE)
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}



X0 = torch.load(f"{cfg['data_path']}/X0.pt")[:cfg["n_trajectories"]].float()
V = torch.load(f"{cfg['data_path']}/V.pt")[:cfg["n_trajectories"],:cfg["n_velocity_steps"],:].float()
pos = torch.load(f"{cfg['data_path']}/pos.pt")[:cfg["n_trajectories"], :cfg["n_velocity_steps"], :].float()


dataset = MyTrajectoryDataset(X0, V, pos)


X0_val = torch.load(f"{cfg['data_path']}/X0.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+1000].float()
V_val = torch.load(f"{cfg['data_path']}/V.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+1000,:cfg["n_velocity_steps"],:].float()
pos_val = torch.load(f"{cfg['data_path']}/pos.pt")[cfg["n_trajectories"]:cfg["n_trajectories"]+1000, :cfg["n_velocity_steps"], :].float()


val_dataset = MyTrajectoryDataset(X0_val, V_val, pos_val)

n_runs = 10
for run_idx in range(n_runs):
    print(f"=== Starting run {run_idx} ===")

    # Optional: set a unique random seed for each run
    seed = run_idx + 42  # or random.randint(0, 1e6)
    set_seed(seed)

    # Create a unique output directory for this run
    run_output_dir = os.path.join(cfg["output_dir"], f"run_{run_idx}")
    run_logging_dir = os.path.join(cfg["logging_dir"], f"run_{run_idx}")
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_logging_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=run_output_dir,
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=float(cfg["learning_rate"]),
        logging_dir=run_logging_dir,
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        eval_strategy="epoch",  # To evaluate every epoch
        load_best_model_at_end=True,  # Load best model based on MSE
        metric_for_best_model="mse",  # Use MSE for best model selection
        greater_is_better=False,      # Lower MSE is better
        report_to="none",
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=float(cfg["weight_decay"]),
    )

    # Fresh model for each run
    if cfg["model_name"]:
        hf_config = GPT2Config.from_pretrained(cfg["model_name"])
    else:
        hf_config = GPT2Config(
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            vocab_size=cfg["vocab_size"],
        )

    model = ContinuousGPT2(hf_config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])
    #model = ContinuousRotaryTransformer(hf_config, cfg["num_x_tokens"], cfg["x_dim"], cfg["v_dim"], cfg["y_dim"])
    #model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=20,  # Stop after 10 evaluations with no improvement
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[SimpleLoggerCallback(),
                  EarlyStoppingCallback(early_stopping_patience=20),],
        compute_metrics=compute_mse,
    )

    trainer.train()
    trainer.save_model()
