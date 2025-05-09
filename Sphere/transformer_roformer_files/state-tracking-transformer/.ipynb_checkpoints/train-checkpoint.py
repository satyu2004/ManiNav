import yaml
from transformers import GPT2Config, TrainingArguments, Trainer
from src.model import ContinuousGPT2
from src.collator import collate_fn
from src.callbacks import SimpleLoggerCallback

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.dataset import MyTrajectoryDataset
import torch

# Load real data
X0 = torch.load(f"{cfg['data_path']}/X0.pt")[:cfg["n_trajectories"]].float()
V = torch.load(f"{cfg['data_path']}/V.pt")[:cfg["n_trajectories"]].float()
pos = torch.load(f"{cfg['data_path']}/pos.pt")[:cfg["n_trajectories"]].float()

dataset = MyTrajectoryDataset(X0, V, pos)

'''
# By saving the data as h5 files instead of pt files you can load lazily and avoid some error creep. 
import h5py

class MyTrajectoryDataset(Dataset):
    def __init__(self, X0_file, V_file, pos_file):
        # Use h5py to lazily load data from disk
        self.X0_file = h5py.File(X0_file, 'r')
        self.V_file = h5py.File(V_file, 'r')
        self.pos_file = h5py.File(pos_file, 'r')
        
        self.n_trajectories = len(self.X0_file['X0'])  # Assuming each dataset is a key in h5py

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, idx):
        X0 = torch.tensor(self.X0_file['X0'][idx])  # Lazy loading data from disk
        V = torch.tensor(self.V_file['V'][idx])
        pos = torch.tensor(self.pos_file['pos'][idx])
        
        return {
            "x_inputs": X0.float(),
            "v_inputs": V.float(),
            "labels": pos.float(),
        }

# Usage
dataset = MyTrajectoryDataset('path_to_X0.h5', 'path_to_V.h5', 'path_to_pos.h5')
'''


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

args = TrainingArguments(
    output_dir=cfg["output_dir"],
    per_device_train_batch_size=cfg["batch_size"],
    num_train_epochs=cfg["num_epochs"],
    learning_rate=float(cfg["learning_rate"]),
    logging_dir=cfg["logging_dir"],
    logging_steps=cfg["logging_steps"],
    save_strategy=cfg["save_strategy"],
    report_to="none",
    lr_scheduler_type=cfg["lr_scheduler_type"],  # Cosine decay scheduler
    weight_decay=float(cfg["weight_decay"]),  # Common default weight decay for transformer models
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    callbacks=[SimpleLoggerCallback()],
)

trainer.train()
