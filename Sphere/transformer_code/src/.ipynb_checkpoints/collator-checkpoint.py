import torch

def collate_fn(batch):
    x_inputs = torch.stack([item["x_inputs"] for item in batch])
    v_inputs = torch.stack([item["v_inputs"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.ones((x_inputs.size(0), 3 + v_inputs.size(1)))
    return {
        "x_inputs": x_inputs,
        "v_inputs": v_inputs,
        "labels": labels,
        "attention_mask": attention_mask
    }
