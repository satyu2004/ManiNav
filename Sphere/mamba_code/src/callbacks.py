# src/callbacks.py
import numpy as np

class SimpleLogger:
    def __init__(self, log_every=10):
        self.log_every = log_every
        self.best_eval_loss = float("inf")

    def log_step(self, epoch, step, loss):
        if step % self.log_every == 0:
            print(f"[Epoch {epoch} | Step {step}] Loss: {loss:.4f}")

    def log_epoch(self, epoch, avg_loss):
        print(f"[Epoch {epoch} completed] Avg Loss: {avg_loss:.4f}")

    def log_eval(self, epoch, eval_loss, checkpoint_dir=None, model=None):
        print(f"[Eval @ Epoch {epoch}] Test Loss: {eval_loss:.4f}")

        if eval_loss < self.best_eval_loss:
            print(f"  🥇 New best model! Previous best: {self.best_eval_loss:.4f}")
            self.best_eval_loss = eval_loss

            #if model is not None and checkpoint_dir is not None:
            #    ckpt_path = f"{checkpoint_dir}/best_model.pt"
            #    torch.save(model.state_dict(), ckpt_path)
            #    print(f"  💾 Saved best model to {ckpt_path}")
