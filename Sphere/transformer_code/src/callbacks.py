from transformers import TrainerCallback

class SimpleLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        epoch = int(state.epoch) if state.epoch is not None else -1
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        mse = logs.get("mse")

        if loss is not None:
            print(f"[Epoch {epoch}] Training Loss: {loss:.4f}")
        elif eval_loss is not None:
            print(f"[Epoch {epoch}] Eval Loss: {eval_loss:.4f}")
        elif mse is not None:
            print(f"[Epoch {epoch}] Eval MSE: {mse:.4f}")
        else:
            print(f"[Epoch {epoch}] No loss or metric found to log.")