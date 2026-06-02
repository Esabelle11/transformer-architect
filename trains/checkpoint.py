import torch
import os

def save_checkpoint(path, model, optimizer, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt