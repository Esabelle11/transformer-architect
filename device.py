import torch
from torch import nn

def device_select():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" # MPS = Metal Performance Shaders
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return device



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # torch.nn.init.xavier_uniform_(m.weight)
