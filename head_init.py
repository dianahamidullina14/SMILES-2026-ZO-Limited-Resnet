import torch
import torch.nn as nn
import os


def init_last_layer(layer: nn.Linear) -> None:
    path = "/kaggle/working/pinv_head.pt"
    if os.path.exists(path):
        weights = torch.load(path, map_location="cpu")
        with torch.no_grad():
            layer.weight.copy_(weights["weight"])
            layer.bias.copy_(weights["bias"])
        print("  [head_init] Loaded pseudo-inverse weights ")
    else:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        print("  [head_init] Xavier init (fallback)")
