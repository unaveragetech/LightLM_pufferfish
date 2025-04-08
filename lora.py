import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=32):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        self.scaling = alpha / rank
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora

def add_lora_layers(model, rank=8, alpha=32, target_modules=["q_proj", "v_proj"]):
    """Apply LoRA to specific layers in the model"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    return model