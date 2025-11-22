"""
LoRA Layer Implementation
Implements Low-Rank Adaptation as described in Hu et al. (2021)

Key equation: h = W_0 * x + (B * A) * x * (alpha / r)
Where:
- W_0: frozen pretrained weights
- B: trainable matrix (d x r)
- A: trainable matrix (r x k)
- r: rank (r << min(d, k))
- alpha: scaling factor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """
    Base LoRA layer that adds low-rank adaptation to any layer.
    """
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.scaling = self.lora_alpha / self.r


class LoRALinear(nn.Module):
    """
    LoRA applied to Linear layer.
    Replaces nn.Linear while keeping original weights frozen.
    
    Forward pass: output = W_0 @ x + (B @ A @ x) * scaling
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = False,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.merge_weights = merge_weights
        
        # Create the original linear layer (will be frozen)
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        
        # LoRA low-rank matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = self.lora_alpha / self.r
            
            # Initialize A with Kaiming uniform (like nn.Linear weight init)
            # Initialize B with zeros (so initially LoRA produces zero output)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
        # Freeze the original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original output + LoRA adaptation
        """
        # Original linear transformation (frozen)
        result = self.linear(x)
        
        # Add LoRA adaptation if rank > 0
        if self.r > 0:
            # LoRA path: x -> A -> dropout -> B -> scale
            lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_output * self.scaling
        
        return result
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        """
        Create LoRALinear from existing nn.Linear layer, preserving weights.
        """
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None
        )
        
        # Copy the original weights
        lora_linear.linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_linear.linear.bias.data = linear.bias.data.clone()
        
        return lora_linear


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freeze all parameters except LoRA parameters.
    
    Args:
        model: The model to modify
        bias: How to handle bias terms ('none', 'all', 'lora_only')
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    
    if bias == 'none':
        return
    elif bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        for name, param in model.named_parameters():
            if 'bias' in name and 'lora_' in name:
                param.requires_grad = True
    else:
        raise ValueError(f"Invalid bias option: {bias}")


def get_lora_parameters(model: nn.Module):
    """
    Get all LoRA parameters from a model.
    """
    return [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]


def count_parameters(model: nn.Module):
    """
    Count trainable and total parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
