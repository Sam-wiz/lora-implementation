"""
LoRA Model Wrapper
Functions to inject LoRA layers into pretrained models.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from lora_layer import LoRALinear, mark_only_lora_as_trainable, count_parameters


def inject_lora_into_model(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """
    Inject LoRA layers into a pretrained model.
    
    Args:
        model: Pretrained model (e.g., DistilBERT)
        target_modules: List of module names to replace with LoRA
                       e.g., ['q_lin', 'v_lin'] for attention query and value
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
    
    Returns:
        Model with LoRA layers injected
    """
    # Recursively replace target Linear layers with LoRALinear
    for name, module in model.named_modules():
        # Check if this module's name contains any target module string
        if any(target in name for target in target_modules):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Check if it's a Linear layer
            if isinstance(module, nn.Linear):
                # Replace with LoRALinear
                lora_layer = LoRALinear.from_linear(
                    module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(parent, attr_name, lora_layer)
                print(f"✓ Injected LoRA into: {name}")
    
    # Freeze all non-LoRA parameters
    mark_only_lora_as_trainable(model, bias='none')
    
    return model


def print_trainable_parameters(model: nn.Module):
    """
    Print the number of trainable parameters in the model.
    """
    trainable, total = count_parameters(model)
    trainable_percent = 100 * trainable / total
    
    print(f"\n{'='*60}")
    print(f"Parameter Statistics:")
    print(f"{'='*60}")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {trainable_percent:.4f}%")
    print(f"Reduction:        {100 - trainable_percent:.2f}% parameters frozen")
    print(f"{'='*60}\n")
    
    return trainable, total, trainable_percent


def get_lora_state_dict(model: nn.Module):
    """
    Extract only LoRA parameters for saving.
    Much smaller checkpoint since we don't save frozen weights.
    """
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict, strict: bool = False):
    """
    Load LoRA parameters into a model.
    """
    return model.load_state_dict(lora_state_dict, strict=strict)
