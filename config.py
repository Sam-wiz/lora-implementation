"""
Configuration for LoRA fine-tuning
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA hyperparameters"""
    r: int = 8  # Rank of LoRA matrices
    lora_alpha: int = 16  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    target_modules: List[str] = field(default_factory=lambda: ['q_lin', 'v_lin'])  # Which layers to adapt
    bias: str = 'none'  # Bias training: 'none', 'all', 'lora_only'


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = 'distilbert-base-uncased'  # HuggingFace model name
    num_labels: int = 2  # Number of classes for classification


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Dataset
    task: str = 'sst2'  # GLUE task: sst2, mrpc, cola, etc.
    max_length: int = 128  # Max sequence length
    
    # Training
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 3e-4  # Higher LR for LoRA (paper uses 3e-4)
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Logging & Saving
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    output_dir: str = './outputs'
    
    # Compute
    device: str = 'cuda'  # 'cuda' or 'cpu'
    seed: int = 42


@dataclass
class Config:
    """Complete configuration"""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Default configuration
def get_default_config():
    return Config()


# DistilBERT on SST-2 (recommended starting point)
def get_sst2_config():
    config = Config()
    config.model.model_name = 'distilbert-base-uncased'
    config.model.num_labels = 2
    config.training.task = 'sst2'
    config.training.num_epochs = 3
    config.training.batch_size = 32
    config.lora.r = 8
    config.lora.lora_alpha = 16
    config.lora.target_modules = ['q_lin', 'v_lin']  # DistilBERT attention
    return config


# DistilBERT on MRPC (alternative task)
def get_mrpc_config():
    config = Config()
    config.model.model_name = 'distilbert-base-uncased'
    config.model.num_labels = 2
    config.training.task = 'mrpc'
    config.training.num_epochs = 5
    config.training.batch_size = 32
    config.lora.r = 8
    config.lora.lora_alpha = 16
    config.lora.target_modules = ['q_lin', 'v_lin']
    return config
