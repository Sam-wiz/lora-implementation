"""
Dataset loading and preprocessing for GLUE tasks
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Tuple


class GLUEDataset:
    """
    Wrapper for GLUE datasets with tokenization.
    """
    def __init__(self, task: str, model_name: str, max_length: int = 128):
        self.task = task.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Task-specific configurations
        self.task_to_keys = {
            'sst2': ('sentence', None),
            'mrpc': ('sentence1', 'sentence2'),
            'cola': ('sentence', None),
            'qnli': ('question', 'sentence'),
            'qqp': ('question1', 'question2'),
            'rte': ('sentence1', 'sentence2'),
            'wnli': ('sentence1', 'sentence2'),
        }
        
        if self.task not in self.task_to_keys:
            raise ValueError(f"Task {task} not supported. Choose from: {list(self.task_to_keys.keys())}")
        
        self.sentence1_key, self.sentence2_key = self.task_to_keys[self.task]
        
        # Load dataset
        print(f"Loading {task} dataset...")
        if self.task == 'sst2':
            self.dataset = load_dataset('glue', 'sst2')
        elif self.task == 'mrpc':
            self.dataset = load_dataset('glue', 'mrpc')
        elif self.task == 'cola':
            self.dataset = load_dataset('glue', 'cola')
        elif self.task == 'qnli':
            self.dataset = load_dataset('glue', 'qnli')
        elif self.task == 'qqp':
            self.dataset = load_dataset('glue', 'qqp')
        elif self.task == 'rte':
            self.dataset = load_dataset('glue', 'rte')
        elif self.task == 'wnli':
            self.dataset = load_dataset('glue', 'wnli')
        
        print(f"✓ Loaded {task} dataset")
        print(f"  Train: {len(self.dataset['train'])} examples")
        print(f"  Validation: {len(self.dataset['validation'])} examples")
        if 'test' in self.dataset:
            print(f"  Test: {len(self.dataset['test'])} examples")
    
    def preprocess_function(self, examples):
        """
        Tokenize examples based on task type.
        """
        # Get sentences
        if self.sentence2_key is None:
            # Single sentence tasks
            texts = examples[self.sentence1_key]
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            # Sentence pair tasks
            texts1 = examples[self.sentence1_key]
            texts2 = examples[self.sentence2_key]
            tokenized = self.tokenizer(
                texts1,
                texts2,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        return tokenized
    
    def get_dataloaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation dataloaders.
        """
        # Tokenize datasets
        train_dataset = self.dataset['train'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names
        )
        val_dataset = self.dataset['validation'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset['validation'].column_names
        )
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch')
        val_dataset.set_format(type='torch')
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility, increase if needed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def get_num_labels(self) -> int:
        """
        Get number of labels for the task.
        """
        label_list = self.dataset['train'].features['label'].names
        return len(label_list)


def get_glue_dataloaders(task: str, model_name: str, batch_size: int = 32, max_length: int = 128):
    """
    Convenience function to get dataloaders.
    """
    dataset = GLUEDataset(task, model_name, max_length)
    train_loader, val_loader = dataset.get_dataloaders(batch_size)
    num_labels = dataset.get_num_labels()
    
    return train_loader, val_loader, num_labels, dataset.tokenizer
