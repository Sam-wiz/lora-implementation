"""
Evaluation script for LoRA fine-tuned models
"""

import torch
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm.auto import tqdm
import json
import os

from config import get_sst2_config
from dataset import get_glue_dataloaders
from lora_model import inject_lora_into_model, load_lora_state_dict


def evaluate_lora_model(checkpoint_path, config=None):
    """
    Load and evaluate a LoRA fine-tuned model.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint (.pt file)
        config: Configuration (uses default if None)
    """
    # Load config
    if config is None:
        config = get_sst2_config()
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading validation dataset...")
    _, val_loader, num_labels, tokenizer = get_glue_dataloaders(
        task=config.training.task,
        model_name=config.model.model_name,
        batch_size=config.training.batch_size,
        max_length=config.training.max_length
    )
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=num_labels
    )
    
    # Inject LoRA
    print("Injecting LoRA layers...")
    model = inject_lora_into_model(
        model,
        target_modules=config.lora.target_modules,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout
    )
    
    # Load LoRA weights
    print(f"Loading LoRA checkpoint from {checkpoint_path}...")
    lora_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(lora_state, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("\nEvaluating...")
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(val_loader)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'loss': avg_loss
    }


def compare_lora_vs_full(lora_checkpoint, full_checkpoint, config=None):
    """
    Compare LoRA vs full fine-tuning performance.
    """
    print("="*60)
    print("Comparing LoRA vs Full Fine-tuning")
    print("="*60)
    
    # Evaluate LoRA
    print("\n1. Evaluating LoRA model...")
    lora_results = evaluate_lora_model(lora_checkpoint, config)
    
    # Evaluate full model
    print("\n2. Evaluating full fine-tuned model...")
    # Load full model (simplified - assumes same architecture)
    if config is None:
        config = get_sst2_config()
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    _, val_loader, num_labels, _ = get_glue_dataloaders(
        task=config.training.task,
        model_name=config.model.model_name,
        batch_size=config.training.batch_size,
        max_length=config.training.max_length
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=num_labels
    )
    model.load_state_dict(torch.load(full_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating full model"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    full_accuracy = accuracy_score(all_labels, all_predictions)
    full_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Compare
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"{'Metric':<15} {'LoRA':<15} {'Full':<15} {'Retention':<15}")
    print("-"*60)
    print(f"{'Accuracy':<15} {lora_results['accuracy']:<15.4f} {full_accuracy:<15.4f} {(lora_results['accuracy']/full_accuracy)*100:.2f}%")
    print(f"{'F1 Score':<15} {lora_results['f1']:<15.4f} {full_f1:<15.4f} {(lora_results['f1']/full_f1)*100:.2f}%")
    print("="*60)
    
    return lora_results, {'accuracy': full_accuracy, 'f1': full_f1}


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <checkpoint_path>")
        print("Example: python evaluate.py outputs/lora_20260202_121500/best_lora.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Evaluate
    results = evaluate_lora_model(checkpoint_path)
