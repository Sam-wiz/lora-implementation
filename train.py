"""
Training script for LoRA fine-tuning
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import os
import json
from datetime import datetime

from config import get_sst2_config
from dataset import get_glue_dataloaders
from lora_model import inject_lora_into_model, print_trainable_parameters, get_lora_state_dict


def train_lora(config=None, use_lora=True):
    """
    Train a model with LoRA adaptation.
    
    Args:
        config: Configuration object (uses default if None)
        use_lora: If True, use LoRA. If False, full fine-tuning (for baseline comparison)
    """
    # Load config
    if config is None:
        config = get_sst2_config()
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config.training.seed)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    train_loader, val_loader, num_labels, tokenizer = get_glue_dataloaders(
        task=config.training.task,
        model_name=config.model.model_name,
        batch_size=config.training.batch_size,
        max_length=config.training.max_length
    )
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=num_labels
    )
    
    # Inject LoRA if enabled
    if use_lora:
        print(f"\nInjecting LoRA (r={config.lora.r}, alpha={config.lora.lora_alpha})...")
        model = inject_lora_into_model(
            model,
            target_modules=config.lora.target_modules,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout
        )
    else:
        print("\nUsing FULL fine-tuning (baseline mode)")
    
    model = model.to(device)
    
    # Print parameter statistics
    trainable, total, percent = print_trainable_parameters(model)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "lora" if use_lora else "full"
    output_dir = os.path.join(config.training.output_dir, f"{mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'lora': config.lora.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'use_lora': use_lora,
            'trainable_params': trainable,
            'total_params': total,
            'trainable_percent': percent
        }, f, indent=2)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_val_acc = 0.0
    global_step = 0
    training_history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Logging
            if global_step % config.training.logging_steps == 0:
                avg_loss = train_loss / train_steps
                training_history['train_loss'].append(avg_loss)
        
        # Validation
        print("\nRunning validation...")
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss/train_steps:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best validation accuracy: {best_val_acc:.4f}")
            
            if use_lora:
                # Save only LoRA parameters (much smaller!)
                lora_state = get_lora_state_dict(model)
                torch.save(lora_state, os.path.join(output_dir, 'best_lora.pt'))
            else:
                # Save full model
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")
    
    return model, training_history, output_dir


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
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
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


if __name__ == '__main__':
    # Train with LoRA
    print("Training with LoRA...")
    model, history, output_dir = train_lora(use_lora=True)
    
    print("\n" + "="*60)
    print("Training complete! Check output directory for results.")
    print("="*60)
