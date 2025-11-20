"""
Quick verification script to test LoRA implementation
Run this to verify everything is set up correctly BEFORE training.
"""

import sys

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        
        import tqdm
        print(f"✓ tqdm")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nRun: pip install -r requirements.txt")
        return False


def test_cuda():
    """Test CUDA availability."""
    import torch
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available - will use CPU (slower but works)")
    return True


def test_modules():
    """Test that our custom modules load correctly."""
    print("\nTesting custom modules...")
    try:
        from lora_layer import LoRALinear
        print("✓ lora_layer.py")
        
        from lora_model import inject_lora_into_model
        print("✓ lora_model.py")
        
        from config import get_sst2_config
        print("✓ config.py")
        
        from dataset import get_glue_dataloaders
        print("✓ dataset.py")
        
        from train import train_lora
        print("✓ train.py")
        
        from evaluate import evaluate_lora_model
        print("✓ evaluate.py")
        
        return True
    except ImportError as e:
        print(f"✗ Module import failed: {e}")
        return False


def test_lora_layer():
    """Test LoRA layer creation."""
    print("\nTesting LoRA layer...")
    try:
        import torch
        import torch.nn as nn
        from lora_layer import LoRALinear
        
        # Create a regular Linear layer
        linear = nn.Linear(768, 768)
        
        # Convert to LoRA
        lora_linear = LoRALinear.from_linear(linear, r=8, lora_alpha=16)
        
        # Test forward pass
        x = torch.randn(2, 10, 768)
        output = lora_linear(x)
        
        assert output.shape == (2, 10, 768), "Output shape mismatch"
        
        # Check that base weights are frozen
        assert not lora_linear.linear.weight.requires_grad, "Base weights not frozen"
        
        # Check that LoRA weights are trainable
        assert lora_linear.lora_A.requires_grad, "LoRA A not trainable"
        assert lora_linear.lora_B.requires_grad, "LoRA B not trainable"
        
        print("✓ LoRA layer works correctly")
        print(f"  Base params: {lora_linear.linear.weight.numel():,} (frozen)")
        print(f"  LoRA params: {lora_linear.lora_A.numel() + lora_linear.lora_B.numel():,} (trainable)")
        
        return True
    except Exception as e:
        print(f"✗ LoRA layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test that we can load a model."""
    print("\nTesting model loading...")
    try:
        from transformers import AutoModelForSequenceClassification
        
        print("  Loading DistilBERT (this may take a minute)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("  This might be due to internet connectivity")
        return False


def test_injection():
    """Test LoRA injection into model."""
    print("\nTesting LoRA injection...")
    try:
        from transformers import AutoModelForSequenceClassification
        from lora_model import inject_lora_into_model, print_trainable_parameters
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        # Inject LoRA
        model = inject_lora_into_model(
            model,
            target_modules=['q_lin', 'v_lin'],
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        # Check parameters
        trainable, total, percent = print_trainable_parameters(model)
        
        if percent < 2.0:
            print(f"✓ LoRA injection successful!")
            print(f"  Only {percent:.2f}% parameters trainable (target: <2%)")
        else:
            print(f"⚠ More parameters trainable than expected: {percent:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ LoRA injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("LoRA Implementation Verification")
    print("="*60)
    
    results = []
    
    results.append(("Package imports", test_imports()))
    if not results[-1][1]:
        print("\n" + "="*60)
        print("FAILED: Install dependencies first")
        print("Run: pip install -r requirements.txt")
        print("="*60)
        return
    
    results.append(("CUDA check", test_cuda()))
    results.append(("Custom modules", test_modules()))
    
    if not results[-1][1]:
        print("\n" + "="*60)
        print("FAILED: Make sure you're in the project2-lora-implementation directory")
        print("="*60)
        return
    
    results.append(("LoRA layer", test_lora_layer()))
    results.append(("Model loading", test_model_loading()))
    results.append(("LoRA injection", test_injection()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*60)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou're ready to train!")
        print("Next step: jupyter notebook train_lora.ipynb")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Fix the issues above before training")
    print("="*60)


if __name__ == '__main__':
    main()
