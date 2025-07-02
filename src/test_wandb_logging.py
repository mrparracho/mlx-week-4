#!/usr/bin/env python3
"""
Test script for W&B logging functionality.

This script tests the new W&B image logging and model checkpoint logging features.
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import (
    ImageCaptioningModel, 
    log_test_images_to_wandb, 
    log_model_checkpoint_to_wandb,
    init_wandb,
    finish_wandb
)


def create_test_image(width=224, height=224):
    """Create a test image for demonstration."""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_wandb_image_logging():
    """Test the W&B image logging functionality."""
    print("🧪 Testing W&B Image Logging...")
    
    # Initialize W&B
    config = {
        "test": True,
        "feature": "image_logging"
    }
    
    wandb_initialized = init_wandb(
        project_name="test-image-captioning",
        run_name="test-wandb-logging",
        config=config
    )
    
    if not wandb_initialized:
        print("❌ W&B not initialized, skipping test")
        return False
    
    try:
        # Create test samples
        test_samples = []
        for i in range(3):
            image = create_test_image()
            sample = {
                'image': image,
                'reference_caption': f"Test reference caption {i+1}",
                'generated_caption': f"Test generated caption {i+1}",
                'metrics': {
                    'BLEU-4': 0.1 + i * 0.1,
                    'METEOR': 0.2 + i * 0.1,
                    'ROUGE-L': 0.3 + i * 0.1
                }
            }
            test_samples.append(sample)
        
        # Test image logging
        log_test_images_to_wandb(test_samples, epoch=1, step=100)
        
        print("✅ W&B image logging test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ W&B image logging test failed: {e}")
        return False
    finally:
        finish_wandb()


def test_wandb_checkpoint_logging():
    """Test the W&B checkpoint logging functionality."""
    print("🧪 Testing W&B Checkpoint Logging...")
    
    # Initialize W&B
    config = {
        "test": True,
        "feature": "checkpoint_logging"
    }
    
    wandb_initialized = init_wandb(
        project_name="test-image-captioning",
        run_name="test-checkpoint-logging",
        config=config
    )
    
    if not wandb_initialized:
        print("❌ W&B not initialized, skipping test")
        return False
    
    try:
        # Create a dummy model and save it
        model = ImageCaptioningModel()
        
        # Create a dummy checkpoint
        checkpoint_path = "test_checkpoint.pth"
        torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'val_loss': 2.5,
        }, checkpoint_path)
        
        # Test checkpoint logging
        log_model_checkpoint_to_wandb(
            checkpoint_path=checkpoint_path,
            epoch=1,
            save_criterion='test_checkpoint',
            val_loss=2.5,
            eval_metrics={'BLEU-4': 0.3, 'METEOR': 0.4},
            step=100
        )
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        print("✅ W&B checkpoint logging test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ W&B checkpoint logging test failed: {e}")
        return False
    finally:
        finish_wandb()


def main():
    """Main test function."""
    print("🚀 Testing W&B Logging Functionality")
    print("=" * 50)
    
    # Check if WANDB_API_KEY is set
    if not os.environ.get('WANDB_API_KEY'):
        print("⚠️  WANDB_API_KEY not set. Set it to test W&B functionality:")
        print("   export WANDB_API_KEY='your-api-key'")
        print("   Or run: wandb login")
        return
    
    # Run tests
    image_test_passed = test_wandb_image_logging()
    checkpoint_test_passed = test_wandb_checkpoint_logging()
    
    print("\n📊 Test Results:")
    print(f"   Image Logging: {'✅ PASSED' if image_test_passed else '❌ FAILED'}")
    print(f"   Checkpoint Logging: {'✅ PASSED' if checkpoint_test_passed else '❌ FAILED'}")
    
    if image_test_passed and checkpoint_test_passed:
        print("\n🎉 All W&B logging tests passed!")
        print("   You can now use --wandb flag during training to log:")
        print("   - Test images with captions")
        print("   - Model checkpoints as artifacts")
        print("   - Training metrics and visualizations")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main() 