#!/usr/bin/env python3
"""
Model utilities for the dual model saving approach.

This script demonstrates how to use the new model loading and management functions.
"""

import os
import sys
import argparse
from typing import Optional

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import ImageCaptioningModel, load_best_model, list_available_models


def compare_models(checkpoint_dir: str = "checkpoints"):
    """
    Compare the available models and their metadata.
    
    Args:
        checkpoint_dir: Directory containing saved models
    """
    print("🔍 Comparing available models...")
    print("=" * 60)
    
    models_info = list_available_models(checkpoint_dir)
    
    if not models_info:
        print("❌ No saved models found!")
        return
    
    # Compare evaluation vs loss models
    eval_model = models_info.get('best_model_eval.pth')
    loss_model = models_info.get('best_model_loss.pth')
    
    if eval_model and loss_model:
        print("📊 Model Comparison:")
        print(f"  Evaluation Model (epoch {eval_model['epoch']}):")
        print(f"    Save criterion: {eval_model['save_criterion']}")
        print(f"    Validation loss: {eval_model['val_loss']:.4f}")
        if 'eval_metrics' in eval_model:
            print("    Evaluation metrics:")
            for metric, score in eval_model['eval_metrics'].items():
                print(f"      {metric}: {score:.4f}")
        
        print(f"\n  Loss Model (epoch {loss_model['epoch']}):")
        print(f"    Save criterion: {loss_model['save_criterion']}")
        print(f"    Validation loss: {loss_model['val_loss']:.4f}")
        
        # Determine which model has better loss
        if loss_model['val_loss'] < eval_model['val_loss']:
            print(f"\n✅ Loss model has better validation loss by {eval_model['val_loss'] - loss_model['val_loss']:.4f}")
        else:
            print(f"\n✅ Evaluation model has better validation loss by {loss_model['val_loss'] - eval_model['val_loss']:.4f}")
    
    elif eval_model:
        print("📊 Found evaluation model only:")
        print(f"  Epoch: {eval_model['epoch']}")
        print(f"  Save criterion: {eval_model['save_criterion']}")
        print(f"  Validation loss: {eval_model['val_loss']:.4f}")
    
    elif loss_model:
        print("📊 Found loss model only:")
        print(f"  Epoch: {loss_model['epoch']}")
        print(f"  Save criterion: {loss_model['save_criterion']}")
        print(f"  Validation loss: {loss_model['val_loss']:.4f}")
    
    # Show available checkpoints
    if 'checkpoints' in models_info:
        print(f"\n📁 Available checkpoints: {len(models_info['checkpoints'])}")
        for checkpoint in sorted(models_info['checkpoints']):
            print(f"  - {checkpoint}")


def load_and_test_model(checkpoint_dir: str = "checkpoints", 
                       model_type: str = "auto",
                       test_image: Optional[str] = None):
    """
    Load a model and optionally test it on an image.
    
    Args:
        checkpoint_dir: Directory containing saved models
        model_type: Type of model to load ("eval", "loss", or "auto")
        test_image: Optional path to test image
    """
    print(f"🔄 Loading {model_type}-based model...")
    
    try:
        # Initialize model
        model = ImageCaptioningModel()
        
        # Load the model
        model, metadata = load_best_model(model, checkpoint_dir, model_type)
        
        print("✅ Model loaded successfully!")
        
        # Test on image if provided
        if test_image and os.path.exists(test_image):
            print(f"\n🖼️  Testing on image: {test_image}")
            from PIL import Image
            
            image = Image.open(test_image)
            caption = model.generate_caption(image)
            print(f"Generated caption: {caption}")
        
        return model, metadata
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def main():
    """Main function for the model utilities script."""
    parser = argparse.ArgumentParser(description="Model utilities for dual model saving")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory containing saved models")
    parser.add_argument("--action", type=str, choices=["compare", "load", "list"], 
                       default="compare", help="Action to perform")
    parser.add_argument("--model-type", type=str, choices=["eval", "loss", "auto"],
                       default="auto", help="Type of model to load")
    parser.add_argument("--test-image", type=str, help="Path to test image")
    
    args = parser.parse_args()
    
    if args.action == "compare":
        compare_models(args.checkpoint_dir)
    
    elif args.action == "load":
        load_and_test_model(args.checkpoint_dir, args.model_type, args.test_image)
    
    elif args.action == "list":
        models_info = list_available_models(args.checkpoint_dir)
        print("📁 Available models:")
        for model_name, info in models_info.items():
            if model_name == 'checkpoints':
                print(f"  {model_name}: {len(info)} files")
            else:
                print(f"  {model_name}: epoch {info['epoch']}, {info['save_criterion']}")


if __name__ == "__main__":
    main() 