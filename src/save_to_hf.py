#!/usr/bin/env python3
"""
Script to save a trained image captioning model to Hugging Face Hub.
"""

import torch
import argparse
import os
from typing import Optional

from src.train import ImageCaptioningModel, save_model_to_huggingface


def main():
    parser = argparse.ArgumentParser(description="Save trained model to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--repo-name", type=str, default=None,
                       help="Repository name (format: username/model-name)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token (if not provided, will use HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--commit-message", type=str, default=None,
                       help="Commit message for the upload (default: auto-generated)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # Get values from environment variables if not provided
    if args.token is None:
        args.token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if args.repo_name is None:
        args.repo_name = os.environ.get('HF_REPO_NAME')
    
    if args.commit_message is None:
        args.commit_message = os.environ.get('HF_COMMIT_MESSAGE', 'Add image captioning model')
    
    if not args.private:
        # Check if private flag is set via environment
        args.private = os.environ.get('HF_PRIVATE', '').lower() in ('true', '1', 'yes')
    
    if args.device is None:
        args.device = os.environ.get('DEVICE')
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Select device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    print("🔄 Loading model from checkpoint...")
    model = ImageCaptioningModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Checkpoint info:")
    if 'epoch' in checkpoint:
        print(f"   - Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"   - Validation Loss: {checkpoint['val_loss']:.4f}")
    if 'eval_metrics' in checkpoint:
        print(f"   - Evaluation Metrics:")
        for metric, score in checkpoint['eval_metrics'].items():
            print(f"     {metric}: {score:.4f}")
    
    # Save to Hugging Face Hub
    print(f"\n🔄 Saving model to Hugging Face Hub: {args.repo_name}")
    success = save_model_to_huggingface(
        model=model,
        repo_name=args.repo_name,
        token=args.token,
        commit_message=args.commit_message,
        private=args.private
    )
    
    if success:
        print("✅ Model successfully uploaded to Hugging Face Hub!")
        print(f"🌐 View your model at: https://huggingface.co/{args.repo_name}")
    else:
        print("❌ Failed to upload model to Hugging Face Hub")


if __name__ == "__main__":
    main() 