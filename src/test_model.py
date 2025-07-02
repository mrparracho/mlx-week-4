#!/usr/bin/env python3
"""
Test script to evaluate the trained image captioning model on random samples.
"""

import torch
import argparse
import os
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from src.train import ImageCaptioningModel, select_device
from src.data.flickr_dataset import FlickrDataset


def test_model_on_random_samples(
    model: ImageCaptioningModel,
    test_dataset: FlickrDataset,
    device: str,
    num_samples: int = 3,
    save_plot: bool = True,
    plot_path: str = "test_results.png",
    temperature: float = 0.8,
    max_length: int = 50,
    num_beams: int = 5
):
    """
    Test the model on random samples from the dataset.
    """
    model.eval()
    
    # Get random samples
    dataset_size = len(test_dataset)
    print(f"📁 Dataset size: {dataset_size}")
    
    if dataset_size == 0:
        print("❌ Error: No samples found in test dataset!")
        print("💡 This might be because:")
        print("   - The test split is empty")
        print("   - The dataset path is incorrect")
        print("   - The dataset hasn't been downloaded")
        print("   - Try using 'val' split instead: --data-dir data/flickr30k --split val")
        return
    
    if dataset_size < num_samples:
        num_samples = dataset_size
        print(f"⚠️  Warning: Only {num_samples} samples available in dataset")
    
    # Get random indices
    random_indices = random.sample(range(dataset_size), num_samples)
    selected_samples = [test_dataset[idx] for idx in random_indices]
    
    # Create subplot
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL ON {num_samples} RANDOM SAMPLES")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for i, (image, caption, _) in enumerate(selected_samples):
            # Convert tensor to PIL for generation
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            # Generate caption
            generated_caption = model.generate_caption(
                image, 
                max_length=max_length, 
                num_beams=num_beams, 
                temperature=temperature
            )
            
            # Display results
            print(f"\n📸 Sample {i+1}:")
            print(f"   Generated: {generated_caption}")
            print(f"   Reference: {caption}")
            
            # Plot image with captions
            ax = axes[i]
            ax.imshow(image)
            ax.set_title(f"Sample {i+1}", fontsize=12, fontweight='bold')
            
            # Add captions as text
            caption_text = f"Generated: {generated_caption}\nReference: {caption}"
            
            ax.text(0.5, -0.15, caption_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.axis('off')
    
    # Save plot
    if save_plot:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📁 Plot saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test image captioning model on random samples")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--data-dir", type=str, default="data/flickr30k", 
                       help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                       help="Dataset split to use (train, val, test)")
    parser.add_argument("--num-samples", type=int, default=3, 
                       help="Number of random samples to test")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--plot-path", type=str, default="test_results.png",
                       help="Path to save the plot")
    parser.add_argument("--no-plot", action="store_true",
                       help="Don't save the plot")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Temperature for text generation (0.1-2.0)")
    parser.add_argument("--max-length", type=int, default=50, 
                       help="Maximum caption length")
    parser.add_argument("--num-beams", type=int, default=5, 
                       help="Number of beams for beam search")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    device = args.device if args.device else select_device()
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
    
    # Create test dataset
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print(f"📁 Loading {args.split} split from {args.data_dir}...")
    test_dataset = FlickrDataset(
        root_dir=args.data_dir,
        split=args.split,
        tokenizer=model.tokenizer,
        transform=image_transforms
    )
    
    print(f"📁 {args.split.capitalize()} dataset size: {len(test_dataset)}")
    
    # Test model
    test_model_on_random_samples(
        model=model,
        test_dataset=test_dataset,
        device=device,
        num_samples=args.num_samples,
        save_plot=not args.no_plot,
        plot_path=args.plot_path,
        temperature=args.temperature,
        max_length=args.max_length,
        num_beams=args.num_beams
    )


if __name__ == "__main__":
    main() 