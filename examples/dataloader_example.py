#!/usr/bin/env python3
"""
Example script demonstrating how to use the Flickr dataset and dataloader.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.flickr_dataset import FlickrDataset, create_sample_flickr_data
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch


def main():
    """Demonstrate the Flickr dataset usage."""
    
    # Configuration
    data_dir = "data/flickr_dataset"
    
    # Create sample data if it doesn't exist
    if not os.path.exists(data_dir):
        print("Creating sample Flickr dataset...")
        create_sample_flickr_data(data_dir, num_samples=50)
        print("Sample dataset created! Please add actual images to proceed.")
        return
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = FlickrDataset(
        root_dir=data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=77
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    # Iterate through a few batches
    print("\nIterating through batches:")
    for batch_idx, (images, captions, caption_tokens) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape if hasattr(images, 'shape') else 'PIL Images'}")
        print(f"  Number of captions: {len(captions)}")
        print(f"  Caption tokens shape: {caption_tokens.shape if caption_tokens is not None else 'None'}")
        
        # Print first caption
        if captions:
            print(f"  First caption: {captions[0]}")
        
        # Print tokenized caption info
        if caption_tokens is not None:
            decoded = tokenizer.decode(caption_tokens[0], skip_special_tokens=True)
            print(f"  Decoded tokens: {decoded}")
        
        # Only show first 3 batches
        if batch_idx >= 2:
            break
    
    print("\nDataset demonstration completed!")


if __name__ == "__main__":
    main() 