#!/usr/bin/env python3
"""
Download and prepare the Flickr30k dataset from Hugging Face for image captioning training.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import random
from typing import Optional


def download_flickr30k_from_hf(output_dir: str = "data/flickr30k", max_images: Optional[int] = None):
    """
    Download Flickr30k dataset from Hugging Face and prepare it for training.
    
    Args:
        output_dir: Directory to save the processed dataset
        max_images: Maximum number of images to download (None for all)
    """
    
    print("📥 Loading Flickr30k dataset from Hugging Face...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("nlphuji/flickr30k")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Process each split
    all_data = []
    
    for split_name, split_data in dataset.items():
        print(f"\n📝 Processing {split_name} split ({len(split_data)} samples)...")
        
        # Limit samples if specified
        if max_images and len(split_data) > max_images:
            split_data = split_data.select(range(max_images))
            print(f"Limited to {max_images} samples")
        
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Get image and captions
            image = sample['image']
            captions = sample['caption']
            filename = sample['filename']
            
            # Download image if it doesn't exist
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                try:
                    image.save(image_path)
                except Exception as e:
                    print(f"Failed to save image {filename}: {e}")
                    continue
            
            # Create entries for each caption
            for caption_idx, caption in enumerate(captions):
                data_entry = {
                    'image': filename,
                    'caption': caption,
                    'split': split_name,
                    'caption_id': caption_idx,
                    'sample_id': idx
                }
                all_data.append(data_entry)
    
    # Create train/val splits from available data
    print(f"\n✂️  Creating train/val splits...")
    
    # Group by image to ensure we don't split individual images across splits
    image_groups = {}
    for item in all_data:
        image_name = item['image']
        if image_name not in image_groups:
            image_groups[image_name] = []
        image_groups[image_name].append(item)
    
    # Get unique images and shuffle them
    unique_images = list(image_groups.keys())
    random.shuffle(unique_images)
    
    # Split images: 80% train, 20% val
    train_end = int(len(unique_images) * 0.8)
    train_images = unique_images[:train_end]
    val_images = unique_images[train_end:]
    
    # Create new data with proper splits
    new_all_data = []
    
    for image_name in train_images:
        for item in image_groups[image_name]:
            new_item = item.copy()
            new_item['split'] = 'train'
            new_all_data.append(new_item)
    
    for image_name in val_images:
        for item in image_groups[image_name]:
            new_item = item.copy()
            new_item['split'] = 'val'
            new_all_data.append(new_item)
    
    all_data = new_all_data
    
    # Save captions
    captions_file = os.path.join(output_dir, "captions.json")
    with open(captions_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Print statistics
    print(f"\n✅ Flickr30k dataset prepared successfully!")
    print(f"📊 Statistics:")
    
    # Count by split
    split_counts = {}
    for item in all_data:
        split = item['split']
        split_counts[split] = split_counts.get(split, 0) + 1
    
    for split, count in split_counts.items():
        print(f"   - {split}: {count} samples")
    
    print(f"   - Total samples: {len(all_data)}")
    print(f"   - Images: {len(os.listdir(images_dir))}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Captions file: {captions_file}")
    
    return all_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Flickr30k dataset from Hugging Face")
    parser.add_argument("--output-dir", type=str, default="data/flickr30k", help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to download")
    
    args = parser.parse_args()
    
    # Download dataset
    success = download_flickr30k_from_hf(args.output_dir, args.max_images)
    
    if success:
        print("\n🎉 Dataset is ready to use!")
        print("You can now run: uv run python src/train.py")
        print("\nNote: Update src/train.py to use 'data/flickr30k' as the data_dir")
    else:
        print("❌ Failed to prepare dataset") 