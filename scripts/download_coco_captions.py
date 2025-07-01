#!/usr/bin/env python3
"""
Download and prepare a subset of COCO captions dataset for image captioning training.
This is an alternative to Flickr8k that's easier to access.
"""

import os
import requests
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import random


def download_file(url: str, filename: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_coco_subset(output_dir: str = "data/coco_captions", max_images: int = 1000):
    """Download a subset of COCO captions dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # COCO API URLs (these are publicly available)
    coco_urls = {
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'images': 'http://images.cocodataset.org/zips/val2017.zip'  # Using validation set (smaller)
    }
    
    print("📥 Downloading COCO dataset...")
    print("This will download ~1GB of data. Using validation set for smaller size.")
    
    # Download annotations
    annotations_file = os.path.join(output_dir, "annotations.zip")
    if not os.path.exists(annotations_file):
        print("Downloading annotations...")
        download_file(coco_urls['annotations'], annotations_file)
    
    # Download images
    images_file = os.path.join(output_dir, "images.zip")
    if not os.path.exists(images_file):
        print("Downloading images...")
        download_file(coco_urls['images'], images_file)
    
    # Extract files
    print("📦 Extracting files...")
    with zipfile.ZipFile(annotations_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    with zipfile.ZipFile(images_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Parse annotations
    annotations_path = os.path.join(output_dir, "annotations", "captions_val2017.json")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Process captions
    print("📝 Processing captions...")
    captions_data = []
    
    # Group by image
    image_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(ann['caption'])
    
    # Get image info
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Create dataset entries
    for image_id, captions in image_captions.items():
        if image_id in image_info:
            img_info = image_info[image_id]
            filename = img_info['file_name']
            
            # Select one caption per image for simplicity
            caption = random.choice(captions)
            
            captions_data.append({
                'image': filename,
                'caption': caption,
                'image_id': image_id
            })
    
    # Limit to max_images
    if len(captions_data) > max_images:
        captions_data = random.sample(captions_data, max_images)
    
    # Create train/val/test splits
    random.shuffle(captions_data)
    train_end = int(len(captions_data) * 0.8)
    val_end = int(len(captions_data) * 0.9)
    
    train_data = [{'split': 'train', **item} for item in captions_data[:train_end]]
    val_data = [{'split': 'val', **item} for item in captions_data[train_end:val_end]]
    test_data = [{'split': 'test', **item} for item in captions_data[val_end:]]
    
    all_data = train_data + val_data + test_data
    
    # Copy images to our images directory
    print("🖼️  Copying images...")
    source_images_dir = os.path.join(output_dir, "val2017")
    import shutil
    
    for item in all_data:
        src = os.path.join(source_images_dir, item['image'])
        dst = os.path.join(images_dir, item['image'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # Save captions
    captions_file = os.path.join(output_dir, "captions.json")
    with open(captions_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Clean up downloaded files
    print("🧹 Cleaning up...")
    os.remove(annotations_file)
    os.remove(images_file)
    import shutil
    shutil.rmtree(os.path.join(output_dir, "annotations"))
    shutil.rmtree(source_images_dir)
    
    print(f"✅ COCO subset prepared successfully!")
    print(f"📊 Statistics:")
    print(f"   - Total samples: {len(all_data)}")
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    print(f"   - Test samples: {len(test_data)}")
    print(f"   - Images: {len(os.listdir(images_dir))}")
    print(f"📁 Output directory: {output_dir}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download COCO captions subset")
    parser.add_argument("--output-dir", type=str, default="data/coco_captions", help="Output directory")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum number of images to download")
    
    args = parser.parse_args()
    
    success = download_coco_subset(args.output_dir, args.max_images)
    if success:
        print("🎉 Dataset is ready to use!")
        print("You can now run: uv run python src/train.py")
    else:
        print("❌ Failed to prepare dataset") 