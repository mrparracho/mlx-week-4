#!/usr/bin/env python3
"""
Download and prepare the Flickr8k dataset for image captioning training.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import json
import re
from tqdm import tqdm


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


def extract_archive(archive_path: str, extract_to: str):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path} to {extract_to}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def parse_flickr8k_captions(captions_file: str):
    """Parse Flickr8k captions file and return structured data."""
    captions_data = []
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Flickr8k format: image_name#caption_number caption_text
                match = re.match(r'(\d+\.jpg)#(\d+)\s+(.+)', line)
                if match:
                    image_name, caption_num, caption_text = match.groups()
                    captions_data.append({
                        'image': image_name,
                        'caption': caption_text,
                        'caption_id': int(caption_num)
                    })
    
    return captions_data


def create_train_val_test_splits(captions_data, train_ratio=0.8, val_ratio=0.1):
    """Create train/val/test splits from captions data."""
    # Group by image
    image_groups = {}
    for item in captions_data:
        image_name = item['image']
        if image_name not in image_groups:
            image_groups[image_name] = []
        image_groups[image_name].append(item)
    
    # Get unique images
    unique_images = list(image_groups.keys())
    total_images = len(unique_images)
    
    # Calculate split indices
    train_end = int(total_images * train_ratio)
    val_end = int(total_images * (train_ratio + val_ratio))
    
    # Split images
    train_images = unique_images[:train_end]
    val_images = unique_images[train_end:val_end]
    test_images = unique_images[val_end:]
    
    # Create splits
    train_data = []
    val_data = []
    test_data = []
    
    for image_name in train_images:
        for item in image_groups[image_name]:
            train_data.append({**item, 'split': 'train'})
    
    for image_name in val_images:
        for item in image_groups[image_name]:
            val_data.append({**item, 'split': 'val'})
    
    for image_name in test_images:
        for item in image_groups[image_name]:
            test_data.append({**item, 'split': 'test'})
    
    return train_data, val_data, test_data


def download_flickr8k_dataset(output_dir: str = "data/flickr8k"):
    """Download and prepare the Flickr8k dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Flickr8k dataset URLs (you'll need to get these from the official source)
    # Note: These are example URLs - you'll need to replace with actual URLs
    dataset_urls = {
        'images': 'https://example.com/flickr8k_images.zip',  # Replace with actual URL
        'captions': 'https://example.com/flickr8k_captions.txt'  # Replace with actual URL
    }
    
    print("⚠️  Note: You need to manually download the Flickr8k dataset")
    print("The dataset is not freely available for automated download.")
    print("\nTo get the dataset:")
    print("1. Visit: https://forms.illinois.edu/sec/1713398")
    print("2. Fill out the form to request access")
    print("3. Download the files manually")
    print("\nExpected files:")
    print("- Flickr8k_Dataset.zip (contains images)")
    print("- Flickr8k_text/Flickr8k.token.txt (contains captions)")
    
    # Alternative: Create a script that works with manually downloaded files
    print("\nOnce you have the files, place them in the data/ directory and run:")
    print("python scripts/prepare_flickr8k.py")
    
    return False


def prepare_manual_flickr8k(dataset_dir: str, output_dir: str = "data/flickr8k"):
    """Prepare Flickr8k dataset from manually downloaded files."""
    
    # Expected file structure after manual download
    images_zip = os.path.join(dataset_dir, "Flickr8k_Dataset.zip")
    captions_file = os.path.join(dataset_dir, "Flickr8k_text", "Flickr8k.token.txt")
    
    if not os.path.exists(images_zip):
        print(f"❌ Images file not found: {images_zip}")
        return False
    
    if not os.path.exists(captions_file):
        print(f"❌ Captions file not found: {captions_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print("📦 Extracting images...")
    extract_archive(images_zip, output_dir)
    
    # Move images to the correct location
    extracted_images_dir = os.path.join(output_dir, "Flicker8k_Dataset")
    if os.path.exists(extracted_images_dir):
        import shutil
        for file in os.listdir(extracted_images_dir):
            if file.endswith('.jpg'):
                src = os.path.join(extracted_images_dir, file)
                dst = os.path.join(images_dir, file)
                shutil.move(src, dst)
        shutil.rmtree(extracted_images_dir)
    
    print("📝 Parsing captions...")
    captions_data = parse_flickr8k_captions(captions_file)
    
    print("✂️  Creating train/val/test splits...")
    train_data, val_data, test_data = create_train_val_test_splits(captions_data)
    
    # Combine all data
    all_data = train_data + val_data + test_data
    
    # Save captions
    captions_output_file = os.path.join(output_dir, "captions.json")
    with open(captions_output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Dataset prepared successfully!")
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
    
    parser = argparse.ArgumentParser(description="Download and prepare Flickr8k dataset")
    parser.add_argument("--dataset-dir", type=str, help="Directory containing manually downloaded Flickr8k files")
    parser.add_argument("--output-dir", type=str, default="data/flickr8k", help="Output directory for processed dataset")
    
    args = parser.parse_args()
    
    if args.dataset_dir:
        # Prepare from manually downloaded files
        success = prepare_manual_flickr8k(args.dataset_dir, args.output_dir)
        if success:
            print("🎉 Dataset is ready to use!")
        else:
            print("❌ Failed to prepare dataset")
    else:
        # Try to download (will show instructions)
        download_flickr8k_dataset(args.output_dir) 