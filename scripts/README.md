# Dataset Download Scripts

This directory contains scripts to download and prepare datasets for image captioning training.

## Available Datasets

### 1. Flickr30k (Recommended - Easy to get from Hugging Face)
**File**: `download_flickr30k.py`

**Pros**:
- ✅ Publicly available via Hugging Face
- ✅ No registration required
- ✅ Automatically downloads and processes
- ✅ High-quality captions
- ✅ Standard benchmark dataset

**Usage**:
```bash
# Download all images (default)
uv run python scripts/download_flickr30k.py

# Download limited number of images
uv run python scripts/download_flickr30k.py --max-images 1000

# Custom output directory
uv run python scripts/download_flickr30k.py --output-dir data/my_flickr30k_data
```

### 2. COCO Captions (Alternative - Easy to get)
**File**: `download_coco_captions.py`

**Pros**:
- ✅ Publicly available
- ✅ No registration required
- ✅ Automatically downloads and processes
- ✅ ~1GB download size

**Usage**:
```bash
# Download 1000 images (default)
uv run python scripts/download_coco_captions.py

# Download custom number of images
uv run python scripts/download_coco_captions.py --max-images 500

# Custom output directory
uv run python scripts/download_coco_captions.py --output-dir data/my_coco_data
```

### 3. Flickr8k (Requires manual download)
**File**: `download_flickr8k.py`

**Pros**:
- ✅ High-quality captions
- ✅ Standard benchmark dataset
- ✅ Smaller size (~1GB)

**Cons**:
- ❌ Requires manual registration
- ❌ Manual download required

**Usage**:
1. Visit: https://forms.illinois.edu/sec/1713398
2. Fill out the form to request access
3. Download the files manually
4. Run the preparation script:
```bash
uv run python scripts/download_flickr8k.py --dataset-dir /path/to/downloaded/files
```

## After Downloading

Once you have a dataset, update the training script to use it:

```python
# In src/train.py, change this line:
data_dir = "data/coco_captions"  # or "data/flickr8k"
```

Then run training:
```bash
uv run python src/train.py
```

## Dataset Structure

Both scripts create the same structure:
```
data/
├── coco_captions/          # or flickr8k/
│   ├── images/
│   │   ├── 000000123456.jpg
│   │   ├── 000000123457.jpg
│   │   └── ...
│   └── captions.json
```

The `captions.json` file contains:
```json
[
  {
    "image": "000000123456.jpg",
    "caption": "A person walking on the beach",
    "split": "train"
  }
]
```

## Recommendation

For getting started quickly, use the **Flickr30k** dataset (best quality and easiest to get):
```bash
uv run python scripts/download_flickr30k.py
```

Or use **COCO captions** as an alternative:
```bash
uv run python scripts/download_coco_captions.py
``` 