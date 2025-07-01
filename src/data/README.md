# Flickr Dataset for Image Captioning

This module provides a PyTorch dataset and dataloader for the Flickr dataset, designed for training image captioning models.

## Dataset Structure

The expected directory structure is:

```
flickr_dataset/
├── images/
│   ├── 123456.jpg
│   ├── 123457.jpg
│   └── ...
└── captions.json
```

## Captions JSON Format

The `captions.json` file should contain a list of dictionaries with the following structure:

```json
[
  {
    "image": "123456.jpg",
    "caption": "A person walking on the beach",
    "split": "train"
  },
  {
    "image": "123457.jpg", 
    "caption": "A cat sitting on a windowsill",
    "split": "val"
  }
]
```

## Usage

### Basic Usage

```python
from src.data.flickr_dataset import FlickrDataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create dataset
dataset = FlickrDataset(
    root_dir="data/flickr_dataset",
    split="train",
    tokenizer=tokenizer,
    max_length=77
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for images, captions, caption_tokens in dataloader:
    # images: List of PIL Images or torch.Tensor if transform is applied
    # captions: List of caption strings
    # caption_tokens: torch.Tensor of tokenized captions
    pass
```

### Creating Sample Data

For testing purposes, you can create sample data:

```python
from src.data.flickr_dataset import create_sample_flickr_data

# Create sample dataset with 100 samples
create_sample_flickr_data("data/flickr_dataset", num_samples=100)
```

## Features

- **Flexible splits**: Support for train/val/test splits
- **Tokenization**: Optional integration with HuggingFace tokenizers
- **Image preprocessing**: Support for custom image transformations
- **Batching**: Efficient batching with custom collate functions
- **Error handling**: Robust error handling for missing files

## Integration with Training

The dataset is designed to work seamlessly with the image captioning training pipeline in `src/train.py`. It provides:

- Image encoding with CLIP vision encoder
- Caption generation with GPT-2 decoder
- Cross-attention between image features and text tokens
- End-to-end training with validation 