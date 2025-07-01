import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import random


class FlickrDataset(Dataset):
    """
    Flickr dataset for image captioning training.
    
    Expected directory structure:
    flickr_dataset/
    ├── images/
    │   ├── 123456.jpg
    │   ├── 123457.jpg
    │   └── ...
    └── captions.json
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        max_length: int = 77,
        image_size: int = 224,
        transform=None,
        tokenizer=None
    ):
        """
        Initialize the Flickr dataset.
        
        Args:
            root_dir: Root directory containing images and captions
            split: Dataset split ('train', 'val', 'test')
            max_length: Maximum caption length
            image_size: Size to resize images to
            transform: Optional image transformations
            tokenizer: Tokenizer for caption processing
        """
        self.root_dir = root_dir
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.images_dir = os.path.join(root_dir, "images")
        self.captions_file = os.path.join(root_dir, "captions.json")
        
        # Load captions
        self.captions_data = self._load_captions()
        
        # Filter by split if split information is available
        if split in ["train", "val", "test"]:
            self.captions_data = [
                item for item in self.captions_data 
                if item.get("split", "train") == split
            ]
        
        print(f"Loaded {len(self.captions_data)} samples for {split} split")
    
    def _load_captions(self) -> List[Dict]:
        """Load captions from JSON file."""
        if not os.path.exists(self.captions_file):
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")
        
        with open(self.captions_file, 'r') as f:
            captions_data = json.load(f)
        
        return captions_data
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        full_path = os.path.join(self.images_dir, image_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def _process_caption(self, caption: str) -> torch.Tensor:
        """Process caption with tokenizer."""
        if self.tokenizer is None:
            return torch.tensor([])  # Return empty tensor if no tokenizer
        
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return tokens.input_ids.squeeze(0)
    
    def __len__(self) -> int:
        return len(self.captions_data)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, torch.Tensor]:
        """Get a single sample."""
        item = self.captions_data[idx]
        
        # Load image
        image_path = item['image']
        image = self._load_image(image_path)
        
        # Get caption
        caption = item['caption']
        
        # Process caption if tokenizer is available
        caption_tokens = self._process_caption(caption)
        
        return image, caption, caption_tokens
    
    def get_caption_vocab(self) -> List[str]:
        """Get all unique captions for vocabulary analysis."""
        return [item['caption'] for item in self.captions_data]


class FlickrDataLoader:
    """
    Data loader for Flickr dataset with batching and preprocessing.
    """
    
    def __init__(
        self,
        dataset: FlickrDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        collate_fn=None
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset: FlickrDataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            collate_fn: Custom collate function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    def _default_collate_fn(self, batch):
        """Default collate function for batching."""
        images, captions, caption_tokens = zip(*batch)
        
        # Stack images if they're tensors
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        
        # Stack caption tokens
        if len(caption_tokens[0]) > 0:
            caption_tokens = torch.stack(caption_tokens)
        else:
            caption_tokens = None
        
        return images, captions, caption_tokens
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_sample_flickr_data(root_dir: str, num_samples: int = 100):
    """
    Create sample Flickr dataset structure for testing.
    
    Args:
        root_dir: Directory to create the sample data in
        num_samples: Number of sample images/captions to create
    """
    os.makedirs(root_dir, exist_ok=True)
    images_dir = os.path.join(root_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Sample captions
    sample_captions = [
        "A person walking on the beach",
        "A cat sitting on a windowsill",
        "A car parked on the street",
        "A dog playing in the park",
        "A building with modern architecture",
        "A flower blooming in the garden",
        "A child playing with toys",
        "A sunset over the mountains",
        "A bird flying in the sky",
        "A boat sailing on the water"
    ]
    
    # Create sample data
    captions_data = []
    for i in range(num_samples):
        image_name = f"sample_{i:06d}.jpg"
        caption = random.choice(sample_captions)
        split = random.choice(["train", "train", "train", "val", "test"])  # 75% train, 15% val, 10% test
        
        captions_data.append({
            "image": image_name,
            "caption": caption,
            "split": split
        })
    
    # Save captions
    captions_file = os.path.join(root_dir, "captions.json")
    with open(captions_file, 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    print(f"Created sample Flickr dataset at {root_dir}")
    print(f"Captions saved to {captions_file}")
    print(f"Please add {num_samples} images to {images_dir}")
    
    return captions_data 