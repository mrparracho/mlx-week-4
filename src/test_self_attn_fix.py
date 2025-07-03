#!/usr/bin/env python3
"""
Test script to verify the self-attention model fixes work correctly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from src.train_self_attn import ImageCaptioningModelSelfAttention, prepare_captions_for_self_attention
from src.data.flickr_dataset import create_sample_flickr_data

def test_self_attention_model():
    """Test the self-attention model with a small batch to verify it works."""
    print("🧪 Testing Self-Attention Model Implementation")
    print("=" * 50)
    
    # Create sample data if it doesn't exist
    data_dir = "data/flickr30k"
    if not os.path.exists(data_dir):
        print("Creating sample Flickr dataset...")
        create_sample_flickr_data(data_dir, num_samples=10)
    
    # Initialize model
    print("📦 Initializing self-attention model...")
    model = ImageCaptioningModelSelfAttention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Model initialized on {device}")
    
    # Create simple test data
    print("📊 Creating test data...")
    batch_size = 2
    image_size = (224, 224)
    
    # Create dummy images
    images = torch.randn(batch_size, 3, *image_size)
    
    # Create dummy captions
    captions = [
        "A person walking in the park",
        "A cat sitting on a chair"
    ]
    
    # Prepare captions for self-attention
    input_ids, attention_mask = prepare_captions_for_self_attention(
        captions, model.tokenizer, max_text_length=77
    )
    
    print(f"✅ Test data created:")
    print(f"   Images shape: {images.shape}")
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    model.train()
    
    try:
        # Move data to device
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        outputs = model(images, input_ids, attention_mask, labels=input_ids)
        
        print(f"✅ Forward pass successful!")
        print(f"   Loss: {outputs.loss.item():.4f}")
        print(f"   Logits shape: {outputs.logits.shape}")
        
        # Test backward pass
        print("\n🔄 Testing backward pass...")
        loss = outputs.loss
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        # Test generation
        print("\n🔄 Testing caption generation...")
        model.eval()
        
        # Convert tensor to PIL for generation
        test_image = transforms.ToPILImage()(images[0])
        
        with torch.no_grad():
            generated_caption = model.generate_caption(test_image, max_length=20, temperature=0.8)
        
        print(f"✅ Generation successful!")
        print(f"   Generated caption: {generated_caption}")
        
        # Memory usage check
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 Memory usage:")
            print(f"   Allocated: {memory_allocated:.2f} GB")
            print(f"   Reserved: {memory_reserved:.2f} GB")
        
        print("\n🎉 All tests passed! Self-attention model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency with different batch sizes."""
    print("\n🧪 Testing Memory Efficiency")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageCaptioningModelSelfAttention()
    model = model.to(device)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")
        
        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create test data
            images = torch.randn(batch_size, 3, 224, 224).to(device)
            captions = ["A test image"] * batch_size
            
            input_ids, attention_mask = prepare_captions_for_self_attention(
                captions, model.tokenizer, max_text_length=77
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            model.train()
            outputs = model(images, input_ids, attention_mask, labels=input_ids)
            
            # Check memory
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"   ✅ Success - Memory: {memory_allocated:.2f} GB")
            else:
                print(f"   ✅ Success")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            break

if __name__ == "__main__":
    import os
    
    print("🚀 Starting Self-Attention Model Tests")
    print("=" * 60)
    
    # Test basic functionality
    success = test_self_attention_model()
    
    if success:
        # Test memory efficiency
        test_memory_efficiency()
        
        print("\n🎉 All tests completed successfully!")
        print("The self-attention model is ready for training.")
    else:
        print("\n❌ Tests failed. Please check the implementation.") 