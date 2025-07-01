#!/usr/bin/env python3
"""
Debug inference script to diagnose caption generation issues.
"""

import torch
from PIL import Image
import argparse
import os
from src.train import ImageCaptioningModel, select_device

def debug_generate_caption(model, image, max_length=50, num_beams=5):
    """Debug version of generate_caption with detailed output."""
    model.eval()
    
    # Encode image
    print("Encoding image...")
    image_embeds = model.encode_images(image)
    print(f"Image embeddings shape: {image_embeds.shape}")
    
    # Try different generation strategies
    strategies = [
        {
            "name": "BOS token only",
            "input_ids": torch.tensor([[model.tokenizer.bos_token_id]], dtype=torch.long),
            "config": {
                "max_length": max_length,
                "num_beams": num_beams,
                "early_stopping": False,  # Disable early stopping
                "do_sample": True,
                "temperature": 1.0,  # Higher temperature
                "top_p": 0.9,
                "no_repeat_ngram_size": 0
            }
        },
        {
            "name": "With prompt 'A photo of'",
            "input_ids": model.tokenizer("A photo of", return_tensors="pt").input_ids,
            "config": {
                "max_length": max_length,
                "num_beams": num_beams,
                "early_stopping": False,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
                "no_repeat_ngram_size": 0
            }
        },
        {
            "name": "Greedy decoding",
            "input_ids": torch.tensor([[model.tokenizer.bos_token_id]], dtype=torch.long),
            "config": {
                "max_length": max_length,
                "num_beams": 1,  # Greedy
                "early_stopping": False,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0
            }
        }
    ]
    
    device = next(model.parameters()).device
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Trying strategy: {strategy['name']}")
        print(f"{'='*50}")
        
        # Setup generation config
        generation_config = model.decoder.generation_config
        for key, value in strategy['config'].items():
            setattr(generation_config, key, value)
        
        print(f"Generation config: {generation_config}")
        
        # Prepare input
        input_ids = strategy['input_ids'].to(device)
        attention_mask = torch.ones_like(input_ids)
        
        print(f"Starting with input_ids: {input_ids}")
        print(f"Input text: '{model.tokenizer.decode(input_ids[0], skip_special_tokens=True)}'")
        
        with torch.no_grad():
            generated_ids = model.decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                generation_config=generation_config
            )
        
        print(f"Generated token IDs: {generated_ids[0]}")
        print(f"Generated token IDs shape: {generated_ids.shape}")
        
        # Decode step by step
        print("\nDecoding tokens:")
        for i, token_id in enumerate(generated_ids[0]):
            token = model.tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"Token {i}: {token_id} -> '{token}'")
        
        # Decode full caption
        caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\nFinal caption: '{caption}'")
        
        if caption.strip():  # If we got a non-empty caption, return it
            return caption
    
    return ""

def main():
    parser = argparse.ArgumentParser(description="Debug image captioning inference")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--num-beams", type=int, default=5, help="Number of beams for beam search")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Image not found: {args.image_path}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    device = args.device if args.device else select_device()
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = ImageCaptioningModel()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Generate caption with debug info
    caption = debug_generate_caption(model, image, args.max_length, args.num_beams)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULT: '{caption}'")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 