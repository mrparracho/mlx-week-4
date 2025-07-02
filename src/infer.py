import torch
from PIL import Image
import argparse
import os
from src.train import ImageCaptioningModel, select_device

def load_model(checkpoint_path, device):
    model = ImageCaptioningModel()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for text generation (0.1-2.0)")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum caption length")
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
    model = load_model(args.checkpoint, device)

    # Load and preprocess image
    image = Image.open(args.image_path).convert('RGB')
    # The model's generate_caption method will handle preprocessing

    print(f"Generation parameters:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max length: {args.max_length}")
    print(f"  Num beams: {args.num_beams}")
    print()

    caption = model.generate_caption(
        image, 
        max_length=args.max_length, 
        num_beams=args.num_beams, 
        temperature=args.temperature
    )
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main() 