import torch
from PIL import Image
import argparse
import os
from src.train import ImageCaptioningModel, select_device

def load_model(checkpoint_path, device):
    model = ImageCaptioningModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
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

    caption = model.generate_caption(image)
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main() 