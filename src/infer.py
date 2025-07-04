import torch
import torch.nn as nn
from PIL import Image
import argparse
import os
from src.train import ImageCaptioningModel, select_device
from src.train_self import ImageCaptioningModel as ImageCaptioningModelSelf
# Qwen imports
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM


def load_model(checkpoint_path, device):
    model = ImageCaptioningModel()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_model_self(checkpoint_path, device):
    model = ImageCaptioningModelSelf(decoder_mode='self_attention')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

# Qwen inference loader
class QwenWithImagePrefix(nn.Module):
    def __init__(self, qwen_model, image_embed_dim, qwen_embed_dim):
        super().__init__()
        self.qwen = qwen_model
        self.img_proj = nn.Linear(image_embed_dim, qwen_embed_dim)

    def forward(self, image_embeds, input_ids, attention_mask):
        prefix_len = image_embeds.size(1)  # dynamically get the number of image tokens

        img_tokens = self.img_proj(image_embeds)  # [B, prefix_len, 1024]
        tok_embeddings = self.qwen.model.embed_tokens(input_ids)  # [B, T, 1024]
        full_embeddings = torch.cat([img_tokens, tok_embeddings], dim=1)  # [B, prefix + T, 1024]

        prefix_mask = torch.ones((input_ids.size(0), prefix_len), device=input_ids.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, prefix_len + T]

        outputs = self.qwen(inputs_embeds=full_embeddings, attention_mask=full_mask)

        return outputs.logits[:, prefix_len:, :]  # [B, T, V] → skip image token outputs
    
def load_model_qwen(model_path, device, qwen_dir="Qwen/Qwen3-0.6B-Base", clip_model="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_model = CLIPVisionModel.from_pretrained(clip_model, use_safetensors=True).to(device)
    image_model.eval()
    qwen = AutoModelForCausalLM.from_pretrained(qwen_dir).to(device).eval()
    model = QwenWithImagePrefix(qwen_model=qwen, image_embed_dim=768, qwen_embed_dim=1024).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    def generate_caption(image, max_length=91, num_beams=1, temperature=1.0):
        prompt_tokens = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        input_ids = prompt_tokens.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        img_embeddings = image_model(pixel_values=image_tensor.unsqueeze(0).to(device)).last_hidden_state
        generated_ids = []
        for _ in range(max_length):
            logits = model(img_embeddings, input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id.unsqueeze(0))], dim=1)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generate_caption

# CLI for Qwen inference
def main_qwen():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a Qwen-based model.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--model-path", type=str, default="checkpoints/model_2.pth", help="Path to Qwen model checkpoint")
    parser.add_argument("--qwen-dir", type=str, default="Qwen/Qwen3-0.6B-Base", help="HuggingFace repo for Qwen model")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name or path")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--max-length", type=int, default=91, help="Maximum caption length")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Image not found: {args.image_path}")
        return
    if not os.path.exists(args.model_path):
        print(f"Qwen model checkpoint not found: {args.model_path}")
        return

    device = args.device if args.device else select_device()
    print(f"Using device: {device}")
    generate_caption = load_model_qwen(args.model_path, device, qwen_dir=args.qwen_dir, clip_model=args.clip_model)
    image = Image.open(args.image_path).convert("RGB")
    print(f"Generation parameters:")
    print(f"  Max length: {args.max_length}")
    print()
    caption = generate_caption(image, max_length=args.max_length)
    print(f"Generated caption: {caption}")

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
    image = Image.open(args.image_path).convert('RGB')
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