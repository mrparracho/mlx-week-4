#!/usr/bin/env python3

import torch
import argparse
import os
import random
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, AutoTokenizer, CLIPVisionModel, AutoModelForCausalLM
from PIL import Image

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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

@torch.no_grad()
def generate_caption(model, image_tensor, tokenizer, max_len, prompt_tokens, image_model):
    model.eval()
    input_ids = prompt_tokens.unsqueeze(0).to(DEVICE)
    attention_mask = torch.ones_like(input_ids)

    img_embeddings = image_model(pixel_values=image_tensor.unsqueeze(0).to(DEVICE)).last_hidden_state

    generated_ids = []

    for _ in range(max_len):
        logits = model(img_embeddings, input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id.unsqueeze(0))], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def test_model_on_random_samples(model, dataset, tokenizer, processor, image_model,
                                 num_samples=3, save_plot=True, plot_path="test_results.png",
                                 max_length=50, output_dir="generated_images"):
    dataset_size = len(dataset)
    print(f"📁 Dataset size: {dataset_size}")

    if dataset_size == 0:
        print("❌ No samples found in the test dataset.")
        return

    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(num_samples, dataset_size)
    random_indices = random.sample(range(dataset_size), num_samples)
    selected_samples = [dataset[i] for i in random_indices]

    prompt_tokens = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for i, sample in enumerate(selected_samples):
        image = sample["image"].convert("RGB")
        reference_caption = random.choice(sample["caption"])
        image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        generated_caption = generate_caption(model, image_tensor, tokenizer, max_length, prompt_tokens, image_model)

        print(f"\n📸 Sample {i + 1}")
        print(f"   Generated: {generated_caption}")
        print(f"   Reference: {reference_caption}")

        # Create individual figure for each image
        fig_ind, ax_ind = plt.subplots(figsize=(5, 5))
        ax_ind.imshow(image)
        ax_ind.set_title(f"Sample {i + 1}", fontsize=12, fontweight='bold')
        ax_ind.text(0.5, -0.15,
                    f"Generated: {generated_caption}\nReference: {reference_caption}",
                    transform=ax_ind.transAxes,
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_ind.axis('off')

        # Save individual annotated image
        image_path = os.path.join(output_dir, f"sample_{i+1}.png")
        fig_ind.tight_layout()
        fig_ind.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)

        # Also fill the shared subplot
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f"Sample {i+1}", fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15,
                f"Generated: {generated_caption}\nReference: {reference_caption}",
                transform=ax.transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.axis('off')

    if save_plot:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📁 Combined plot saved to: {plot_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-CLIP image captioning model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to local Flickr30k dataset split (e.g. ./flickr_dataset/test)")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to test")
    parser.add_argument("--max-length", type=int, default=50, help="Max caption generation length")
    parser.add_argument("--no-plot", action="store_true", help="Disable saving the output plot")
    parser.add_argument("--plot-path", type=str, default="test_results.png", help="Path to save the output plot")
    args = parser.parse_args()

    # Load tokenizer and vision processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("./qwen3", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    qwen = AutoModelForCausalLM.from_pretrained("./qwen3").to(DEVICE).eval()
    model = QwenWithImagePrefix(qwen_model=qwen, image_embed_dim=768, qwen_embed_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully!")

    # Load test dataset from local disk
    print(f"📁 Loading dataset from: {args.data_dir}")
    test_data = load_from_disk(args.data_dir)

    # Evaluate on random samples
    test_model_on_random_samples(
        model=model,
        dataset=test_data,
        tokenizer=tokenizer,
        processor=processor,
        image_model=image_model,
        num_samples=args.num_samples,
        save_plot=not args.no_plot,
        plot_path=args.plot_path,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
    # to run:
    # python test_model_qwen_random.py --model-path qwen_runs/model_20250703_153230/model_6.pth --data-dir ./flickr_dataset/test --num-samples 4 --plot-path test_results.png