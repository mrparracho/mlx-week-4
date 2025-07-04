import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from image_caption_qwen_clip import QwenWithImagePrefix  # <- your wrapper class
import sys
import os

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def greedy_decode(model, image_tensor, tokenizer, max_len, prompt_tokens):
    model.eval()
    input_ids = prompt_tokens.unsqueeze(0).to(DEVICE)  # [1, T_prompt]
    attention_mask = torch.ones_like(input_ids)

    # Encode image
    img_embeddings = image_model(pixel_values=image_tensor.unsqueeze(0).to(DEVICE)).last_hidden_state

    generated_ids = []

    for _ in range(max_len):
        logits = model(img_embeddings, input_ids=input_ids, attention_mask=attention_mask)  # [1, T, vocab]
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break
        
        generated_ids.append(next_token_id.item())

        # if next_token_id.item() == tokenizer.encode(".")[0]:
        #     break

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id.unsqueeze(0))], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(image_path, model_path, max_len):
    # Load processor and tokenizer
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("./qwen3", use_fast=True) #change to qwen3 from huggingface
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompt
    prompt_tokens = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

    # Load vision encoder (CLIP)
    global image_model
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
    image_model.eval()

    # Load Qwen decoder model
    qwen = AutoModelForCausalLM.from_pretrained("./qwen3").to(DEVICE).eval()

    model = QwenWithImagePrefix(qwen_model=qwen, image_embed_dim=768, qwen_embed_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    # Generate caption
    caption = greedy_decode(model, image_tensor, tokenizer, max_len, prompt_tokens)

    print("*************************************")
    print("Generated caption:", caption)
    print("*************************************")


if __name__ == "__main__":
    MAX_SEQ_LEN = 91

    if len(sys.argv) != 2:
        print("Usage: python qwen_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "qwen_runs/model_20250703_153230/model_2.pth"  # adjust as needed
    main(image_path, model_path=model_path, max_len=MAX_SEQ_LEN)
