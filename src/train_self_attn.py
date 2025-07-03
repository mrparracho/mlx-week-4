import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import (
    CLIPVisionModel, 
    CLIPImageProcessor,
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    GPT2Config
)
from transformers.generation.configuration_utils import GenerationConfig
from PIL import Image
import os
from typing import Optional
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

# For evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    
    # Download required NLTK data only once at startup
    def download_nltk_data():
        """Download NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet corpus...")
            nltk.download('wordnet', quiet=True)
    
    # Download data once
    download_nltk_data()
    EVALUATION_AVAILABLE = True
except ImportError:
    print("Warning: Evaluation metrics not available. Install with: pip install nltk rouge-score")
    EVALUATION_AVAILABLE = False

# For Hugging Face Hub
try:
    from huggingface_hub import login
    from transformers.modeling_utils import PreTrainedModel
    HF_AVAILABLE = True
except ImportError:
    print("Warning: Hugging Face Hub not available. Install with: pip install huggingface_hub")
    HF_AVAILABLE = False

# For Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: Weights & Biases not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

from src.data.flickr_dataset import FlickrDataset, FlickrDataLoader, create_sample_flickr_data


def init_wandb(project_name: str = "image-captioning", run_name: str | None = None, config: dict | None = None):
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name: Name of the W&B project
        run_name: Name for this specific run
        config: Configuration dictionary to log
    
    Returns:
        bool: True if wandb was initialized successfully, False otherwise
    """
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping logging")
        return False
    
    try:
        # Get API key from environment variable
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            print("Using WANDB_API_KEY from environment variable")
            wandb.login(key=api_key)
        else:
            print("No WANDB_API_KEY found in environment variables")
            print("Please set WANDB_API_KEY or run 'wandb login' manually")
            return False
        
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            resume="allow"
        )
        print(f"✅ W&B logging initialized: {wandb.run.url}")
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        return False


def log_to_wandb(metrics: dict, step: int | None = None, prefix: str = ""):
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number for logging
        prefix: Prefix to add to metric names
    """
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    try:
        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        print(f"🔍 Logging to W&B: {metrics}")  # Debug line
        
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    except Exception as e:
        print(f"Warning: Failed to log to W&B: {e}")
        import traceback
        traceback.print_exc()


def finish_wandb():
    """Finish the W&B run."""
    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()


def compute_composite_score(metrics: dict, weights: dict = None) -> float:
    """Compute weighted composite score from multiple metrics."""
    if weights is None:
        weights = {
            'METEOR': 0.4,      # Primary metric
            'BLEU-4': 0.2,      # Secondary
            'ROUGE-L': 0.2,     # Secondary
            'BLEU-1': 0.1,      # Tertiary
            'BLEU-2': 0.1       # Tertiary
        }
    
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            score += metrics[metric] * weight
    
    return score


def is_pareto_improvement(current_metrics: dict, best_metrics: dict) -> bool:
    """Check if current metrics dominate previous best (Pareto improvement)."""
    improved = False
    for metric in ['METEOR', 'BLEU-4', 'ROUGE-L']:
        if metric in current_metrics and metric in best_metrics:
            if current_metrics[metric] < best_metrics[metric]:
                return False  # Not Pareto improvement
            elif current_metrics[metric] > best_metrics[metric]:
                improved = True
    
    return improved


def should_save_model_multi_criteria(current_metrics: dict, best_metrics: dict, 
                                   primary_metric: str = 'METEOR',
                                   tolerance: float = 0.01) -> bool:
    """
    Save if primary metric improves OR if primary metric is close 
    but other metrics improve significantly.
    """
    if primary_metric not in current_metrics or primary_metric not in best_metrics:
        return False
    
    primary_improvement = current_metrics[primary_metric] - best_metrics[primary_metric]
    
    # Direct improvement in primary metric
    if primary_improvement > 0:
        return True
    
    # Small decrease in primary metric but significant improvements elsewhere
    if primary_improvement > -tolerance:
        other_improvements = 0
        for metric in ['BLEU-4', 'ROUGE-L', 'BLEU-1']:
            if metric in current_metrics and metric in best_metrics:
                improvement = current_metrics[metric] - best_metrics[metric]
                if improvement > 0.02:  # Significant improvement threshold
                    other_improvements += 1
        
        return other_improvements >= 2  # At least 2 other metrics improved significantly
    
    return False


def should_save_model_meteor_centric(current_metrics: dict, best_metrics: dict) -> bool:
    """
    Save model based on METEOR-centric evaluation with tolerance for other metrics.
    """
    if 'METEOR' not in current_metrics:
        print(f"⚠️  METEOR not found in current metrics: {list(current_metrics.keys())}")
        return False
    
    # If best_metrics is empty (first epoch), always save
    if not best_metrics:
        print(f"✅ First epoch - saving model with METEOR: {current_metrics['METEOR']:.4f}")
        return True
    
    if 'METEOR' not in best_metrics:
        print(f"⚠️  METEOR not found in best metrics: {list(best_metrics.keys())}")
        return False
    
    meteor_improvement = current_metrics['METEOR'] - best_metrics['METEOR']
    
    print(f"🔍 METEOR-centric check: current={current_metrics['METEOR']:.4f}, best={best_metrics['METEOR']:.4f}, improvement={meteor_improvement:.4f}")
    
    # Direct METEOR improvement
    if meteor_improvement > 0:
        print(f"✅ Direct METEOR improvement: {meteor_improvement:.4f}")
        return True
    
    # Small METEOR decrease but good overall performance
    if meteor_improvement > -0.02:  # 2% tolerance
        # Check if other metrics compensate
        bleu4_improvement = current_metrics.get('BLEU-4', 0) - best_metrics.get('BLEU-4', 0)
        rouge_improvement = current_metrics.get('ROUGE-L', 0) - best_metrics.get('ROUGE-L', 0)
        
        print(f"🔍 Compensation check: BLEU-4 improvement={bleu4_improvement:.4f}, ROUGE-L improvement={rouge_improvement:.4f}")
        
        # Save if other metrics improved significantly
        if bleu4_improvement > 0.01 or rouge_improvement > 0.02:
            print(f"✅ Compensating improvement: BLEU-4={bleu4_improvement:.4f}, ROUGE-L={rouge_improvement:.4f}")
            return True
    
    print(f"❌ No METEOR-centric improvement found")
    return False


def should_save_model_single_metric(current_metrics: dict, best_metrics: dict, metric_name: str) -> bool:
    """
    Save model based on a single metric improvement.
    """
    if metric_name not in current_metrics:
        print(f"⚠️  {metric_name} not found in current metrics: {list(current_metrics.keys())}")
        return False
    
    # If best_metrics is empty (first epoch), always save
    if not best_metrics:
        print(f"✅ First epoch - saving model with {metric_name}: {current_metrics[metric_name]:.4f}")
        return True
    
    if metric_name not in best_metrics:
        print(f"⚠️  {metric_name} not found in best metrics: {list(best_metrics.keys())}")
        return False
    
    improvement = current_metrics[metric_name] - best_metrics[metric_name]
    print(f"🔍 {metric_name} check: current={current_metrics[metric_name]:.4f}, best={best_metrics[metric_name]:.4f}, improvement={improvement:.4f}")
    
    should_save = current_metrics[metric_name] > best_metrics[metric_name]
    if should_save:
        print(f"✅ {metric_name} improvement: {improvement:.4f}")
    else:
        print(f"❌ No {metric_name} improvement")
    
    return should_save


def determine_model_save(eval_metrics: dict, best_metrics: dict, eval_strategy: str) -> tuple[bool, str]:
    """
    Determine if model should be saved based on evaluation strategy.
    
    Args:
        eval_metrics: Current epoch evaluation metrics
        best_metrics: Best metrics so far
        eval_strategy: Evaluation strategy to use
    
    Returns:
        tuple: (should_save, reason)
    """
    if not eval_metrics:
        print("⚠️  No evaluation metrics available")
        return False, "No evaluation metrics available"
    
    print(f"🔍 Model save check - Strategy: {eval_strategy}")
    print(f"   Current metrics: {eval_metrics}")
    print(f"   Best metrics: {best_metrics}")
    
    if eval_strategy == "weighted-composite":
        current_score = compute_composite_score(eval_metrics)
        best_score = compute_composite_score(best_metrics) if best_metrics else 0
        should_save = current_score > best_score
        reason = f"Composite score: {current_score:.4f} > {best_score:.4f}"
        
    elif eval_strategy == "pareto":
        should_save = is_pareto_improvement(eval_metrics, best_metrics)
        reason = "Pareto improvement achieved"
        
    elif eval_strategy == "multi-criteria":
        should_save = should_save_model_multi_criteria(eval_metrics, best_metrics)
        reason = "Multi-criteria improvement achieved"
        
    elif eval_strategy == "meteor-centric":
        print(f"🔍 Calling should_save_model_meteor_centric with eval_strategy: {eval_strategy}")
        should_save = should_save_model_meteor_centric(eval_metrics, best_metrics)
        reason = "METEOR-centric improvement achieved"
        
    elif eval_strategy == "bleu":
        should_save = should_save_model_single_metric(eval_metrics, best_metrics, 'BLEU-4')
        reason = f"BLEU-4: {eval_metrics.get('BLEU-4', 0):.4f} > {best_metrics.get('BLEU-4', 0):.4f}"
        
    elif eval_strategy == "rouge":
        should_save = should_save_model_single_metric(eval_metrics, best_metrics, 'ROUGE-L')
        reason = f"ROUGE-L: {eval_metrics.get('ROUGE-L', 0):.4f} > {best_metrics.get('ROUGE-L', 0):.4f}"
        
    elif eval_strategy == "meteor":
        should_save = should_save_model_single_metric(eval_metrics, best_metrics, 'METEOR')
        reason = f"METEOR: {eval_metrics.get('METEOR', 0):.4f} > {best_metrics.get('METEOR', 0):.4f}"
        
    else:
        # Fallback to validation loss
        return False, "Unknown evaluation strategy, using validation loss"
    
    return should_save, reason


class ImageCaptioningModel(nn.Module):
    """
    Image captioning model using CLIP vision encoder and GPT-2 decoder.
    """
    
    def __init__(self, clip_model_name: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        
        # CLIP Vision Encoder
        self.clip_vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP encoder
        for param in self.clip_vision_encoder.parameters():
            param.requires_grad = False
        
        # GPT-2 Decoder with cross-attention
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set special tokens properly
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        
        config = GPT2Config.from_pretrained('gpt2')
        config.add_cross_attention = True
        config.loss_type = "causal_lm"  # Explicitly set loss type
        
        self.decoder = GPT2LMHeadModel(config)
        
        # Projection layer to match dimensions if needed
        clip_hidden_size = self.clip_vision_encoder.config.hidden_size
        gpt_hidden_size = self.decoder.config.hidden_size
        
        if clip_hidden_size != gpt_hidden_size:
            self.projection = nn.Linear(clip_hidden_size, gpt_hidden_size)
        else:
            self.projection = nn.Identity()
    
    def encode_images(self, images):
        """Encode images using CLIP vision encoder."""
        device = next(self.parameters()).device
        
        # Handle different input types
        if isinstance(images, list):
            # List of PIL images
            pil_images = images
            inputs = self.clip_processor(pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif isinstance(images, torch.Tensor):
            # Tensor input (from training)
            images = images.to(device)
            
            # Convert tensors back to PIL for CLIP processing
            if images.dim() == 4:  # Batch of tensors
                pil_images = []
                for i in range(images.shape[0]):
                    img = transforms.ToPILImage()(images[i])
                    pil_images.append(img)
                
                # Process with CLIP processor
                inputs = self.clip_processor(pil_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                # Single tensor
                img = transforms.ToPILImage()(images)
                inputs = self.clip_processor([img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            # Single PIL image
            inputs = self.clip_processor([images], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_vision_encoder(**inputs)
        
        # Get patch-level embeddings and project to GPT-2 dimension
        image_embeds = outputs.last_hidden_state
        image_embeds = self.projection(image_embeds)
        
        return image_embeds
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """Forward pass for training."""
        # Encode images
        image_embeds = self.encode_images(images)
        
        # Decode with GPT-2
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels
        )
        
        return outputs
    
    def generate_caption(self, image, max_length=50, num_beams=5, temperature=0.8):
        """Generate caption for a single image."""
        self.eval()
        
        # Encode image - pass the PIL image directly
        image_embeds = self.encode_images(image)
        
        # Generate caption with better parameters
        generation_config = GenerationConfig(
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=False,  # Don't stop early
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,  # Now configurable
            top_p=0.9,
            no_repeat_ngram_size=2,  # Avoid repetition
            repetition_penalty=1.1  # Slight penalty for repetition
        )
        
        # Start with a prompt that encourages longer captions
        prompt = "A photo shows"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids)
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            generated_ids = self.decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                generation_config=generation_config
            )
        
        # Decode caption
        caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption


class ImageCaptioningModelSelfAttention(nn.Module):
    """
    Image captioning model using CLIP vision encoder and GPT-2 decoder with self-attention.
    CLIP embeddings are concatenated with caption embeddings and processed through self-attention.
    """
    def __init__(self, clip_model_name: str = 'openai/clip-vit-base-patch32', max_seq_length: int = 256):
        super().__init__()
        self.clip_vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        for param in self.clip_vision_encoder.parameters():
            param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        # Add special tokens for image boundaries
        special_tokens = {"additional_special_tokens": ["<image>", "</image>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        config = GPT2Config.from_pretrained('gpt2')
        config.add_cross_attention = False
        config.n_positions = max_seq_length
        config.loss_type = "causal_lm"
        self.decoder = GPT2LMHeadModel(config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        clip_hidden_size = self.clip_vision_encoder.config.hidden_size
        gpt_hidden_size = self.decoder.config.hidden_size
        if clip_hidden_size != gpt_hidden_size:
            self.projection = nn.Linear(clip_hidden_size, gpt_hidden_size)
        else:
            self.projection = nn.Identity()
        self.max_seq_length = max_seq_length
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids("</image>")
    def encode_images(self, images):
        device = next(self.parameters()).device
        if isinstance(images, list):
            pil_images = images
            inputs = self.clip_processor(pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif isinstance(images, torch.Tensor):
            images = images.to(device)
            if images.dim() == 4:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                inputs = self.clip_processor(pil_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                img = transforms.ToPILImage()(images)
                inputs = self.clip_processor([img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = self.clip_processor([images], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_vision_encoder(**inputs)
        image_embeds = outputs.last_hidden_state
        image_embeds = self.projection(image_embeds)
        return image_embeds
    def create_combined_embeddings(self, input_ids, image_embeds):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        text_embeds = self.decoder.transformer.wte(input_ids)
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_mask = torch.ones(batch_size, image_embeds.size(1), device=device)
        text_mask = (input_ids != self.tokenizer.pad_token_id).float()
        combined_attention_mask = torch.cat([image_mask, text_mask], dim=1)
        text_start_idx = image_embeds.size(1)
        return combined_embeds, combined_attention_mask, text_start_idx
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        image_embeds = self.encode_images(images)
        combined_embeds, combined_attention_mask, text_start_idx = self.create_combined_embeddings(input_ids, image_embeds)
        batch_size, total_seq_len = combined_embeds.shape[:2]
        device = combined_embeds.device
        # Causal mask: text tokens can't attend to future text tokens
        causal_mask = torch.triu(torch.ones(total_seq_len, total_seq_len, device=device), diagonal=1).bool()
        for i in range(text_start_idx, total_seq_len):
            causal_mask[i, i:] = True
        combined_attention_mask = combined_attention_mask.unsqueeze(1).unsqueeze(2)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Convert both masks to boolean for bitwise operations
        combined_attention_mask = combined_attention_mask.bool()
        final_attention_mask = combined_attention_mask & ~causal_mask
        if labels is not None:
            padded_labels = torch.full((batch_size, total_seq_len), -100, device=device, dtype=labels.dtype)
            padded_labels[:, text_start_idx:text_start_idx + labels.size(1)] = labels
        else:
            padded_labels = None
        hidden_states = combined_embeds
        for layer in self.decoder.transformer.h:
            hidden_states = layer.ln_1(hidden_states)
            attn_output = layer.attn(
                hidden_states,
                attention_mask=final_attention_mask,
                head_mask=None,
                output_attentions=False
            )[0]
            hidden_states = hidden_states + attn_output
            hidden_states = layer.ln_2(hidden_states)
            mlp_output = layer.mlp(hidden_states)
            hidden_states = hidden_states + mlp_output
        hidden_states = self.decoder.transformer.ln_f(hidden_states)
        text_hidden_states = hidden_states[:, text_start_idx:, :]
        logits = self.decoder.lm_head(text_hidden_states)
        if padded_labels is not None:
            text_labels = padded_labels[:, text_start_idx:]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), text_labels.reshape(-1))
        else:
            loss = None
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        })()
    def generate_caption(self, image, max_length=50, num_beams=5, temperature=0.8):
        self.eval()
        image_embeds = self.encode_images(image)
        input_ids = torch.tensor([[self.image_start_token_id]], device=next(self.parameters()).device)
        with torch.no_grad():
            generated_ids = self._generate_with_self_attention(image_embeds, input_ids, max_length, temperature)
        caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    def _generate_with_self_attention(self, image_embeds, input_ids, max_length, temperature):
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        for step in range(max_length):
            combined_embeds, combined_attention_mask, text_start_idx = self.create_combined_embeddings(input_ids, image_embeds)
            batch_size, total_seq_len = combined_embeds.shape[:2]
            causal_mask = torch.triu(torch.ones(total_seq_len, total_seq_len, device=device), diagonal=1).bool()
            for i in range(text_start_idx, total_seq_len):
                causal_mask[i, i:] = True
            combined_attention_mask_ = combined_attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask_ = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Convert both masks to boolean for bitwise operations
            combined_attention_mask_ = combined_attention_mask_.bool()
            final_attention_mask = combined_attention_mask_ & ~causal_mask_
            hidden_states = combined_embeds
            for layer in self.decoder.transformer.h:
                hidden_states = layer.ln_1(hidden_states)
                attn_output = layer.attn(
                    hidden_states,
                    attention_mask=final_attention_mask,
                    head_mask=None,
                    output_attentions=False
                )[0]
                hidden_states = hidden_states + attn_output
                hidden_states = layer.ln_2(hidden_states)
                mlp_output = layer.mlp(hidden_states)
                hidden_states = hidden_states + mlp_output
            hidden_states = self.decoder.transformer.ln_f(hidden_states)
            text_hidden_states = hidden_states[:, text_start_idx:, :]
            logits = self.decoder.lm_head(text_hidden_states)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == self.tokenizer.eos_token_id).any():
                break
        return input_ids


def prepare_captions_for_self_attention(captions, tokenizer, max_text_length=77):
    processed_captions = []
    for caption in captions:
        processed_caption = f"<image> {caption} </image>"
        processed_captions.append(processed_caption)
    tokenized = tokenizer(
        processed_captions,
        padding=True,
        truncation=True,
        max_length=max_text_length + 2,
        return_tensors="pt"
    )
    return tokenized.input_ids, tokenized.attention_mask


def train_model(
    model: ImageCaptioningModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints",
    compute_eval_metrics: bool = True,
    test_dataset = None,
    test_samples_per_epoch: int = 3,
    test_metrics: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "image-captioning",
    wandb_run_name: str | None = None,
    eval_strategy: str = "meteor-centric"
):
    """
    Train the image captioning model.
    """
    # Initialize wandb if requested
    wandb_initialized = False
    if use_wandb:
        config = {
            "model": "CLIP-GPT2-ImageCaptioning",
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_dataloader.batch_size,
            "device": device,
            "compute_eval_metrics": compute_eval_metrics,
            "test_samples_per_epoch": test_samples_per_epoch,
            "eval_strategy": eval_strategy,
        }
        wandb_initialized = init_wandb(
            project_name=wandb_project,
            run_name=wandb_run_name,
            config=config
        )
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.eos_token_id)
    
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_bleu_score = 0.0
    best_metrics = {}
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n🔄 Starting Epoch {epoch+1}/{num_epochs}")
        print(f"📊 Current best_metrics: {best_metrics}")
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, captions, caption_tokens) in enumerate(progress_bar):
            # Skip if no tokenizer was used
            if caption_tokens is None:
                # Tokenize captions on the fly
                tokenized = model.tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                input_ids = tokenized.input_ids
                attention_mask = tokenized.attention_mask
            else:
                input_ids = caption_tokens
                attention_mask = torch.ones_like(input_ids)
            
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(images, input_ids, attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log to wandb every 128 batches
            if wandb_initialized and global_step % 128 == 0:
                log_to_wandb({
                    "train/loss": loss.item(),
                    "train/avg_loss": total_loss / num_batches,
                    "train/learning_rate": learning_rate,
                }, step=global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
        
        # Log epoch training metrics to wandb
        if wandb_initialized:
            log_to_wandb({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
            }, step=global_step)
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, captions, caption_tokens in val_dataloader:
                    if caption_tokens is None:
                        input_ids, attention_mask = prepare_captions_for_self_attention(
                            captions, model.tokenizer, max_text_length=77
                        )
                    else:
                        # Convert pre-tokenized captions back to text and re-tokenize with image tokens
                        try:
                            # Decode the pre-tokenized captions back to text
                            decoded_captions = []
                            for token_ids in caption_tokens:
                                # Remove padding tokens and decode
                                clean_tokens = [tid for tid in token_ids if tid != model.tokenizer.pad_token_id]
                                decoded_text = model.tokenizer.decode(clean_tokens, skip_special_tokens=True)
                                decoded_captions.append(decoded_text)
                            
                            # Re-tokenize with image tokens
                            input_ids, attention_mask = prepare_captions_for_self_attention(
                                decoded_captions, model.tokenizer, max_text_length=77
                            )
                        except Exception as e:
                            print(f"Warning: Failed to process pre-tokenized captions in validation: {e}")
                            continue
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    outputs = model(images, input_ids, attention_mask, labels=input_ids)
                    val_loss += outputs.loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Log validation loss to wandb
            if wandb_initialized:
                log_to_wandb({
                    "val/loss": avg_val_loss,
                }, step=global_step)
            
            # Compute evaluation metrics if requested
            eval_metrics = {}
            if compute_eval_metrics and val_dataloader is not None:
                eval_metrics = evaluate_model_on_validation(model, val_dataloader, device, max_samples=50)
                print(f"Evaluation Metrics:")
                for metric_name, score in eval_metrics.items():
                    print(f"  {metric_name}: {score:.4f}")
                
                # Log evaluation metrics to wandb
                if wandb_initialized:
                    log_to_wandb(eval_metrics, step=global_step, prefix="eval")
                
                # Check for evaluation-based improvement
                should_save_eval, reason = determine_model_save(eval_metrics, best_metrics, eval_strategy)
                
                if should_save_eval:
                    best_metrics = eval_metrics.copy()
                    print(f"✅ New best model based on {eval_strategy}! Reason: {reason}")
                    print(f"📊 Updated best_metrics: {best_metrics}")
                else:
                    print(f"❌ No evaluation-based improvement. Reason: {reason}")
                    
                    # Save evaluation-based best model
                    eval_save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'save_criterion': f'evaluation_{eval_strategy}',
                        'eval_metrics': eval_metrics,
                        'best_metrics': best_metrics,
                    }
                    torch.save(eval_save_dict, os.path.join(save_dir, 'best_model_eval.pth'))
                    print(f"Saved best model based on {eval_strategy} evaluation")
                    
                    # Log best metrics to wandb
                    if wandb_initialized:
                        for metric_name, score in eval_metrics.items():
                            log_to_wandb({
                                f"best/{metric_name.lower()}": score,
                            }, step=global_step)
                        
                        # Log model checkpoint to wandb
                        checkpoint_path = os.path.join(save_dir, 'best_model_eval.pth')
                        log_model_checkpoint_to_wandb(
                            checkpoint_path=checkpoint_path,
                            epoch=epoch + 1,
                            save_criterion=f'evaluation_{eval_strategy}',
                            val_loss=avg_val_loss,
                            eval_metrics=eval_metrics,
                            step=global_step
                        )
            
            # Always check for loss-based improvement and save if better
            loss_improved = avg_val_loss < best_val_loss
            if loss_improved:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {avg_val_loss:.4f}")
                
                # Save loss-based best model
                loss_save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'save_criterion': 'validation_loss',
                    'best_val_loss': best_val_loss,
                }
                torch.save(loss_save_dict, os.path.join(save_dir, 'best_model_loss.pth'))
                print(f"Saved best model based on validation loss: {avg_val_loss:.4f}")
                
                # Log best loss to wandb
                if wandb_initialized:
                    log_to_wandb({
                        "best/val_loss": best_val_loss,
                    }, step=global_step)
                    
                    # Log model checkpoint to wandb
                    checkpoint_path = os.path.join(save_dir, 'best_model_loss.pth')
                    log_model_checkpoint_to_wandb(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch + 1,
                        save_criterion='validation_loss',
                        val_loss=avg_val_loss,
                        step=global_step
                    )
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)
            
            # Log regular checkpoint to wandb
            if wandb_initialized:
                log_model_checkpoint_to_wandb(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch + 1,
                    save_criterion='regular_checkpoint',
                    val_loss=avg_val_loss if 'avg_val_loss' in locals() else 0.0,
                    step=global_step
                )
        
        # Test on random samples after each epoch
        if test_dataset and test_samples_per_epoch > 0:
            test_results, test_samples = test_model_on_random_samples_during_training(
                model, test_dataset, device, test_samples_per_epoch, 
                compute_metrics=test_metrics
            )
            print(f"Test Results after Epoch {epoch+1}:")
            for metric_name, score in test_results.items():
                print(f"  {metric_name}: {score:.4f}")
            
            # Log test metrics to wandb
            if wandb_initialized and test_results:
                log_to_wandb(test_results, step=global_step, prefix="test")
            
            # Log test images to wandb
            if wandb_initialized and test_samples:
                log_test_images_to_wandb(test_samples, epoch + 1, global_step)
    
    # Finish wandb run
    if wandb_initialized:
        finish_wandb()


def tokenize_caption(caption: str) -> list:
    """Tokenize caption into words."""
    if not EVALUATION_AVAILABLE:
        return caption.lower().split()
    
    try:
        return nltk.word_tokenize(caption.lower())
    except:
        return caption.lower().split()


def compute_bleu_scores(generated: str, references: list) -> dict:
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    if not EVALUATION_AVAILABLE:
        return {f'BLEU-{n}': 0.0 for n in range(1, 5)}
    
    try:
        generated_tokens = tokenize_caption(generated)
        reference_tokens = [tokenize_caption(ref) for ref in references]
        
        smoothing = SmoothingFunction().method1
        
        scores = {}
        for n in range(1, 5):
            weights = tuple([1.0/n] * n)
            score = sentence_bleu(reference_tokens, generated_tokens, 
                                weights=weights, smoothing_function=smoothing)
            scores[f'BLEU-{n}'] = score
            
        return scores
    except:
        return {f'BLEU-{n}': 0.0 for n in range(1, 5)}


def compute_meteor_score(generated: str, references: list) -> float:
    """Compute METEOR score."""
    if not EVALUATION_AVAILABLE:
        return 0.0
    
    try:
        generated_tokens = tokenize_caption(generated)
        reference_tokens = [tokenize_caption(ref) for ref in references]
        return meteor_score(reference_tokens, generated_tokens)
    except:
        return 0.0


def compute_rouge_scores(generated: str, references: list) -> dict:
    """Compute ROUGE scores."""
    if not EVALUATION_AVAILABLE:
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        reference = references[0] if references else ""
        scores = scorer.score(reference, generated)
        
        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }
    except:
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}


def evaluate_model_on_validation(model: ImageCaptioningModel, 
                                val_dataloader: DataLoader,
                                device: str,
                                max_samples: int = 100) -> dict:
    """Evaluate model on validation set and return metrics."""
    if not EVALUATION_AVAILABLE:
        return {'BLEU-4': 0.0, 'METEOR': 0.0, 'ROUGE-L': 0.0}
    
    model.eval()
    all_metrics = []
    sample_count = 0
    
    print("Computing evaluation metrics on validation set...")
    
    with torch.no_grad():
        for batch_idx, (images, captions, _) in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            if sample_count >= max_samples:
                break
                
            # Generate captions for each image in batch
            for i, image in enumerate(images):
                if sample_count >= max_samples:
                    break
                    
                # Convert tensor to PIL for generation
                if isinstance(image, torch.Tensor):
                    image = transforms.ToPILImage()(image)
                
                # Generate caption
                generated_caption = model.generate_caption(image)
                
                # Compute metrics
                bleu_scores = compute_bleu_scores(generated_caption, [captions[i]])
                meteor_score_val = compute_meteor_score(generated_caption, [captions[i]])
                rouge_scores = compute_rouge_scores(generated_caption, [captions[i]])
                
                metrics = {**bleu_scores, 'METEOR': meteor_score_val, **rouge_scores}
                all_metrics.append(metrics)
                sample_count += 1
    
    # Average metrics
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        avg_metrics[metric_name] = float(np.mean([m[metric_name] for m in all_metrics]))
    
    return avg_metrics


def log_model_checkpoint_to_wandb(checkpoint_path: str, epoch: int, save_criterion: str, 
                                 val_loss: float, eval_metrics: dict = None, step: int | None = None):
    """
    Log model checkpoint information to Weights & Biases.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        epoch: Epoch number when checkpoint was saved
        save_criterion: Reason why the model was saved
        val_loss: Validation loss at this epoch
        eval_metrics: Optional evaluation metrics
        step: Optional step number for logging
    """
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    # Check if the checkpoint file actually exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found, skipping W&B upload: {checkpoint_path}")
        return
    
    try:
        # Create checkpoint artifact
        artifact = wandb.Artifact(
            name=f"model-checkpoint-epoch-{epoch}",
            type="model",
            description=f"Model saved at epoch {epoch} based on {save_criterion}"
        )
        
        # Add the checkpoint file to the artifact
        artifact.add_file(checkpoint_path)
        
        # Log the artifact
        wandb.log_artifact(artifact)
        
        # Log metadata
        metadata = {
            f"checkpoint/epoch": epoch,
            f"checkpoint/save_criterion": save_criterion,
            f"checkpoint/val_loss": val_loss,
            f"checkpoint/path": checkpoint_path,
        }
        
        if eval_metrics:
            for metric_name, score in eval_metrics.items():
                metadata[f"checkpoint/{metric_name.lower()}"] = score
        
        if step is not None:
            wandb.log(metadata, step=step)
        else:
            wandb.log(metadata)
        
        print(f"📦 Logged model checkpoint to W&B: {checkpoint_path}")
        
    except Exception as e:
        print(f"Warning: Failed to log model checkpoint to W&B: {e}")
        import traceback
        traceback.print_exc()


def log_test_images_to_wandb(test_samples: list, epoch: int, step: int | None = None):
    """
    Log test images with captions to Weights & Biases.
    
    Args:
        test_samples: List of dicts with 'image', 'reference_caption', 'generated_caption', 'metrics'
        epoch: Current epoch number
        step: Optional step number for logging
    """
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    try:
        # Create a table for the test samples
        test_data = []
        
        for i, sample in enumerate(test_samples):
            # Convert PIL image to wandb image
            wandb_image = wandb.Image(
                sample['image'],
                caption=f"Generated: {sample['generated_caption']}\nReference: {sample['reference_caption']}"
            )
            
            # Create row data
            row_data = {
                "sample_id": i + 1,
                "image": wandb_image,
                "reference_caption": sample['reference_caption'],
                "generated_caption": sample['generated_caption'],
            }
            
            # Add metrics if available
            if sample['metrics']:
                for metric_name, score in sample['metrics'].items():
                    row_data[f"metric_{metric_name.lower()}"] = f"{score:.4f}"
            
            test_data.append(row_data)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(test_data)
        
        # Log as a table
        if step is not None:
            wandb.log({
                f"test_samples_epoch_{epoch}": wandb.Table(dataframe=df)
            }, step=step)
        else:
            wandb.log({
                f"test_samples_epoch_{epoch}": wandb.Table(dataframe=df)
            })
        
        print(f"📸 Logged {len(test_samples)} test samples to W&B for epoch {epoch}")
        
    except Exception as e:
        print(f"Warning: Failed to log test images to W&B: {e}")
        import traceback
        traceback.print_exc()


def test_model_on_random_samples_during_training(
    model: ImageCaptioningModel,
    test_dataset,
    device: str,
    num_samples: int = 3,
    compute_metrics: bool = True
) -> tuple[dict, list]:
    """
    Test the model on random samples during training (without plotting).
    
    Args:
        model: The trained ImageCaptioningModel
        test_dataset: Dataset to test on
        device: Device to use
        num_samples: Number of random samples to test
        compute_metrics: Whether to compute evaluation metrics
    
    Returns:
        tuple: (evaluation_metrics, test_samples)
            - evaluation_metrics: dict of average metrics
            - test_samples: list of dicts with image, reference_caption, generated_caption, metrics
    """
    model.eval()
    
    # Get random samples
    dataset_size = len(test_dataset)
    if dataset_size == 0:
        print("❌ No samples found in test dataset!")
        return {}, []
    
    if dataset_size < num_samples:
        num_samples = dataset_size
    
    # Get random indices
    import random
    random_indices = random.sample(range(dataset_size), num_samples)
    selected_samples = [test_dataset[idx] for idx in random_indices]
    
    all_metrics = []
    test_samples = []
    
    print(f"\n{'='*50}")
    print(f"TESTING ON {num_samples} RANDOM SAMPLES")
    print(f"{'='*50}")
    
    with torch.no_grad():
        for i, (image, caption, _) in enumerate(selected_samples):
            # Convert tensor to PIL for generation
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            # Generate caption
            generated_caption = model.generate_caption(image)
            
            # Compute metrics
            metrics = {}
            if compute_metrics:
                bleu_scores = compute_bleu_scores(generated_caption, [caption])
                meteor_score_val = compute_meteor_score(generated_caption, [caption])
                rouge_scores = compute_rouge_scores(generated_caption, [caption])
                
                metrics = {**bleu_scores, 'METEOR': meteor_score_val, **rouge_scores}
                all_metrics.append(metrics)
            
            # Store sample data
            sample_data = {
                'image': image,
                'reference_caption': caption,
                'generated_caption': generated_caption,
                'metrics': metrics
            }
            test_samples.append(sample_data)
            
            # Display results
            print(f"\n📸 Sample {i+1}:")
            print(f"   Generated: {generated_caption}")
            print(f"   Reference: {caption}")
            if compute_metrics:
                print(f"   BLEU-4: {metrics.get('BLEU-4', 0):.4f}")
                print(f"   METEOR: {metrics.get('METEOR', 0):.4f}")
                print(f"   ROUGE-L: {metrics.get('ROUGE-L', 0):.4f}")
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = sum(m[metric_name] for m in all_metrics) / len(all_metrics)
        
        print(f"\n📊 AVERAGE METRICS:")
        print(f"{'='*30}")
        for metric_name, score in avg_metrics.items():
            print(f"   {metric_name}: {score:.4f}")
        
        return avg_metrics, test_samples
    else:
        return {}, test_samples


def save_model_to_huggingface(
    model: ImageCaptioningModel,
    repo_name: str,
    token: Optional[str] = None,
    commit_message: str = "Add image captioning model",
    private: bool = False
) -> bool:
    """
    Save the trained model to Hugging Face Hub.
    
    Args:
        model: The trained ImageCaptioningModel
        repo_name: Repository name (e.g., "username/model-name")
        token: Hugging Face token (if None, will try environment variable HUGGINGFACE_TOKEN)
        commit_message: Commit message for the upload
        private: Whether the repository should be private
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not HF_AVAILABLE:
        print("Error: Hugging Face Hub not available. Install with: pip install huggingface_hub")
        return False
    
    try:
        # Get token from environment variable if not provided
        if token is None:
            token = os.environ.get('HUGGINGFACE_TOKEN')
            if token:
                print("Using HUGGINGFACE_TOKEN from environment variable")
            else:
                print("No token provided and HUGGINGFACE_TOKEN environment variable not set")
                print("Please set HUGGINGFACE_TOKEN or provide --hf-token argument")
                return False
        
        # Login to Hugging Face
        login(token=token)
        
        # Create a custom model class that can be saved to HF Hub
        class HFImageCaptioningModel(PreTrainedModel):
            def __init__(self, model: ImageCaptioningModel):
                super().__init__(model.decoder.config)
                self.model = model
                self.config = model.decoder.config
                
                # Essential HF Hub attributes
                self.config.model_type = "gpt2"
                self.config.architectures = ["GPT2LMHeadModel"]
            
            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)
            
            def generate_caption(self, image, **kwargs):
                return self.model.generate_caption(image, **kwargs)
            
            def encode_images(self, images):
                return self.model.encode_images(images)
        
        # Wrap the model
        hf_model = HFImageCaptioningModel(model)
        
        # Save model and tokenizer directly to HF Hub
        hf_model.save_pretrained(
            save_directory="./temp_hf_model",
            repo_id=repo_name,
            push_to_hub=True,
            commit_message=commit_message,
            private=private
        )
        
        # Save tokenizer separately
        model.tokenizer.push_to_hub(
            repo_id=repo_name,
            commit_message=f"{commit_message} - tokenizer",
            private=private
        )
        
        # Clean up temporary directory
        import shutil
        if os.path.exists("./temp_hf_model"):
            shutil.rmtree("./temp_hf_model")
        
        print(f"✅ Model successfully saved to Hugging Face Hub: https://huggingface.co/{repo_name}")
        print(f"📝 You can now use it with:")
        print(f"   from transformers import AutoTokenizer, AutoModel")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{repo_name}')")
        print(f"   model = AutoModel.from_pretrained('{repo_name}')")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model to Hugging Face Hub: {e}")
        return False


def select_device():
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_best_model(model: ImageCaptioningModel, save_dir: str = "checkpoints", 
                   model_type: str = "eval") -> tuple[ImageCaptioningModel, dict]:
    """
    Load the best model from checkpoints.
    
    Args:
        model: The model instance to load weights into
        save_dir: Directory containing saved models
        model_type: Type of model to load ("eval", "loss", or "auto")
                   - "eval": Load best_model_eval.pth (evaluation-based)
                   - "loss": Load best_model_loss.pth (loss-based)
                   - "auto": Automatically choose based on availability
    
    Returns:
        tuple: (loaded_model, metadata)
    """
    if model_type == "auto":
        # Try evaluation model first, then loss model
        eval_path = os.path.join(save_dir, 'best_model_eval.pth')
        loss_path = os.path.join(save_dir, 'best_model_loss.pth')
        
        if os.path.exists(eval_path):
            model_type = "eval"
            print("Found evaluation-based model, loading best_model_eval.pth")
        elif os.path.exists(loss_path):
            model_type = "loss"
            print("Found loss-based model, loading best_model_loss.pth")
        else:
            raise FileNotFoundError(f"No saved models found in {save_dir}")
    
    if model_type == "eval":
        checkpoint_path = os.path.join(save_dir, 'best_model_eval.pth')
    elif model_type == "loss":
        checkpoint_path = os.path.join(save_dir, 'best_model_loss.pth')
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Use 'eval', 'loss', or 'auto'")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading {model_type}-based model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model information
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Save criterion: {checkpoint.get('save_criterion', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    if 'eval_metrics' in checkpoint:
        print("Evaluation metrics:")
        for metric, score in checkpoint['eval_metrics'].items():
            print(f"  {metric}: {score:.4f}")
    
    return model, checkpoint


def list_available_models(save_dir: str = "checkpoints") -> dict:
    """
    List all available saved models and their metadata.
    
    Args:
        save_dir: Directory containing saved models
    
    Returns:
        dict: Information about available models
    """
    models_info = {}
    
    # Check for evaluation-based model
    eval_path = os.path.join(save_dir, 'best_model_eval.pth')
    if os.path.exists(eval_path):
        checkpoint = torch.load(eval_path, map_location='cpu')
        models_info['best_model_eval.pth'] = {
            'epoch': checkpoint['epoch'],
            'save_criterion': checkpoint.get('save_criterion', 'unknown'),
            'val_loss': checkpoint.get('val_loss', 'unknown'),
            'eval_metrics': checkpoint.get('eval_metrics', {}),
        }
    
    # Check for loss-based model
    loss_path = os.path.join(save_dir, 'best_model_loss.pth')
    if os.path.exists(loss_path):
        checkpoint = torch.load(loss_path, map_location='cpu')
        models_info['best_model_loss.pth'] = {
            'epoch': checkpoint['epoch'],
            'save_criterion': checkpoint.get('save_criterion', 'unknown'),
            'val_loss': checkpoint.get('val_loss', 'unknown'),
        }
    
    # Check for regular checkpoints
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        models_info['checkpoints'] = checkpoint_files
    
    return models_info


def train_model_self_attention(
    model: ImageCaptioningModelSelfAttention,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints",
    compute_eval_metrics: bool = True,
    test_dataset = None,
    test_samples_per_epoch: int = 3,
    test_metrics: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "image-captioning-self-attn",
    wandb_run_name: str | None = None,
    eval_strategy: str = "meteor-centric"
):
    """
    Train the self-attention image captioning model.
    """
    wandb_initialized = False
    if use_wandb:
        config = {
            "model": "CLIP-GPT2-SelfAttention-ImageCaptioning",
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_dataloader.batch_size,
            "device": device,
            "compute_eval_metrics": compute_eval_metrics,
            "test_samples_per_epoch": test_samples_per_epoch,
            "eval_strategy": eval_strategy,
        }
        wandb_initialized = init_wandb(
            project_name=wandb_project,
            run_name=wandb_run_name,
            config=config
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=learning_rate)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_metrics = {}
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\n🔄 Starting Epoch {epoch+1}/{num_epochs}")
        print(f"📊 Current best_metrics: {best_metrics}")
        model.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, captions, caption_tokens) in enumerate(progress_bar):
            # Prepare captions for self-attention model
            if caption_tokens is None:
                input_ids, attention_mask = prepare_captions_for_self_attention(
                    captions, model.tokenizer, max_text_length=77
                )
            else:
                # Convert pre-tokenized captions back to text and re-tokenize with image tokens
                try:
                    # Decode the pre-tokenized captions back to text
                    decoded_captions = []
                    for token_ids in caption_tokens:
                        # Remove padding tokens and decode
                        clean_tokens = [tid for tid in token_ids if tid != model.tokenizer.pad_token_id]
                        decoded_text = model.tokenizer.decode(clean_tokens, skip_special_tokens=True)
                        decoded_captions.append(decoded_text)
                    
                    # Re-tokenize with image tokens
                    input_ids, attention_mask = prepare_captions_for_self_attention(
                        decoded_captions, model.tokenizer, max_text_length=77
                    )
                except Exception as e:
                    print(f"Warning: Failed to process pre-tokenized captions: {e}")
                    continue
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(images, input_ids, attention_mask, labels=input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if wandb_initialized and global_step % 128 == 0:
                log_to_wandb({
                    "train/loss": loss.item(),
                    "train/avg_loss": total_loss / num_batches,
                    "train/learning_rate": learning_rate,
                }, step=global_step)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
        
        if wandb_initialized:
            log_to_wandb({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
            }, step=global_step)
        
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, captions, caption_tokens in val_dataloader:
                    if caption_tokens is None:
                        input_ids, attention_mask = prepare_captions_for_self_attention(
                            captions, model.tokenizer, max_text_length=77
                        )
                    else:
                        # Convert pre-tokenized captions back to text and re-tokenize with image tokens
                        try:
                            # Decode the pre-tokenized captions back to text
                            decoded_captions = []
                            for token_ids in caption_tokens:
                                # Remove padding tokens and decode
                                clean_tokens = [tid for tid in token_ids if tid != model.tokenizer.pad_token_id]
                                decoded_text = model.tokenizer.decode(clean_tokens, skip_special_tokens=True)
                                decoded_captions.append(decoded_text)
                            
                            # Re-tokenize with image tokens
                            input_ids, attention_mask = prepare_captions_for_self_attention(
                                decoded_captions, model.tokenizer, max_text_length=77
                            )
                        except Exception as e:
                            print(f"Warning: Failed to process pre-tokenized captions in validation: {e}")
                            continue
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    outputs = model(images, input_ids, attention_mask, labels=input_ids)
                    val_loss += outputs.loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            if wandb_initialized:
                log_to_wandb({
                    "val/loss": avg_val_loss,
                }, step=global_step)
            
            eval_metrics = {}
            if compute_eval_metrics and val_dataloader is not None:
                eval_metrics = evaluate_model_on_validation_self_attention(
                    model, val_dataloader, device, max_samples=50
                )
                print(f"Evaluation Metrics:")
                for metric_name, score in eval_metrics.items():
                    print(f"  {metric_name}: {score:.4f}")
                
                if wandb_initialized:
                    log_to_wandb(eval_metrics, step=global_step, prefix="eval")
                
                should_save_eval, reason = determine_model_save(eval_metrics, best_metrics, eval_strategy)
                if should_save_eval:
                    best_metrics = eval_metrics.copy()
                    print(f"✅ New best model based on {eval_strategy}! Reason: {reason}")
                    print(f"📊 Updated best_metrics: {best_metrics}")
                    eval_save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'save_criterion': f'evaluation_{eval_strategy}',
                        'eval_metrics': eval_metrics,
                        'best_metrics': best_metrics,
                    }
                    torch.save(eval_save_dict, os.path.join(save_dir, 'best_model_eval_self_attn.pth'))
                    print(f"Saved best model based on {eval_strategy} evaluation")
                    if wandb_initialized:
                        for metric_name, score in eval_metrics.items():
                            log_to_wandb({
                                f"best/{metric_name.lower()}": score,
                            }, step=global_step)
                        checkpoint_path = os.path.join(save_dir, 'best_model_eval_self_attn.pth')
                        log_model_checkpoint_to_wandb(
                            checkpoint_path=checkpoint_path,
                            epoch=epoch + 1,
                            save_criterion=f'evaluation_{eval_strategy}',
                            val_loss=avg_val_loss,
                            eval_metrics=eval_metrics,
                            step=global_step
                        )
                else:
                    print(f"❌ No evaluation-based improvement. Reason: {reason}")
            
            loss_improved = avg_val_loss < best_val_loss
            if loss_improved:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {avg_val_loss:.4f}")
                loss_save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'save_criterion': 'validation_loss',
                    'best_val_loss': best_val_loss,
                }
                torch.save(loss_save_dict, os.path.join(save_dir, 'best_model_loss_self_attn.pth'))
                print(f"Saved best model based on validation loss: {avg_val_loss:.4f}")
                if wandb_initialized:
                    log_to_wandb({
                        "best/val_loss": best_val_loss,
                    }, step=global_step)
                    checkpoint_path = os.path.join(save_dir, 'best_model_loss_self_attn.pth')
                    log_model_checkpoint_to_wandb(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch + 1,
                        save_criterion='validation_loss',
                        val_loss=avg_val_loss,
                        step=global_step
                    )
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_self_attn.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)
            if wandb_initialized:
                log_model_checkpoint_to_wandb(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch + 1,
                    save_criterion='regular_checkpoint',
                    val_loss=avg_val_loss if 'avg_val_loss' in locals() else 0.0,
                    step=global_step
                )
        
        if test_dataset and test_samples_per_epoch > 0:
            test_results, test_samples = test_model_on_random_samples_during_training_self_attention(
                model, test_dataset, device, test_samples_per_epoch, 
                compute_metrics=test_metrics
            )
            print(f"Test Results after Epoch {epoch+1}:")
            for metric_name, score in test_results.items():
                print(f"  {metric_name}: {score:.4f}")
            
            if wandb_initialized and test_results:
                log_to_wandb(test_results, step=global_step, prefix="test")
            
            if wandb_initialized and test_samples:
                log_test_images_to_wandb(test_samples, epoch + 1, global_step)
    
    if wandb_initialized:
        finish_wandb()


def evaluate_model_on_validation_self_attention(model: ImageCaptioningModelSelfAttention, 
                                              val_dataloader: DataLoader,
                                              device: str,
                                              max_samples: int = 100) -> dict:
    if not EVALUATION_AVAILABLE:
        return {'BLEU-4': 0.0, 'METEOR': 0.0, 'ROUGE-L': 0.0}
    model.eval()
    all_metrics = []
    sample_count = 0
    print("Computing evaluation metrics on validation set...")
    with torch.no_grad():
        for batch_idx, (images, captions, _) in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            if sample_count >= max_samples:
                break
            for i, image in enumerate(images):
                if sample_count >= max_samples:
                    break
                if isinstance(image, torch.Tensor):
                    image = transforms.ToPILImage()(image)
                generated_caption = model.generate_caption(image)
                generated_caption = generated_caption.replace("<image>", "").replace("</image>", "").strip()
                bleu_scores = compute_bleu_scores(generated_caption, [captions[i]])
                meteor_score_val = compute_meteor_score(generated_caption, [captions[i]])
                rouge_scores = compute_rouge_scores(generated_caption, [captions[i]])
                metrics = {**bleu_scores, 'METEOR': meteor_score_val, **rouge_scores}
                all_metrics.append(metrics)
                sample_count += 1
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        avg_metrics[metric_name] = float(np.mean([m[metric_name] for m in all_metrics]))
    return avg_metrics


def test_model_on_random_samples_during_training_self_attention(
    model: ImageCaptioningModelSelfAttention,
    test_dataset,
    device: str,
    num_samples: int = 3,
    compute_metrics: bool = True
) -> tuple[dict, list]:
    model.eval()
    dataset_size = len(test_dataset)
    if dataset_size == 0:
        print("❌ No samples found in test dataset!")
        return {}, []
    if dataset_size < num_samples:
        num_samples = dataset_size
    import random
    random_indices = random.sample(range(dataset_size), num_samples)
    selected_samples = [test_dataset[idx] for idx in random_indices]
    all_metrics = []
    test_samples = []
    print(f"\n{'='*50}")
    print(f"TESTING SELF-ATTENTION MODEL ON {num_samples} RANDOM SAMPLES")
    print(f"{'='*50}")
    with torch.no_grad():
        for i, (image, caption, _) in enumerate(selected_samples):
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            generated_caption = model.generate_caption(image)
            generated_caption = generated_caption.replace("<image>", "").replace("</image>", "").strip()
            metrics = {}
            if compute_metrics:
                bleu_scores = compute_bleu_scores(generated_caption, [caption])
                meteor_score_val = compute_meteor_score(generated_caption, [caption])
                rouge_scores = compute_rouge_scores(generated_caption, [caption])
                metrics = {**bleu_scores, 'METEOR': meteor_score_val, **rouge_scores}
                all_metrics.append(metrics)
            sample_data = {
                'image': image,
                'reference_caption': caption,
                'generated_caption': generated_caption,
                'metrics': metrics
            }
            test_samples.append(sample_data)
            print(f"\n📸 Sample {i+1}:")
            print(f"   Generated: {generated_caption}")
            print(f"   Reference: {caption}")
            if compute_metrics:
                print(f"   BLEU-4: {metrics.get('BLEU-4', 0):.4f}")
                print(f"   METEOR: {metrics.get('METEOR', 0):.4f}")
                print(f"   ROUGE-L: {metrics.get('ROUGE-L', 0):.4f}")
    if all_metrics:
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = sum(m[metric_name] for m in all_metrics) / len(all_metrics)
        print(f"\n📊 AVERAGE METRICS:")
        print(f"{'='*30}")
        for metric_name, score in avg_metrics.items():
            print(f"   {metric_name}: {score:.4f}")
        return avg_metrics, test_samples
    else:
        return {}, test_samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train image captioning model (cross-attn or self-attn)")
    parser.add_argument("--data-dir", type=str, default="data/flickr30k", 
                       help="Data directory (options: data/flickr30k, data/coco_captions, data/flickr8k)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation metrics during training")
    parser.add_argument("--save-to-hf", type=str, default=None, help="Save model to Hugging Face Hub (format: username/model-name)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (if not provided, will use HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--hf-private", action="store_true", help="Make Hugging Face repository private")
    parser.add_argument("--test-samples-per-epoch", type=int, default=3, help="Number of random samples to test after each epoch")
    parser.add_argument("--no-test", action="store_true", help="Disable testing on random samples during training")
    parser.add_argument("--no-test-metrics", action="store_true", help="Disable evaluation metrics during in-training testing")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="image-captioning", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--eval-strategy", type=str, default="meteor-centric", 
                       choices=["weighted-composite", "pareto", "multi-criteria", "meteor-centric", "bleu", "rouge", "meteor"],
                       help="Evaluation strategy for model saving")
    parser.add_argument("--self-attn", action="store_true", help="Use self-attention model instead of cross-attention")
    args = parser.parse_args()
    # Get values from environment variables if not provided
    if args.save_to_hf is None:
        args.save_to_hf = os.environ.get('HF_REPO_NAME')
    if args.hf_token is None:
        args.hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not args.hf_private:
        args.hf_private = os.environ.get('HF_PRIVATE', '').lower() in ('true', '1', 'yes')
    if not args.wandb:
        args.wandb = os.environ.get('WANDB_ENABLED', '').lower() in ('true', '1', 'yes')
    if args.wandb_project == "image-captioning" and os.environ.get('WANDB_PROJECT'):
        args.wandb_project = os.environ.get('WANDB_PROJECT')
    if args.wandb_run_name is None:
        args.wandb_run_name = os.environ.get('WANDB_RUN_NAME')
    if args.wandb_project is None:
        args.wandb_project = "image-captioning"
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    device = args.device if args.device else select_device()
    compute_eval_metrics = not args.no_eval
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Evaluation metrics: {'Enabled' if compute_eval_metrics else 'Disabled'}")
    if not os.path.exists(data_dir):
        print("Creating sample Flickr dataset...")
        create_sample_flickr_data(data_dir, num_samples=100)
        print("Please add actual images to the dataset before training!")
        return
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    if args.self_attn:
        print("\n*** Using SELF-ATTENTION model ***\n")
        model = ImageCaptioningModelSelfAttention()
    else:
        print("\n*** Using CROSS-ATTENTION model ***\n")
        model = ImageCaptioningModel()
    train_dataset = FlickrDataset(
        root_dir=data_dir,
        split="train",
        tokenizer=model.tokenizer,
        transform=image_transforms
    )
    val_dataset = FlickrDataset(
        root_dir=data_dir,
        split="val",
        tokenizer=model.tokenizer,
        transform=image_transforms
    )
    test_dataset = FlickrDataset(
        root_dir=data_dir,
        split="train",
        tokenizer=model.tokenizer,
        transform=image_transforms
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device != "cpu" else False,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device != "cpu" else False,
        persistent_workers=True,
        prefetch_factor=2
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Testing during training: {'Enabled' if not args.no_test else 'Disabled'}")
    if not args.no_test:
        print(f"Test metrics: {'Enabled' if not args.no_test_metrics else 'Disabled'}")
        print(f"Test samples per epoch: {args.test_samples_per_epoch}")
    if args.self_attn:
        train_model_self_attention(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            compute_eval_metrics=compute_eval_metrics,
            test_dataset=test_dataset if not args.no_test else None,
            test_samples_per_epoch=args.test_samples_per_epoch,
            test_metrics=not args.no_test_metrics,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            eval_strategy=args.eval_strategy
        )
        print("Self-attention training completed!")
    else:
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            compute_eval_metrics=compute_eval_metrics,
            test_dataset=test_dataset if not args.no_test else None,
            test_samples_per_epoch=args.test_samples_per_epoch,
            test_metrics=not args.no_test_metrics,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            eval_strategy=args.eval_strategy
        )
        print("Cross-attention training completed!")
    if args.save_to_hf:
        if args.self_attn:
            print(f"\n🔄 Saving self-attention model to Hugging Face Hub: {args.save_to_hf}")
            success = save_model_to_huggingface(
                model=model,
                repo_name=args.save_to_hf,
                token=args.hf_token,
                commit_message=f"Add self-attention image captioning model trained on {data_dir}",
                private=args.hf_private
            )
            if success:
                print("✅ Self-attention model successfully uploaded to Hugging Face Hub!")
            else:
                print("❌ Failed to upload self-attention model to Hugging Face Hub")
        else:
            print(f"\n🔄 Saving cross-attention model to Hugging Face Hub: {args.save_to_hf}")
            success = save_model_to_huggingface(
                model=model,
                repo_name=args.save_to_hf,
                token=args.hf_token,
                commit_message=f"Add image captioning model trained on {data_dir}",
                private=args.hf_private
            )
            if success:
                print("✅ Model successfully uploaded to Hugging Face Hub!")
            else:
                print("❌ Failed to upload model to Hugging Face Hub")


if __name__ == "__main__":
    main() 