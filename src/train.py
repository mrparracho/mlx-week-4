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

from src.data.flickr_dataset import FlickrDataset, FlickrDataLoader, create_sample_flickr_data


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
    
    def generate_caption(self, image, max_length=50, num_beams=5):
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
            temperature=0.8,  # Slightly higher temperature
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
    test_metrics: bool = True
):
    """
    Train the image captioning model.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.eos_token_id)
    
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_bleu_score = 0.0
    
    for epoch in range(num_epochs):
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, captions, caption_tokens in val_dataloader:
                    if caption_tokens is None:
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
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    outputs = model(images, input_ids, attention_mask, labels=input_ids)
                    val_loss += outputs.loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Compute evaluation metrics if requested
            eval_metrics = {}
            if compute_eval_metrics and val_dataloader is not None:
                eval_metrics = evaluate_model_on_validation(model, val_dataloader, device, max_samples=50)
                print(f"Evaluation Metrics:")
                for metric_name, score in eval_metrics.items():
                    print(f"  {metric_name}: {score:.4f}")
            
            # Save best model based on BLEU-4 score if available, otherwise use loss
            should_save = False
            if eval_metrics and 'BLEU-4' in eval_metrics:
                if eval_metrics['BLEU-4'] > best_bleu_score:
                    best_bleu_score = eval_metrics['BLEU-4']
                    should_save = True
                    print(f"New best BLEU-4 score: {best_bleu_score:.4f}")
            elif avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                should_save = True
                print(f"New best validation loss: {avg_val_loss:.4f}")
            
            if should_save:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }
                if eval_metrics:
                    save_dict['eval_metrics'] = eval_metrics
                
                torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model!")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Test on random samples after each epoch
        if test_dataset and test_samples_per_epoch > 0:
            test_results = test_model_on_random_samples_during_training(
                model, test_dataset, device, test_samples_per_epoch, 
                compute_metrics=test_metrics
            )
            print(f"Test Results after Epoch {epoch+1}:")
            for metric_name, score in test_results.items():
                print(f"  {metric_name}: {score:.4f}")


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
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    return avg_metrics


def test_model_on_random_samples_during_training(
    model: ImageCaptioningModel,
    test_dataset,
    device: str,
    num_samples: int = 3,
    compute_metrics: bool = True
) -> dict:
    """
    Test the model on random samples during training (without plotting).
    
    Args:
        model: The trained ImageCaptioningModel
        test_dataset: Dataset to test on
        device: Device to use
        num_samples: Number of random samples to test
        compute_metrics: Whether to compute evaluation metrics
    
    Returns:
        dict: Evaluation metrics for the samples
    """
    model.eval()
    
    # Get random samples
    dataset_size = len(test_dataset)
    if dataset_size == 0:
        print("❌ No samples found in test dataset!")
        return {}
    
    if dataset_size < num_samples:
        num_samples = dataset_size
    
    # Get random indices
    import random
    random_indices = random.sample(range(dataset_size), num_samples)
    selected_samples = [test_dataset[idx] for idx in random_indices]
    
    all_metrics = []
    
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
            if compute_metrics:
                bleu_scores = compute_bleu_scores(generated_caption, [caption])
                meteor_score_val = compute_meteor_score(generated_caption, [caption])
                rouge_scores = compute_rouge_scores(generated_caption, [caption])
                
                metrics = {**bleu_scores, 'METEOR': meteor_score_val, **rouge_scores}
                all_metrics.append(metrics)
            
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
        
        return avg_metrics
    else:
        return {}


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


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train image captioning model")
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
    
    args = parser.parse_args()
    
    # Get values from environment variables if not provided
    if args.save_to_hf is None:
        args.save_to_hf = os.environ.get('HF_REPO_NAME')
    
    if args.hf_token is None:
        args.hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if not args.hf_private:
        # Check if private flag is set via environment
        args.hf_private = os.environ.get('HF_PRIVATE', '').lower() in ('true', '1', 'yes')
    
    # Configuration
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
    
    # Create sample data if it doesn't exist
    if not os.path.exists(data_dir):
        print("Creating sample Flickr dataset...")
        create_sample_flickr_data(data_dir, num_samples=100)
        print("Please add actual images to the dataset before training!")
        return
    
    # Initialize model
    model = ImageCaptioningModel()
    
    # Create image transforms - just resize and convert to tensor
    # CLIP will handle its own preprocessing
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
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
    
    # Create test dataset for random sample testing
    test_dataset = FlickrDataset(
        root_dir=data_dir,
        split="train",  # Use train split since that's what we have
        tokenizer=model.tokenizer,
        transform=image_transforms
    )
    
    # Create dataloaders with optimizations
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Increased workers
        pin_memory=True if device != "cpu" else False,  # Pin memory for GPU
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch batches
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Increased workers
        pin_memory=True if device != "cpu" else False,  # Pin memory for GPU
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch batches
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Testing during training: {'Enabled' if not args.no_test else 'Disabled'}")
    if not args.no_test:
        print(f"Test metrics: {'Enabled' if not args.no_test_metrics else 'Disabled'}")
        print(f"Test samples per epoch: {args.test_samples_per_epoch}")
    
    # Train model
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
        test_metrics=not args.no_test_metrics
    )
    
    print("Training completed!")
    
    # Save to Hugging Face Hub if requested
    if args.save_to_hf:
        print(f"\n🔄 Saving model to Hugging Face Hub: {args.save_to_hf}")
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