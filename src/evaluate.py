#!/usr/bin/env python3
"""
Evaluation metrics for image captioning models.
"""

import torch
from torch.utils.data import DataLoader
from PIL import Image
import json
import os
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import argparse

# For evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
except ImportError:
    print("Warning: Some evaluation metrics require additional packages.")
    print("Install with: pip install nltk rouge-score")

from src.train import ImageCaptioningModel, select_device
from src.data.flickr_dataset import FlickrDataset


def tokenize_caption(caption: str) -> List[str]:
    """Tokenize caption into words."""
    try:
        import nltk
        return nltk.word_tokenize(caption.lower())
    except:
        # Fallback to simple tokenization
        return caption.lower().split()


def compute_bleu_scores(generated: str, references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
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
    except ImportError:
        return {f'BLEU-{n}': 0.0 for n in range(1, 5)}


def compute_meteor_score(generated: str, references: List[str]) -> float:
    """Compute METEOR score."""
    try:
        from nltk.translate.meteor_score import meteor_score
        
        generated_tokens = tokenize_caption(generated)
        reference_tokens = [tokenize_caption(ref) for ref in references]
        
        return meteor_score(reference_tokens, generated_tokens)
    except ImportError:
        return 0.0


def compute_rouge_scores(generated: str, references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Use the first reference for ROUGE (or average over all)
        reference = references[0] if references else ""
        scores = scorer.score(reference, generated)
        
        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }
    except ImportError:
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}


def compute_cider_score(generated: str, references: List[str]) -> float:
    """Compute CIDEr score (simplified version)."""
    # This is a simplified CIDEr implementation
    # Full CIDEr requires TF-IDF computation and multiple references
    try:
        generated_tokens = set(tokenize_caption(generated))
        reference_tokens = set()
        for ref in references:
            reference_tokens.update(tokenize_caption(ref))
        
        if not reference_tokens:
            return 0.0
        
        # Compute F1 score between generated and reference tokens
        intersection = len(generated_tokens.intersection(reference_tokens))
        precision = intersection / len(generated_tokens) if generated_tokens else 0
        recall = intersection / len(reference_tokens) if reference_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    except:
        return 0.0


def evaluate_caption(generated: str, references: List[str]) -> Dict[str, float]:
    """Evaluate a single generated caption against reference captions."""
    metrics = {}
    
    # BLEU scores
    bleu_scores = compute_bleu_scores(generated, references)
    metrics.update(bleu_scores)
    
    # METEOR score
    metrics['METEOR'] = compute_meteor_score(generated, references)
    
    # ROUGE scores
    rouge_scores = compute_rouge_scores(generated, references)
    metrics.update(rouge_scores)
    
    # CIDEr score
    metrics['CIDEr'] = compute_cider_score(generated, references)
    
    return metrics


def evaluate_model(model: ImageCaptioningModel, 
                  test_dataloader: DataLoader,
                  device: str) -> Dict[str, float]:
    """Evaluate the model on test dataset."""
    model.eval()
    
    all_metrics = []
    generated_captions = []
    reference_captions = []
    
    print("Generating captions for evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, captions, _) in enumerate(tqdm(test_dataloader)):
            # Generate captions for each image in batch
            for i, image in enumerate(images):
                # Convert tensor to PIL for generation
                if isinstance(image, torch.Tensor):
                    from torchvision import transforms
                    image = transforms.ToPILImage()(image)
                
                # Generate caption
                generated_caption = model.generate_caption(image)
                generated_captions.append(generated_caption)
                reference_captions.append(captions[i])
                
                # Evaluate this caption
                metrics = evaluate_caption(generated_caption, [captions[i]])
                all_metrics.append(metrics)
    
    # Average metrics across all samples
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    return avg_metrics, generated_captions, reference_captions


def main():
    parser = argparse.ArgumentParser(description="Evaluate image captioning model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/flickr30k", 
                       help="Data directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    device = args.device if args.device else select_device()
    print(f"Using device: {device}")
    
    # Load model
    model = ImageCaptioningModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    from torchvision import transforms
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = FlickrDataset(
        root_dir=args.data_dir,
        split="test",
        tokenizer=model.tokenizer,
        transform=image_transforms
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # Evaluate model
    metrics, generated_captions, reference_captions = evaluate_model(
        model, test_dataloader, device
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score:.4f}")
    
    # Save results
    results = {
        'metrics': metrics,
        'sample_predictions': [
            {
                'generated': gen,
                'reference': ref
            }
            for gen, ref in zip(generated_captions[:10], reference_captions[:10])
        ]
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")
    print(f"Sample predictions saved (first 10)")


if __name__ == "__main__":
    main() 