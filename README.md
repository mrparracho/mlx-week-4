# Image Captioning with CLIP + GPT-2

A PyTorch implementation of an image captioning model using CLIP as the vision encoder and GPT-2 as the text decoder with cross-attention. This project supports multiple datasets, comprehensive evaluation metrics, and integration with Weights & Biases and Hugging Face Hub.

## Features

- **CLIP Vision Encoder**: Uses OpenAI's CLIP model for robust image understanding
- **GPT-2 Decoder**: Generates captions with cross-attention to image features
- **Multiple Datasets**: Support for Flickr30k, COCO, and Flickr8k
- **Comprehensive Evaluation**: BLEU, METEOR, ROUGE scores with multiple evaluation strategies
- **Dual Model Saving**: Separate models for loss-based and evaluation-based optimization
- **Generation Control**: Configurable temperature, beam search, and length parameters
- **W&B Integration**: Complete logging of metrics, test images, and model checkpoints
- **Hugging Face Hub**: Easy model sharing and deployment
- **Cross-platform**: Supports CUDA, MPS (Apple Silicon), and CPU

## Quickstart

### 1. Setup

```bash
# Clone and setup (recommended)
./setup.sh            # Flickr30k (default)
./setup.sh coco       # COCO dataset
./setup.sh flickr8k   # Flickr8k dataset

# Or manual setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python scripts/download_flickr30k.py
```

### 2. Train

```bash
# Basic training
python -m src.train --data-dir data/flickr30k --num-epochs 10

# With W&B logging and HF Hub saving
python -m src.train \
  --data-dir data/flickr30k \
  --num-epochs 10 \
  --wandb \
  --wandb-project "my-captioning-experiment" \
  --save-to-hf "your-username/image-captioning-model"
```

### 3. Test (on training dataset samples)

```bash
# Test on random samples from the dataset
python -m src.test_model \
  --checkpoint checkpoints/best_model_eval.pth \
  --data-dir data/flickr30k \
  --num-samples 5

# Test with different generation parameters
python -m src.test_model \
  --checkpoint checkpoints/best_model_eval.pth \
  --temperature 1.2 \
  --max-length 60 \
  --num-beams 3
```

### 4. Inference (on your own images)

```bash
# Basic inference
python -m src.infer notebooks/my_image.jpg \
  --checkpoint checkpoints/best_model_eval.pth

# With custom generation parameters
python -m src.infer notebooks/my_image.jpg \
  --checkpoint checkpoints/best_model_eval.pth \
  --temperature 0.9 \
  --max-length 50 \
  --num-beams 5

# Debug mode for troubleshooting
python -m src.debug_infer notebooks/my_image.jpg \
  --checkpoint checkpoints/best_model_eval.pth \
  --temperature 1.0
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Dependencies

```bash
# Core dependencies
pip install torch torchvision transformers pillow tqdm

# Evaluation metrics
pip install nltk rouge-score

# Optional: W&B and HF Hub
pip install wandb huggingface_hub
```

## Training

### Basic Training

```bash
# Train on Flickr30k
python -m src.train --data-dir data/flickr30k --batch-size 16 --num-epochs 10

# Train on COCO
python -m src.train --data-dir data/coco_captions --batch-size 16 --num-epochs 10

# Train with evaluation metrics disabled
python -m src.train --data-dir data/flickr30k --no-eval
```

### Advanced Training Options

```bash
# Train with W&B logging
python -m src.train \
  --data-dir data/flickr30k \
  --wandb \
  --wandb-project "my-captioning-experiment" \
  --wandb-run-name "clip-gpt2-v1"

# Train and save to Hugging Face Hub
python -m src.train \
  --data-dir data/flickr30k \
  --save-to-hf "your-username/image-captioning-model" \
  --hf-token "your-hf-token" \
  --hf-private

# Train with specific evaluation strategy
python -m src.train \
  --data-dir data/flickr30k \
  --eval-strategy meteor-centric
```

### Model Saving Strategy

The training script saves **two separate best models**:

1. **`best_model_loss.pth`**: Best model based on validation loss improvement
2. **`best_model_eval.pth`**: Best model based on evaluation metrics (recommended for production)

**Evaluation Strategies:**
- `meteor-centric` (default): Focus on METEOR score with tolerance for other metrics
- `weighted-composite`: Balanced optimization across multiple metrics
- `pareto`: Conservative approach ensuring no metric degradation
- `multi-criteria`: Primary metric focus with flexibility
- Single metric: `meteor`, `bleu`, `rouge`

### Weights & Biases Integration

When using `--wandb`, the training script provides comprehensive logging:

**📊 Metrics Tracking:**
- Training and validation loss
- Evaluation metrics (BLEU, METEOR, ROUGE)
- Best model metrics over time

**🖼️ Test Images:**
- Sample images with generated vs reference captions
- Individual metrics for each test sample
- Visual progress tracking across epochs

**📦 Model Checkpoints:**
- Automatic artifact logging for all saved models
- Metadata including save criteria and performance metrics

## Inference

### Basic Inference

```bash
# Use evaluation-based model (recommended)
python -m src.infer notebooks/my_image.jpg \
  --checkpoint checkpoints/best_model_eval.pth

# Use loss-based model
python -m src.infer notebooks/my_image.jpg \
  --checkpoint checkpoints/best_model_loss.pth
```

### Generation Parameters

```bash
# High temperature for more creative captions
python -m src.infer notebooks/my_image.jpg \
  --temperature 1.2 \
  --max-length 60

# Low temperature for more deterministic captions
python -m src.infer notebooks/my_image.jpg \
  --temperature 0.3 \
  --num-beams 3

# Debug mode for troubleshooting
python -m src.debug_infer notebooks/my_image.jpg \
  --temperature 1.0 \
  --max-length 50
```

**Parameter Guide:**
- `--temperature` (0.1-2.0): Controls randomness (higher = more creative)
- `--max-length` (10-100): Maximum caption length
- `--num-beams` (1-10): Beam search width (higher = better quality, slower)

### Programmatic Usage

```python
from src.train import ImageCaptioningModel, load_best_model
from PIL import Image

# Load model
model = ImageCaptioningModel()
model, metadata = load_best_model(model, "checkpoints", model_type="eval")

# Generate caption
image = Image.open("my_image.jpg")
caption = model.generate_caption(
    image, 
    temperature=0.8, 
    max_length=50, 
    num_beams=5
)
print(caption)
```

## Testing

### Test on Dataset Samples

```bash
# Test evaluation-based model (recommended)
python -m src.test_model \
  --checkpoint checkpoints/best_model_eval.pth \
  --data-dir data/flickr30k

# Test on more samples
python -m src.test_model \
  --checkpoint checkpoints/best_model_eval.pth \
  --num-samples 5

# Test on validation split
python -m src.test_model \
  --checkpoint checkpoints/best_model_eval.pth \
  --split val

# Test with custom generation parameters
python -m src.test_model \
  --checkpoint checkpoints/best_model_eval.pth \
  --temperature 1.1 \
  --max-length 60 \
  --num-beams 3
```

### Model Management

```bash
# Compare available models
python -m src.model_utils --action compare

# List all available models
python -m src.model_utils --action list

# Load and test a model
python -m src.model_utils \
  --action load \
  --model-type eval \
  --test-image notebooks/my_image.jpg
```

## Evaluation

### Evaluate Model Performance

```bash
# Evaluate evaluation-based model
python -m src.evaluate \
  --checkpoint checkpoints/best_model_eval.pth \
  --data-dir data/flickr30k

# Evaluate loss-based model
python -m src.evaluate \
  --checkpoint checkpoints/best_model_loss.pth \
  --data-dir data/flickr30k
```

### Evaluation Metrics

The model is evaluated using standard image captioning metrics:
- **BLEU-1/2/3/4**: Measures n-gram overlap with reference captions
- **METEOR**: Considers synonyms and word order
- **ROUGE-L**: Measures longest common subsequence

## Hugging Face Hub Integration

### Save During Training

```bash
python -m src.train \
  --data-dir data/flickr30k \
  --save-to-hf "your-username/image-captioning-model" \
  --hf-token "your-hf-token" \
  --hf-private
```

### Save After Training

```bash
python -m src.save_to_hf \
  --checkpoint checkpoints/best_model_eval.pth \
  --repo-name "your-username/image-captioning-model" \
  --token "your-hf-token" \
  --private
```

### Use Your Model

```python
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# Load your model
tokenizer = AutoTokenizer.from_pretrained("your-username/image-captioning-model")
model = AutoModel.from_pretrained("your-username/image-captioning-model")

# Generate caption
image = Image.open("image.jpg")
caption = model.generate_caption(image)
print(caption)
```

## Environment Variables

Configure the model using environment variables:

```bash
# Hugging Face Hub
export HUGGINGFACE_TOKEN="your-hf-token"
export HF_REPO_NAME="your-username/image-captioning-model"
export HF_PRIVATE="false"

# Weights & Biases
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_PROJECT="my-captioning-project"
export WANDB_RUN_NAME="experiment-1"

# Training
export DEVICE="cuda"
export DATA_DIR="data/flickr30k"
export BATCH_SIZE="16"
export NUM_EPOCHS="10"
```

## Dataset Preparation

### Available Datasets

```bash
# Flickr30k (default)
python scripts/download_flickr30k.py

# COCO
python scripts/download_coco_captions.py

# Flickr8k
python scripts/download_flickr8k.py
```

## Model Architecture

The model consists of:
- **CLIP Vision Encoder**: Extracts image features using CLIP's vision transformer
- **Cross-Attention**: GPT-2 decoder attends to image features during text generation
- **Projection Layer**: Maps CLIP features to GPT-2 hidden dimension (if needed)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Empty Captions**: Increase training epochs or adjust generation parameters
3. **Slow Training**: Use GPU acceleration or reduce model complexity
4. **Import Errors**: Install missing dependencies with pip

### Testing W&B Integration

```bash
# Test W&B logging functionality
python -m src.test_wandb_logging

# Set up W&B API key first
export WANDB_API_KEY="your-api-key"
# or
wandb login
```

### Getting Help

- Check the logs for error messages
- Verify your dataset format
- Ensure all dependencies are installed
- Try with a smaller dataset first

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .

# Lint code
uv run flake8 .

# Type check
uv run mypy .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 