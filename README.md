# MLX Week 4 Project

This project uses `uv` as the package manager for fast and reliable Python dependency management.

## Setup

### Recommended (one line):

```bash
./setup.sh            # Flickr30k (default)
./setup.sh coco       # COCO dataset
./setup.sh flickr8k   # Flickr8k dataset
./setup.sh coco 500   # COCO, limit to 500 images
```

This installs dependencies, detects your GPU, downloads the dataset, and shows the training command.

---

*Manual setup:*
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python scripts/download_flickr30k.py
```

## Development

- Install development dependencies: `uv sync --dev`
- Run tests: `uv run pytest`
- Format code: `uv run black .`
- Lint code: `uv run flake8 .`
- Type check: `uv run mypy .`

## Project Structure

- `notebooks/` - Jupyter notebooks for MLX experiments
- `pyproject.toml` - Project configuration and dependencies 

# Image Captioning with CLIP + GPT-2

This project implements an image captioning model using CLIP as the vision encoder and GPT-2 as the text decoder with cross-attention.

## Features

- **CLIP Vision Encoder**: Uses OpenAI's CLIP model for image understanding
- **GPT-2 Decoder**: Generates captions with cross-attention to image features
- **Multiple Datasets**: Support for Flickr30k, COCO, and Flickr8k
- **Evaluation Metrics**: BLEU, METEOR, ROUGE, and CIDEr scores
- **Hugging Face Integration**: Save and share models on Hugging Face Hub
- **Cross-platform**: Supports CUDA, MPS (Apple Silicon), and CPU

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlx-week-4
```

2. Install dependencies:
```bash
pip install torch torchvision transformers pillow tqdm
pip install nltk rouge-score  # For evaluation metrics
pip install huggingface_hub   # For saving to Hugging Face Hub
```

## Usage

### Training

Train the model on your dataset:

```bash
# Train on Flickr30k
python src/train.py --data-dir data/flickr30k --batch-size 16 --num-epochs 10

# Train on COCO
python src/train.py --data-dir data/coco_captions --batch-size 16 --num-epochs 10

# Train with evaluation metrics disabled
python src/train.py --data-dir data/flickr30k --no-eval

# Train and save to Hugging Face Hub
python src/train.py --data-dir data/flickr30k --save-to-hf "your-username/image-captioning-model"

# Train with specific evaluation strategy
python src/train.py --data-dir data/flickr30k --eval-strategy weighted-composite

# Train with W&B logging (includes test images and model checkpoints)
python src/train.py --data-dir data/flickr30k --wandb --wandb-project "my-captioning-experiment"
```

#### Model Saving Strategy

The training script now saves **two separate best models**:

1. **`best_model_loss.pth`**: Best model based on validation loss improvement
2. **`best_model_eval.pth`**: Best model based on evaluation metrics (BLEU, METEOR, etc.)

This allows you to choose the most appropriate model for your use case:
- Use `best_model_eval.pth` for production (optimized for caption quality)
- Use `best_model_loss.pth` for debugging (lowest training loss)
- Compare both models to understand the trade-offs

Available evaluation strategies:
- `meteor-centric` (default): Focus on METEOR score with tolerance for other metrics
- `weighted-composite`: Balanced optimization across multiple metrics
- `pareto`: Conservative approach ensuring no metric degradation
- `multi-criteria`: Primary metric focus with flexibility
- Single metric: `meteor`, `bleu`, `rouge`

See [EVALUATION_STRATEGIES.md](EVALUATION_STRATEGIES.md) for detailed explanations.

#### Weights & Biases Integration

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
- Easy model versioning and comparison

**📈 Tables and Visualizations:**
- Interactive tables showing test samples with captions
- Model performance comparisons
- Training progress dashboards

Example W&B dashboard will show:
- **Charts**: Loss curves, metric trends
- **Tables**: Test samples with images and captions
- **Artifacts**: Model checkpoints with metadata
- **Media**: Generated captions with reference comparisons

### Inference

Generate captions for images:

```bash
# Use evaluation-based model (recommended for production)
python src/infer.py --checkpoint checkpoints/best_model_eval.pth --image path/to/image.jpg

# Use loss-based model
python src/infer.py --checkpoint checkpoints/best_model_loss.pth --image path/to/image.jpg

# Legacy: use old best_model.pth if it exists
python src/infer.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

#### Loading Models Programmatically

You can also load models programmatically using the new helper functions:

```python
from src.train import ImageCaptioningModel, load_best_model, list_available_models

# List available models
models_info = list_available_models("checkpoints")
print("Available models:", models_info)

# Load evaluation-based model (recommended)
model = ImageCaptioningModel()
model, metadata = load_best_model(model, "checkpoints", model_type="eval")

# Load loss-based model
model = ImageCaptioningModel()
model, metadata = load_best_model(model, "checkpoints", model_type="loss")

# Auto-select best available model
model = ImageCaptioningModel()
model, metadata = load_best_model(model, "checkpoints", model_type="auto")
```

### Evaluation

Evaluate model performance:

```bash
# Evaluate evaluation-based model
python src/evaluate.py --checkpoint checkpoints/best_model_eval.pth --data-dir data/flickr30k

# Evaluate loss-based model
python src/evaluate.py --checkpoint checkpoints/best_model_loss.pth --data-dir data/flickr30k

# Compare both models
python src/evaluate.py --checkpoint checkpoints/best_model_eval.pth --data-dir data/flickr30k
python src/evaluate.py --checkpoint checkpoints/best_model_loss.pth --data-dir data/flickr30k
```

### Testing

Test the model on random samples with visualization:

```bash
# Test evaluation-based model (recommended)
python src/test_model.py --checkpoint checkpoints/best_model_eval.pth --data-dir data/flickr30k

# Test loss-based model
python src/test_model.py --checkpoint checkpoints/best_model_loss.pth --data-dir data/flickr30k

# Test on more samples
python src/test_model.py --checkpoint checkpoints/best_model_eval.pth --num-samples 5

# Test on a different split (if available)
python src/test_model.py --checkpoint checkpoints/best_model_eval.pth --split val

# Don't save the plot
python src/test_model.py --checkpoint checkpoints/best_model_eval.pth --no-plot
```

### Model Management

Use the new model utilities to manage and compare your saved models:

```bash
# Compare available models
python src/model_utils.py --action compare

# List all available models
python src/model_utils.py --action list

# Load and test a model
python src/model_utils.py --action load --model-type eval --test-image path/to/image.jpg

# Load loss-based model
python src/model_utils.py --action load --model-type loss
```

## Saving to Hugging Face Hub

### During Training

You can automatically save your model to Hugging Face Hub after training:

```bash
python src/train.py \
    --data-dir data/flickr30k \
    --save-to-hf "your-username/image-captioning-model" \
    --hf-token "your-hf-token" \
    --hf-private  # Optional: make repository private
```

### After Training

Save an existing checkpoint to Hugging Face Hub:

```bash
python src/save_to_hf.py \
    --checkpoint checkpoints/best_model.pth \
    --repo-name "your-username/image-captioning-model" \
    --token "your-hf-token" \
    --commit-message "Add trained image captioning model" \
    --private  # Optional: make repository private
```

### Using Your Model

Once uploaded, others can use your model:

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

### Getting a Hugging Face Token

1. Go to [Hugging Face](https://huggingface.co/)
2. Create an account or sign in
3. Go to Settings → Access Tokens
4. Create a new token with "write" permissions
5. Use this token with the `--hf-token` argument

## Environment Variables

You can configure the model using environment variables for easier automation and CI/CD:

### Hugging Face Hub Variables
```bash
# Required for uploading models
export HUGGINGFACE_TOKEN="your-hf-token"

# Optional: Auto-save to HF Hub during training
export HF_REPO_NAME="your-username/image-captioning-model"

# Optional: Commit message for uploads
export HF_COMMIT_MESSAGE="Add trained image captioning model"

# Optional: Make repositories private (true/false)
export HF_PRIVATE="false"
```

### Training Variables
```bash
# Device selection
export DEVICE="cuda"  # or "mps", "cpu"

# Data directory
export DATA_DIR="data/flickr30k"

# Training parameters
export BATCH_SIZE="16"
export NUM_EPOCHS="10"
export LEARNING_RATE="1e-4"
```

### Example Usage with Environment Variables
```bash
# Set up environment
export HUGGINGFACE_TOKEN="your-token"
export HF_REPO_NAME="your-username/image-captioning-model"
export HF_PRIVATE="false"

# Train and auto-save to HF Hub
python src/train.py --data-dir data/flickr30k

# Save existing checkpoint
python src/save_to_hf.py --checkpoint checkpoints/best_model.pth --repo-name "your-username/model"
```

## Model Architecture

The model consists of:
- **CLIP Vision Encoder**: Extracts image features using CLIP's vision transformer
- **Cross-Attention**: GPT-2 decoder attends to image features during text generation
- **Projection Layer**: Maps CLIP features to GPT-2 hidden dimension (if needed)

## Evaluation Metrics

The model is evaluated using standard image captioning metrics:
- **BLEU-1/2/3/4**: Measures n-gram overlap with reference captions
- **METEOR**: Considers synonyms and word order
- **ROUGE-L**: Measures longest common subsequence
- **CIDEr**: TF-IDF weighted n-gram similarity

## Dataset Preparation

### Flickr30k
```bash
python scripts/download_flickr30k.py
```

### COCO
```bash
python scripts/download_coco_captions.py
```

### Flickr8k
```bash
python scripts/download_flickr8k.py
```

## Training Tips

1. **Start Small**: Begin with a small dataset to test your setup
2. **Monitor Metrics**: Use evaluation metrics to track progress
3. **Adjust Parameters**: Experiment with learning rate, batch size, and epochs
4. **Save Checkpoints**: Regular checkpoints help resume training
5. **Use GPU**: Training is much faster with CUDA or MPS

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Empty Captions**: Increase training epochs or adjust generation parameters
3. **Slow Training**: Use GPU acceleration or reduce model complexity
4. **Import Errors**: Install missing dependencies with pip

### Getting Help

- Check the logs for error messages
- Verify your dataset format
- Ensure all dependencies are installed
- Try with a smaller dataset first 

### Testing W&B Integration

Test the W&B logging functionality:

```bash
# Test W&B image and checkpoint logging (requires WANDB_API_KEY)
python src/test_wandb_logging.py

# Set up W&B API key first
export WANDB_API_KEY="your-api-key"
# or
wandb login
``` 