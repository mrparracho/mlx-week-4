# MLX Week 4 Project

This project uses `uv` as the package manager for fast and reliable Python dependency management.

## Setup

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
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
```

### Inference

Generate captions for images:

```bash
python src/infer.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

### Evaluation

Evaluate model performance:

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --data-dir data/flickr30k
```

### Testing

Test the model on random samples with visualization:

```bash
# Test on 3 random samples from train split
python src/test_model.py --checkpoint checkpoints/best_model.pth --data-dir data/flickr30k

# Test on more samples
python src/test_model.py --checkpoint checkpoints/best_model.pth --num-samples 5

# Test on a different split (if available)
python src/test_model.py --checkpoint checkpoints/best_model.pth --split val

# Don't save the plot
python src/test_model.py --checkpoint checkpoints/best_model.pth --no-plot
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