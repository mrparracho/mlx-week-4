#!/bin/bash

# Simple setup script for image captioning model
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_cmd() {
    echo -e "${GREEN}📝 $1${NC}"
}

# Check GPU
check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
    elif [ "$OSTYPE" = "darwin"* ] && [ "$(uname -m)" = "arm64" ]; then
        echo "mps"
    else
        echo "cpu"
    fi
}

# Display GPU status
show_gpu_status() {
    print_status "Checking GPU availability..."
    local device=$1
    case $device in
        "cuda")
            print_success "NVIDIA GPU detected"
            ;;
        "mps")
            print_success "Apple Silicon detected (MPS available)"
            ;;
        "cpu")
            print_warning "No GPU detected, will use CPU"
            ;;
    esac
}

# Install uv if not present
install_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        print_status "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH - try multiple common locations
        if [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
            print_status "Added ~/.local/bin to PATH"
        elif [ -f "/root/.local/bin/uv" ]; then
            export PATH="/root/.local/bin:$PATH"
            print_status "Added /root/.local/bin to PATH"
        fi
        
        # Source shell profile to ensure PATH is updated
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc" 2>/dev/null || true
        fi
    fi
    print_success "uv is ready"
}

# Setup environment
setup_env() {
    print_status "Setting up environment..."
    uv sync
    print_success "Environment ready"
}

# Download dataset
download_dataset() {
    local dataset=${1:-flickr30k}
    local max_images=${2:-}
    
    print_status "Downloading $dataset dataset..."
    
    if [[ -n "$max_images" ]]; then
        print_status "Limiting to $max_images images"
        uv run python scripts/download_${dataset}.py --max-images "$max_images"
    else
        uv run python scripts/download_${dataset}.py
    fi
    
    print_success "Dataset ready"
}

# Main execution
main() {
    echo "🚀 Image Captioning Setup"
    echo "========================"
    
    # Parse arguments
    DATASET=${1:-flickr30k}
    MAX_IMAGES=${2:-}
    
    # Setup
    install_uv
    setup_env
    
    # Check GPU and display status
    DEVICE=$(check_gpu)
    show_gpu_status "$DEVICE"
    
    download_dataset "$DATASET" "$MAX_IMAGES"
    
    # Show training command
    echo ""
    print_success "Setup completed! You can now run training with:"
    echo ""
    print_cmd "uv run python src/train.py --data-dir data/$DATASET --device $DEVICE"
    echo ""
    echo "Additional options:"
    print_cmd "--batch-size 16          # Set batch size"
    print_cmd "--num-epochs 10          # Set number of epochs"
    print_cmd "--learning-rate 1e-4     # Set learning rate"
    print_cmd "--no-eval                # Disable evaluation metrics"
    print_cmd "--no-test                # Disable in-training testing"
    print_cmd "--save-to-hf username/model  # Save to Hugging Face Hub"
    echo ""
    echo "Example:"
    print_cmd "uv run python src/train.py --data-dir data/$DATASET --device $DEVICE --batch-size 32 --num-epochs 20"
}

# Show usage
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [dataset] [max_images]"
    echo ""
    echo "Examples:"
    echo "  $0                    # Setup with flickr30k dataset (all images)"
    echo "  $0 coco              # Setup with COCO dataset (all images)"
    echo "  $0 flickr30k 1000    # Setup with flickr30k, limit to 1000 images"
    echo "  $0 coco 500          # Setup with COCO, limit to 500 images"
    echo "  $0 flickr8k 200      # Setup with Flickr8k, limit to 200 images"
    echo ""
    echo "Available datasets: flickr30k, coco, flickr8k"
    echo "max_images: Optional limit on number of images to download"
    exit 0
fi

main "$@"
