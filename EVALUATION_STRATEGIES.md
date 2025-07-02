# Evaluation Strategies for Model Saving

This document describes the different evaluation strategies available for determining when to save the best model during training.

## Overview

The training script supports multiple evaluation strategies to determine when a model checkpoint should be saved as the "best model". This allows you to optimize for different aspects of model performance based on your specific requirements.

## Available Strategies

### 1. **meteor-centric** (Default)
**Focus**: METEOR score with tolerance for other metrics

**Logic**:
- Saves if METEOR score improves directly
- Saves if METEOR decreases by < 2% but other metrics improve significantly:
  - BLEU-4 improvement > 1%
  - OR ROUGE-L improvement > 2%

**Best for**: When METEOR is your primary concern but you want some flexibility for other metrics.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy meteor-centric
```

### 2. **weighted-composite**
**Focus**: Weighted combination of multiple metrics

**Weights**:
- METEOR: 40% (Primary)
- BLEU-4: 20% (Secondary)
- ROUGE-L: 20% (Secondary)
- BLEU-1: 10% (Tertiary)
- BLEU-2: 10% (Tertiary)

**Logic**: Saves when the weighted composite score improves.

**Best for**: Balanced optimization across multiple metrics.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy weighted-composite
```

### 3. **pareto**
**Focus**: Pareto frontier improvement

**Logic**: Saves only when no metric gets worse AND at least one metric improves.

**Best for**: Conservative approach that ensures no degradation in any metric.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy pareto
```

### 4. **multi-criteria**
**Focus**: Primary metric with tolerance for other improvements

**Logic**:
- Saves if primary metric (METEOR) improves directly
- Saves if primary metric decreases by < 1% but at least 2 other metrics improve by > 2%

**Best for**: Primary metric focus with some flexibility for overall improvement.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy multi-criteria
```

### 5. **meteor** (Single Metric)
**Focus**: METEOR score only

**Logic**: Saves only when METEOR score improves.

**Best for**: Pure METEOR optimization.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy meteor
```

### 6. **bleu** (Single Metric)
**Focus**: BLEU-4 score only

**Logic**: Saves only when BLEU-4 score improves.

**Best for**: Pure BLEU-4 optimization.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy bleu
```

### 7. **rouge** (Single Metric)
**Focus**: ROUGE-L score only

**Logic**: Saves only when ROUGE-L score improves.

**Best for**: Pure ROUGE-L optimization.

```bash
uv run python src/train.py --data-dir data/flickr30k --eval-strategy rouge
```

## Usage Examples

### Basic Usage
```bash
# Use default meteor-centric strategy
uv run python src/train.py --data-dir data/flickr30k

# Explicitly specify meteor-centric
uv run python src/train.py --data-dir data/flickr30k --eval-strategy meteor-centric
```

### Advanced Usage
```bash
# Train with weighted composite evaluation
uv run python src/train.py \
    --data-dir data/flickr30k \
    --eval-strategy weighted-composite \
    --batch-size 32 \
    --num-epochs 20

# Train with Pareto frontier approach
uv run python src/train.py \
    --data-dir data/flickr30k \
    --eval-strategy pareto \
    --wandb \
    --wandb-project "pareto-optimization"
```

### Single Metric Optimization
```bash
# Optimize purely for METEOR
uv run python src/train.py --data-dir data/flickr30k --eval-strategy meteor

# Optimize purely for BLEU-4
uv run python src/train.py --data-dir data/flickr30k --eval-strategy bleu

# Optimize purely for ROUGE-L
uv run python src/train.py --data-dir data/flickr30k --eval-strategy rouge
```

## Strategy Comparison

| Strategy | Primary Focus | Flexibility | Use Case |
|----------|---------------|-------------|----------|
| `meteor-centric` | METEOR | High | General purpose, METEOR-focused |
| `weighted-composite` | Balanced | Medium | Multi-metric optimization |
| `pareto` | All metrics | Low | Conservative, no degradation |
| `multi-criteria` | METEOR | Medium | Primary + secondary improvements |
| `meteor` | METEOR only | None | Pure METEOR optimization |
| `bleu` | BLEU-4 only | None | Pure BLEU optimization |
| `rouge` | ROUGE-L only | None | Pure ROUGE optimization |

## Understanding the Metrics

### METEOR
- **What it measures**: Semantic similarity considering synonyms and word order
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Capturing meaning and semantic similarity

### BLEU-4
- **What it measures**: N-gram overlap with reference captions
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Grammatical correctness and word overlap

### ROUGE-L
- **What it measures**: Longest common subsequence
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Capturing sentence structure and flow

## Fallback Behavior

If evaluation metrics are not available (e.g., `--no-eval` flag is used), the system falls back to validation loss-based saving:

- Saves when validation loss decreases
- This ensures the model is always saved based on some criterion

## Integration with Weights & Biases

When using W&B logging (`--wandb`), the evaluation strategy is logged in the configuration, and all best metrics are tracked:

```python
# W&B config includes evaluation strategy
config = {
    "eval_strategy": "meteor-centric",
    # ... other config
}
```

## Recommendations

1. **Start with `meteor-centric`**: Good balance of METEOR focus with flexibility
2. **Use `weighted-composite`** for balanced optimization across all metrics
3. **Use `pareto`** if you want to ensure no metric degradation
4. **Use single metric strategies** for specific optimization goals
5. **Experiment with different strategies** to find what works best for your dataset

## Customization

To modify the evaluation strategies, edit the functions in `src/train.py`:

- `compute_composite_score()`: Adjust weights for weighted-composite
- `should_save_model_meteor_centric()`: Modify tolerance thresholds
- `should_save_model_multi_criteria()`: Change improvement thresholds
- `determine_model_save()`: Add new strategies