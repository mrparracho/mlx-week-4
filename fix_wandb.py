#!/usr/bin/env python3
"""
Script to fix W&B logging issues in train.py
"""

import re

def fix_wandb_logging():
    """Fix the W&B logging functions in train.py"""
    
    # Read the current train.py file
    with open('src/train.py', 'r') as f:
        content = f.read()
    
    # Fix the log_to_wandb function to add debugging
    log_to_wandb_pattern = r'def log_to_wandb\(metrics: dict, step: int \| None = None, prefix: str = ""\):\s*""".*?"""\s*if not WANDB_AVAILABLE or not wandb\.run:\s*return\s*try:\s*# Add prefix to metric names if provided\s*if prefix:\s*metrics = \{f"{prefix}/{k}": v for k, v in metrics\.items\(\)\}\s*if step is not None:\s*wandb\.log\(metrics, step=step\)\s*else:\s*wandb\.log\(metrics\)\s*except Exception as e:\s*print\(f"Warning: Failed to log to W&B: {e}"\)'
    
    log_to_wandb_fixed = '''def log_to_wandb(metrics: dict, step: int | None = None, prefix: str = ""):
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
        traceback.print_exc()'''
    
    # Fix the log_test_images_to_wandb function
    log_test_images_pattern = r'def log_test_images_to_wandb\(test_samples: list, epoch: int, step: int \| None = None\):\s*""".*?"""\s*if not WANDB_AVAILABLE or not wandb\.run:\s*return\s*try:\s*# Create a table for the test samples\s*test_data = \[\]\s*for i, sample in enumerate\(test_samples\):\s*# Convert PIL image to wandb image\s*wandb_image = wandb\.Image\(\s*sample\[\'image\'\],\s*caption=f"Generated: \{sample\[\'generated_caption\'\]\}\\nReference: \{sample\[\'reference_caption\'\]\}"\s*\)\s*# Create row data\s*row_data = \{\s*"sample_id": i \+ 1,\s*"image": wandb_image,\s*"reference_caption": sample\[\'reference_caption\'\],\s*"generated_caption": sample\[\'generated_caption\'\],\s*\}\s*# Add metrics if available\s*if sample\[\'metrics\'\]:\s*for metric_name, score in sample\[\'metrics\'\]\.items\(\):\s*row_data\[f"metric_\{metric_name\.lower\(\)\}"\] = f"\{score:\.4f\}"\s*test_data\.append\(row_data\)\s*# Log as a table\s*if step is not None:\s*wandb\.log\(\{\s*f"test_samples_epoch_\{epoch\}": wandb\.Table\(\s*dataframe=test_data,\s*columns=\["sample_id", "image", "reference_caption", "generated_caption"\] \+ \s*\[f"metric_\{k\.lower\(\)\}" for k in \(test_samples\[0\]\[\'metrics\'\]\.keys\(\) if test_samples\[0\]\[\'metrics\'\] else \[\)\]\s*\)\s*\}, step=step\)\s*else:\s*wandb\.log\(\{\s*f"test_samples_epoch_\{epoch\}": wandb\.Table\(\s*dataframe=test_data,\s*columns=\["sample_id", "image", "reference_caption", "generated_caption"\] \+ \s*\[f"metric_\{k\.lower\(\)\}" for k in \(test_samples\[0\]\[\'metrics\'\]\.keys\(\) if test_samples\[0\]\[\'metrics\'\] else \[\)\]\s*\)\s*\}\)\s*print\(f"📸 Logged \{len\(test_samples\)\} test samples to W&B for epoch \{epoch\}"\)\s*except Exception as e:\s*print\(f"Warning: Failed to log test images to W&B: \{e\}"\)'''
    
    log_test_images_fixed = '''def log_test_images_to_wandb(test_samples: list, epoch: int, step: int | None = None):
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
        import pandas as pd
        
        # Create a table for the test samples
        test_data = []
        
        for i, sample in enumerate(test_samples):
            # Convert PIL image to wandb image
            wandb_image = wandb.Image(
                sample['image'],
                caption=f"Generated: {sample['generated_caption']}\\nReference: {sample['reference_caption']}"
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
        traceback.print_exc()'''
    
    # Apply the fixes
    content = re.sub(log_to_wandb_pattern, log_to_wandb_fixed, content, flags=re.DOTALL)
    content = re.sub(log_test_images_pattern, log_test_images_fixed, content, flags=re.DOTALL)
    
    # Write the fixed content back
    with open('src/train.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed W&B logging functions in train.py")

if __name__ == "__main__":
    fix_wandb_logging()
