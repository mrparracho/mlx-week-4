#!/usr/bin/env python3
"""
Wandb sweep agent for quick self-attention vs cross-attention comparison.
"""

import wandb
import subprocess
import sys
import os
from pathlib import Path

def train():
    """Main training function called by wandb sweep agent"""
    
    # Initialize wandb with token authentication
    wandb_token = os.getenv('WANDB_API_KEY')
    if not wandb_token:
        print("❌ WANDB_API_KEY environment variable not set!")
        print("Please set your wandb API key:")
        print("export WANDB_API_KEY=your_api_key_here")
        return
    
    # Initialize wandb with token
    wandb.login(key=wandb_token)
    wandb.init()
    
    # Get the sweep configuration
    config = wandb.config
    
    print(f"🚀 Starting sweep run with config: {config}")
    
    # Determine which model to use and select appropriate batch size
    if config.model == "CLIP-GPT2-SelfAttention-ImageCaptioning":
        script_name = "src/train_self_attn.py"
        self_attn_flag = "--self-attn"
        model_type = "self_attn"
        batch_size = config.batch_size_self
        gradient_accumulation_steps = config.gradient_accumulation_steps
        effective_batch_size = batch_size * gradient_accumulation_steps
    else:  # CLIP-GPT2-ImageCaptioning
        script_name = "src/train_self_attn.py"
        self_attn_flag = ""
        model_type = "cross_attn"
        batch_size = config.batch_size_cross
        gradient_accumulation_steps = 1  # Cross-attention doesn't need accumulation
        effective_batch_size = batch_size
    
    # Build the command
    cmd = [
        "python", script_name,
        "--data-dir", "data/flickr30k",  # Add the required data directory
        "--batch-size", str(batch_size),
        "--num-epochs", str(config.num_epochs),
        "--learning-rate", str(config.learning_rate),
        "--device", config.device,
        "--eval-strategy", config.eval_strategy,
        "--test-samples-per-epoch", str(config.test_samples_per_epoch),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--wandb",
    ]
    
    # Add self-attention flag if needed
    if self_attn_flag:
        cmd.append(self_attn_flag)
    
    # Create a descriptive run name
    run_name = f"{model_type}_lr{config.learning_rate:.2e}_bs{batch_size}_eff{effective_batch_size}"
    if gradient_accumulation_steps > 1:
        run_name += f"_acc{gradient_accumulation_steps}"
    
    # Set the run name
    wandb.run.name = run_name
    
    # Log the configuration to wandb
    wandb.log({
        "config/model_type": model_type,
        "config/batch_size": batch_size,
        "config/effective_batch_size": effective_batch_size,
        "config/gradient_accumulation_steps": gradient_accumulation_steps,
        "config/learning_rate": config.learning_rate,
        "config/num_epochs": config.num_epochs,
    })
    
    print(f"📊 Run name: {run_name}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🤖 Model: {model_type}")
    print(f"📈 Learning rate: {config.learning_rate}")
    print(f"📈 Batch size: {batch_size}")
    print(f"📈 Effective batch size: {effective_batch_size}")
    print(f"🔄 Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Run the training command
    try:
        # Add the project root to Python path
        env = os.environ.copy()
        project_root = str(Path(__file__).parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent,  # Run from project root
            env=env
        )
        
        print("✅ Training completed successfully!")
        print("📤 Output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        print(f"Error output:")
        print(e.stderr)
        
        # Log the error to wandb
        wandb.log({"error": e.stderr, "error_code": e.returncode})
        
        # Re-raise the exception to mark the run as failed
        raise e
    
    print("🎉 Sweep run completed!")

if __name__ == "__main__":
    train() 