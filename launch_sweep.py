#!/usr/bin/env python3
"""
Launch the wandb sweep for quick model comparison.
"""

import wandb
import yaml
import os

def launch_sweep():
    """Launch the wandb sweep"""
    
    # Load the sweep configuration
    with open("sweep.yml", "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize wandb with token authentication
    # You can set the token in environment variable or pass it directly
    wandb_token = os.getenv('WANDB_API_KEY')
    if not wandb_token:
        print("❌ WANDB_API_KEY environment variable not set!")
        print("Please set your wandb API key:")
        print("export WANDB_API_KEY=your_api_key_here")
        print("Or add it to your .env file")
        return
    
    # Login with token
    try:
        wandb.login(key=wandb_token)
        print("✅ Wandb login successful")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # Use the same project name as the training scripts
    project_name = "image-captioning"
    entity_name = "mrparracho-mlx"
    
    print(f"🔍 Using project: {entity_name}/{project_name}")
    
    # Create the sweep directly without testing project access first
    try:
        sweep_id = wandb.sweep(
            sweep_config,
            project=project_name,
            entity=entity_name
        )
        
        print(f"🎯 Sweep created with ID: {sweep_id}")
        print(f"📊 Sweep URL: https://wandb.ai/{entity_name}/{project_name}/sweeps/{sweep_id}")
        print(f"🔢 Expected runs: 16 total")
        print(f"📋 Test combinations:")
        print(f"   - Models: Cross-attention, Self-attention")
        print(f"   - Learning rates: 1e-4, 5e-4")
        print(f"   - Cross-attention batch sizes: 32, 64")
        print(f"   - Self-attention batch sizes: 8, 16")
        print(f"   - Self-attention gradient accumulation: 1, 4")
        print(f"   - Epochs: 5 per run")
        print(f"   - Dynamic wandb logging: 20 logs per epoch regardless of batch size")
        
        # Launch the sweep agent
        wandb.agent(sweep_id, function=train_from_sweep_agent, count=None, project=project_name, entity=entity_name)
        
        print("✅ Sweep completed!")
        
    except Exception as e:
        print(f"❌ Failed to create sweep: {e}")
        print("\n🔧 Alternative approach: Run individual experiments manually")
        print("The sweep creation failed, but you can still run experiments manually:")
        print("1. Cross-attention: python src/train_self_attn.py --batch-size 32 --learning-rate 1e-4 --wandb")
        print("2. Self-attention: python src/train_self_attn.py --self-attn --batch-size 8 --gradient-accumulation-steps 4 --learning-rate 1e-4 --wandb")
        print("3. etc...")

def train_from_sweep_agent():
    """Import and run the train function from sweep_agent.py"""
    from sweep_agent import train
    train()

if __name__ == "__main__":
    launch_sweep() 