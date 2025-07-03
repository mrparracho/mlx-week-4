#!/usr/bin/env python3
"""
Run individual experiments manually if sweep fails.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(model_type, batch_size, learning_rate, gradient_accumulation_steps=1):
    """Run a single experiment"""
    
    cmd = [
        "python", "src/train_self_attn.py",
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--num-epochs", "5",
        "--wandb",
        "--eval",
    ]
    
    if model_type == "self_attn":
        cmd.append("--self-attn")
        cmd.extend(["--gradient-accumulation-steps", str(gradient_accumulation_steps)])
    
    run_name = f"{model_type}_lr{learning_rate:.2e}_bs{batch_size}"
    if gradient_accumulation_steps > 1:
        run_name += f"_acc{gradient_accumulation_steps}"
    
    print(f"🚀 Running: {run_name}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"✅ Completed: {run_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {run_name} - Error code: {e.returncode}")
        return False

def run_all_experiments():
    """Run all 16 experiments manually"""
    
    experiments = [
        # Cross-attention experiments
        ("cross_attn", 32, 1e-4),
        ("cross_attn", 64, 1e-4),
        ("cross_attn", 32, 5e-4),
        ("cross_attn", 64, 5e-4),
        
        # Self-attention experiments
        ("self_attn", 8, 1e-4, 1),
        ("self_attn", 8, 1e-4, 4),
        ("self_attn", 16, 1e-4, 1),
        ("self_attn", 16, 1e-4, 4),
        ("self_attn", 8, 5e-4, 1),
        ("self_attn", 8, 5e-4, 4),
        ("self_attn", 16, 5e-4, 1),
        ("self_attn", 16, 5e-4, 4),
    ]
    
    print("🎯 Running 12 experiments manually (equivalent to sweep)")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n📊 Experiment {i}/{len(experiments)}")
        
        if len(exp) == 3:
            model_type, batch_size, lr = exp
            success = run_experiment(model_type, batch_size, lr)
        else:
            model_type, batch_size, lr, acc_steps = exp
            success = run_experiment(model_type, batch_size, lr, acc_steps)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n🎉 Summary: {successful} successful, {failed} failed")
    print("📊 Check wandb.ai for results!")

if __name__ == "__main__":
    run_all_experiments() 