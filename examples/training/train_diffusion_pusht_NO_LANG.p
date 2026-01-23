#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_diffusion_pusht_NO_LANG.py
# ==============================================================================
"""
Train a baseline diffusion policy WITHOUT language conditioning.
This will work on LeRobot 0.4.3 and verify your training setup works.

Usage:
    python examples/training/train_diffusion_pusht_NO_LANG.py

Environment variables:
    BATCH_SIZE - Training batch size (default: 64)
    LR - Learning rate (default: 1e-4)
    STEPS - Total training steps (default: 100000)
"""

import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training hyperparameters
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    learning_rate = float(os.environ.get("LR", "1e-4"))
    num_steps = int(os.environ.get("STEPS", "100000"))
    eval_freq = 5000
    save_freq = 5000
    
    output_dir = Path("outputs/baseline_diffusion_pusht")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Training Baseline Diffusion Policy (NO Language)")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {num_steps}")
    print(f"Output: {output_dir}\n")
    
    # Create policy config WITHOUT language
    policy_config = make_policy_config(
        "diffusion",
        use_language_cond=False,  # NO language conditioning
    )
    
    print("✓ Created policy config (baseline, no language)")
    print(f"  use_language_cond={policy_config.use_language_cond}\n")
    
    # Create environment config for policy
    env_config = make_env_config("pusht", task="PushT-v0")
    
    # Create policy
    policy = make_policy(policy_config, env_cfg=env_config)
    policy = policy.to(device)
    policy.train()
    
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"✓ Policy created")
    print(f"  Total parameters: {total_params:,}\n")
    
    # Load dataset
    print("Loading PushT dataset...")
    try:
        dataset = LeRobotDataset(
            "lerobot/pusht",
            video_backend=None,  # Use images, not videos
        )
    except Exception as e:
        print(f"Error loading with video_backend=None: {e}")
        print("Trying without video_backend parameter...")
        dataset = LeRobotDataset("lerobot/pusht")
    
    # Verify dataset
    sample = dataset[0]
    print(f"Dataset sample keys: {list(sample.keys())}")
    if "observation.image" in sample:
        print(f"  observation.image: {sample['observation.image'].shape}")
    if "observation.state" in sample:
        print(f"  observation.state: {sample['observation.state'].shape}")
    if "action" in sample:
        print(f"  action: {sample['action'].shape}")
    print()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=learning_rate * 0.1,
    )
    
    # Save config
    config_save_path = output_dir / "training_config.json"
    with open(config_save_path, "w") as f:
        json.dump({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_steps": num_steps,
            "use_language": False,
        }, f, indent=2)
    
    print("✓ Training setup complete\n")
    print("=" * 60)
    print("Starting training (NO language conditioning)...")
    print("=" * 60)
    
    # Training loop
    step = 0
    running_loss = 0.0
    
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass (no language needed)
            optimizer.zero_grad()
            
            try:
                output = policy(batch)
                
                # Extract loss
                if isinstance(output, dict) and "loss" in output:
                    loss = output["loss"]
                else:
                    print(f"Warning: policy output doesn't have 'loss' key")
                    print(f"Output type: {type(output)}")
                    if isinstance(output, dict):
                        print(f"Output keys: {output.keys()}")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                step += 1
                
            except Exception as e:
                print(f"Error in training step {step}: {e}")
                print(f"Batch keys: {batch.keys()}")
                raise
            
            # Logging
            if step % 100 == 0:
                avg_loss = running_loss / 100
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}/{num_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0
            
            # Save checkpoint
            if step % save_freq == 0:
                ckpt_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                
                # Save using LeRobot's save method
                try:
                    policy.save_pretrained(str(ckpt_dir))
                    print(f"✓ Saved checkpoint to {ckpt_dir}")
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
            
            # Evaluation reminder
            if step % eval_freq == 0 and step > 0:
                print(f"\n{'='*60}")
                print(f"📊 Checkpoint at step {step}")
                print(f"   This is a BASELINE (no language)")
                print(f"   Success rate should be similar to language models")
                print(f"{'='*60}\n")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    final_ckpt = output_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"Final checkpoint: {final_ckpt}")
    print("\n⚠️  NOTE: This baseline has NO language conditioning")
    print("   It's a sanity check that your training setup works.")
    print("   To get language conditioning, you need to upgrade LeRobot.")
    print("=" * 60)


if __name__ == "__main__":
    main()