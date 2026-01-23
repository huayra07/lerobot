#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_lang_diffusion_pusht_clip.py
# ==============================================================================
"""
Train a PROPER language-conditioned diffusion policy on PushT with CLIP.
This uses a real text encoder (CLIP) instead of simple learned embeddings.

Usage:
    python examples/training/train_lang_diffusion_pusht_clip.py

Environment variables:
    BATCH_SIZE - Training batch size (default: 64)
    LR - Learning rate (default: 1e-4)
    STEPS - Total training steps (default: 100000)
    OUTPUT_DIR - Output directory (default: outputs/clip_lang_diffusion_pusht)
"""

import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# LeRobot imports - matching your eval script style
from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config

# Dataset import - use LeRobotDataset directly
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
    
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/clip_lang_diffusion_pusht"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training instruction
    instruction = "Push the T-shaped block to the target."
    
    print(f"🚀 Training CLIP Language-Conditioned Diffusion Policy")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {num_steps}")
    print(f"Instruction: '{instruction}'")
    print(f"Output: {output_dir}\n")
    
    # Create policy config using your existing factory
    policy_config = make_policy_config(
        "diffusion",
        
        # CRITICAL: Enable CLIP language conditioning
        use_language_cond=True,
        language_cond_dim=512,  # CLIP embedding dimension
        language_embedding_source="clip",  # Use CLIP encoder
        text_encoder_name="openai/clip-vit-base-patch32",  # Specific CLIP model
        freeze_text_encoder=True,  # Don't fine-tune CLIP (recommended)
        language_text=instruction,  # Save the instruction in config
    )
    
    print("✓ Created policy config with CLIP conditioning")
    print(f"  Text encoder: {policy_config.text_encoder_name}")
    print(f"  Language dim: {policy_config.language_cond_dim}")
    print(f"  Freeze encoder: {policy_config.freeze_text_encoder}\n")
    
    # Create environment config for policy
    env_config = make_env_config("pusht", task="PushT-v0")
    
    # Create policy using your existing factory
    policy = make_policy(policy_config, env_cfg=env_config)
    policy = policy.to(device)
    policy.train()
    
    # Verify CLIP was loaded
    state_dict = policy.state_dict()
    clip_params = [name for name in state_dict.keys() 
                   if "clip" in name.lower() or "text_encoder" in name.lower()]
    print(f"✓ Policy created with {len(clip_params)} CLIP parameters")
    
    if len(clip_params) > 0:
        print(f"  Sample CLIP params:")
        for name in clip_params[:5]:
            print(f"    {name}: {tuple(state_dict[name].shape)}")
    print()
    
    if len(clip_params) < 10:
        print("⚠️  WARNING: Expected 100+ CLIP parameters but found", len(clip_params))
        print("   CLIP may not be properly integrated!")
        print("   Check that your LeRobot version supports CLIP.")
        print("   You may need to update or modify the policy code.\n")
    
    # Load dataset - disable video loading to avoid codec errors
    print("Loading PushT dataset...")
    dataset = LeRobotDataset(
        "lerobot/pusht",
        video_backend=None,  # Disable video loading, use images directly
    )
    
    # Verify dataset structure
    sample = dataset[0]
    print(f"Dataset sample keys: {list(sample.keys())}")
    if "observation.image" in sample:
        print(f"Image shape: {sample['observation.image'].shape}")
    if "observation.state" in sample:
        print(f"State shape: {sample['observation.state'].shape}")
    if "action" in sample:
        print(f"Action shape: {sample['action'].shape}")
    print()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
            "instruction": instruction,
            "language_embedding_source": "clip",
            "text_encoder_name": "openai/clip-vit-base-patch32",
        }, f, indent=2)
    
    print("✓ Training setup complete\n")
    print("=" * 60)
    print("Starting training...")
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
            
            # Add language instruction to batch
            # CLIP expects a list of strings
            batch["language"] = [instruction] * len(batch["action"])
            
            # Forward pass
            optimizer.zero_grad()
            output = policy(batch)
            
            # Extract loss
            if isinstance(output, dict) and "loss" in output:
                loss = output["loss"]
            else:
                # If policy doesn't return loss dict, you may need to compute it
                raise ValueError("Policy did not return loss. Check your policy implementation.")
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            step += 1
            
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
                policy.save_pretrained(str(ckpt_dir))
                
                # Also save config
                config_path = ckpt_dir / "config.json"
                policy_config_dict = policy_config.__dict__ if hasattr(policy_config, '__dict__') else {}
                with open(config_path, "w") as f:
                    json.dump(policy_config_dict, f, indent=2)
                
                print(f"✓ Saved checkpoint to {ckpt_dir}")
            
            # Evaluation reminder
            if step % eval_freq == 0 and step > 0:
                print(f"\n{'='*60}")
                print(f"📊 Checkpoint at step {step} - run evaluation:")
                print(f"  CKPT_EVAL={ckpt_dir.parent} N_EVAL=20 \\")
                print(f"  python examples/training/eval_pusht_clip.py")
                print(f"{'='*60}\n")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    final_ckpt = output_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"Final checkpoint: {final_ckpt}")
    print("=" * 60)
    print(f"\n🎯 Run evaluation:")
    print(f"  CKPT_EVAL={final_ckpt.parent} N_EVAL=50 \\")
    print(f"  python examples/training/eval_pusht_clip.py")


if __name__ == "__main__":
    main()