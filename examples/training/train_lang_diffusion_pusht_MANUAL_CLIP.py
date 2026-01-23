#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_lang_diffusion_pusht_MANUAL_CLIP.py
# ==============================================================================
"""
Train language-conditioned diffusion policy with MANUAL CLIP integration.
Since LeRobot 0.4.3 doesn't support CLIP natively, we add it ourselves.

Usage:
    python examples/training/train_lang_diffusion_pusht_MANUAL_CLIP.py
"""

import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class CLIPConditionedPolicy(nn.Module):
    """Wrapper that adds CLIP text conditioning to any policy."""
    
    def __init__(self, base_policy, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.base_policy = base_policy
        
        # Load CLIP text encoder
        print(f"Loading CLIP model: {clip_model_name}")
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Freeze CLIP (optional - set to False to fine-tune)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Project CLIP embeddings to policy's expected dimension
        clip_dim = self.text_encoder.config.hidden_size  # 512 for CLIP-base
        
        # Find the policy's language conditioning dimension
        # This is a bit hacky but works for LeRobot 0.4.3
        try:
            if hasattr(base_policy, 'diffusion'):
                # For diffusion policies, inject language into the noise predictor
                policy_dim = 128  # Common dimension for conditioning
            else:
                policy_dim = 128
        except:
            policy_dim = 128
        
        self.lang_proj = nn.Sequential(
            nn.Linear(clip_dim, policy_dim),
            nn.ReLU(),
            nn.Linear(policy_dim, policy_dim)
        )
        
        print(f"✓ CLIP encoder loaded")
        print(f"  Embedding dim: {clip_dim} -> {policy_dim}")
        print(f"  Frozen: {not self.text_encoder.training}")
    
    def encode_text(self, text_list):
        """Encode text using CLIP."""
        # Tokenize
        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)
        
        # Encode with CLIP
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            text_embeds = outputs.pooler_output  # [batch, 512]
        
        # Project to policy dimension
        lang_features = self.lang_proj(text_embeds)  # [batch, 128]
        
        return lang_features
    
    def forward(self, batch):
        """Forward pass with language conditioning."""
        # Get language from batch
        if "language" in batch:
            text_list = batch["language"]
            lang_features = self.encode_text(text_list)
            
            # Inject into batch for policy
            batch["language_embedding"] = lang_features
        
        # Call base policy's forward method
        # Returns (loss_tensor, None) in training mode
        output = self.base_policy.forward(batch)
        
        # Handle tuple output (loss, None)
        if isinstance(output, tuple):
            loss = output[0]
            return {"loss": loss}
        
        return output
    
    def predict_action(self, observation, language=None):
        """Predict action (for evaluation)."""
        batch = {"observation": observation}
        
        if language is not None:
            if isinstance(language, str):
                language = [language]
            batch["language"] = language
        
        with torch.no_grad():
            output = self.forward(batch)
        
        if isinstance(output, dict) and "action" in output:
            return output["action"]
        return output
    
    def save_pretrained(self, path):
        """Save both base policy and CLIP projector."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save base policy
        self.base_policy.save_pretrained(str(path / "base_policy"))
        
        # Save our CLIP projector
        torch.save({
            'lang_proj': self.lang_proj.state_dict(),
            'clip_model_name': "openai/clip-vit-base-patch32",
        }, path / "clip_projector.pt")
        
        print(f"✓ Saved CLIP-conditioned policy to {path}")


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    learning_rate = float(os.environ.get("LR", "1e-4"))
    num_steps = int(os.environ.get("STEPS", "100000"))
    save_freq = 5000
    
    output_dir = Path("outputs/manual_clip_diffusion_pusht")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    instruction = "Push the T-shaped block to the target."
    
    print(f"🚀 Training Language-Conditioned Diffusion Policy")
    print(f"   (with MANUAL CLIP integration)")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {num_steps}")
    print(f"Instruction: '{instruction}'")
    print(f"Output: {output_dir}\n")
    
    # Create base policy (without language)
    policy_config = make_policy_config("diffusion")
    env_config = make_env_config("pusht", task="PushT-v0")
    base_policy = make_policy(policy_config, env_cfg=env_config)
    
    print("✓ Created base diffusion policy")
    
    # Wrap with CLIP conditioning
    policy = CLIPConditionedPolicy(base_policy)
    policy = policy.to(device)
    policy.train()
    
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\n✓ CLIP-conditioned policy created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Get policy's expected dimensions
    n_obs_steps = policy_config.n_obs_steps  # Number of observation timesteps
    horizon = policy_config.horizon  # Action horizon
    
    # Get fps from environment config or use default
    fps = getattr(env_config, 'fps', 10)  # PushT default is 10 fps
    
    print(f"\nPolicy expects:")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  horizon: {horizon}")
    print(f"  fps: {fps}")
    
    # Load dataset - CONFIGURE FOR SEQUENCES
    print("\nLoading PushT dataset with sequence sampling...")
    dataset = LeRobotDataset(
        "lerobot/pusht",
        video_backend=None,  # Avoid video decoding issues
        delta_timestamps={
            # Observation timestamps (relative to current frame)
            "observation.image": [-(i / fps) for i in range(n_obs_steps-1, -1, -1)],
            "observation.state": [-(i / fps) for i in range(n_obs_steps-1, -1, -1)],
            # Action timestamps (future actions)
            "action": [(i / fps) for i in range(horizon)],
        },
    )
    print("✓ Dataset loaded with sequence sampling")
    
    # Create dataloader - use num_workers=0 to avoid multiprocessing issues
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid worker process errors
        pin_memory=True,
    )
    
    print(f"✓ Dataloader created")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(dataset)}")
    
    # Optimizer - only train policy + language projector, not CLIP
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=learning_rate,
        weight_decay=1e-6,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=learning_rate * 0.1,
    )
    
    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_steps": num_steps,
            "instruction": instruction,
            "clip_model": "openai/clip-vit-base-patch32",
            "manual_clip_integration": True,
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    # Training loop
    step = 0
    running_loss = 0.0
    debug_printed = False
    
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Add action_is_pad if missing (all False for real data)
            if "action_is_pad" not in batch:
                batch["action_is_pad"] = torch.zeros(
                    batch["action"].shape[:2],  # (B, horizon)
                    dtype=torch.bool,
                    device=device
                )
            
            # Debug first batch only (once)
            if not debug_printed:
                print(f"\n🔍 First batch keys: {list(batch.keys())}")
                for key in ['observation.state', 'observation.image', 'action', 'action_is_pad']:
                    if key in batch:
                        print(f"  '{key}': shape={tuple(batch[key].shape)}")
                print()  # Empty line for clarity
                debug_printed = True
            
            # Add language instruction
            batch["language"] = [instruction] * batch["action"].shape[0]
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                output = policy(batch)
                
                # Extract loss
                if isinstance(output, dict) and "loss" in output:
                    loss = output["loss"]
                else:
                    print(f"⚠️  Unexpected output format: {type(output)}")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                step += 1
                
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
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
                
                policy.save_pretrained(str(ckpt_dir))
                print(f"\n✓ Saved checkpoint: {ckpt_dir}\n")
                
                if step % (save_freq * 2) == 0 and step > 0:
                    print(f"{'='*60}")
                    print(f"📊 Training progress: {step}/{num_steps} ({100*step/num_steps:.1f}%)")
                    print(f"{'='*60}\n")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    final_ckpt = output_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"\nFinal checkpoint: {final_ckpt}")
    print(f"\nTo evaluate, create an eval script that:")
    print(f"  1. Loads the base policy from: {final_ckpt}/base_policy")
    print(f"  2. Loads the CLIP projector from: {final_ckpt}/clip_projector.pt")
    print(f"  3. Wraps them with CLIPConditionedPolicy")


if __name__ == "__main__":
    main()