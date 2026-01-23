#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py
# ==============================================================================
"""
STEP C FIXED: Hybrid approach - Manual CLIP wrapper + use_language_cond=True

Since LeRobot 0.4.3's built-in CLIP doesn't instantiate (clip_text_encoder=None),
we manually add CLIP BUT configure the base policy with use_language_cond=True
and language_cond_dim=512 to match CLIP's output.

This ensures:
  ✓ Real CLIP embeddings (BPE tokenizer + pretrained encoder)
  ✓ Base policy actually uses language conditioning
  ✓ Proper 512-dim embeddings (not 128-dim learned vectors)

Run:
  # Test with 10 steps (takes ~1 minute):
  STEPS=10 python examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py
  
  # Full training (24-36 hours):
  STEPS=100000 python examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py

Optional env vars:
  OUT_DIR=outputs/stepC_hybrid_clip_diffusion_pusht
  BATCH_SIZE=64  (use 16 for testing)
  LR=1e-4
  STEPS=10  (or 100000 for full training)
  SAVE_FREQ=5000
"""

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CLIPLanguageEncoder(nn.Module):
    """
    Wraps CLIP text encoder to produce 512-dim embeddings.
    This is what LeRobot's built-in CLIP *should* do but doesn't in 0.4.3.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Freeze CLIP (only train the policy)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.text_encoder.eval()
        self.embedding_dim = 512  # CLIP ViT-B/32 output dim
    
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            texts: List of instruction strings
            device: Target device
        
        Returns:
            embeddings: (B, 512) CLIP text embeddings
        """
        with torch.no_grad():
            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)
            
            outputs = self.text_encoder(**tokens)
            # Use pooled output (CLS token representation)
            embeddings = outputs.pooler_output  # (B, 512)
        
        return embeddings


class HybridCLIPDiffusionPolicy(nn.Module):
    """
    Wraps a DiffusionPolicy configured with use_language_cond=True and
    language_cond_dim=512, then manually provides CLIP embeddings.
    """
    def __init__(self, base_policy, clip_encoder: CLIPLanguageEncoder):
        super().__init__()
        self.base_policy = base_policy
        self.clip_encoder = clip_encoder
        
        # Verify base policy is configured correctly
        assert base_policy.diffusion.use_language_cond, "Base policy must have use_language_cond=True"
        assert base_policy.diffusion.language_cond_dim == 512, "Base policy must have language_cond_dim=512"
    
    def forward(self, batch: dict) -> dict:
        """
        Extract language from batch, encode with CLIP, add to batch,
        then forward to base policy.
        """
        # Get language instructions
        if "language" in batch:
            texts = batch["language"]
        elif "language_text" in batch:
            texts = batch["language_text"]
        else:
            raise KeyError("Batch must contain 'language' or 'language_text'")
        
        device = next(self.base_policy.parameters()).device
        
        # Encode with CLIP to get (B, 512) embeddings
        lang_embeddings = self.clip_encoder.encode(texts, device)
        
        # Add to batch - the base policy will use this
        batch["language_embedding"] = lang_embeddings
        
        # Forward through base policy
        return self.base_policy(batch)
    
    def save_pretrained(self, path: str) -> None:
        """Save both components"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save base policy
        self.base_policy.save_pretrained(str(path / "base_policy"))
        
        # Save CLIP encoder info (we don't need to save weights - they're frozen pretrained)
        with open(path / "clip_info.json", "w") as f:
            json.dump({
                "model_name": "openai/clip-vit-base-patch32",
                "embedding_dim": 512,
            }, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str, device: torch.device):
        """Load both components"""
        path = Path(path)
        
        # Load base policy
        from lerobot.policies.factory import make_policy
        base_policy = make_policy(pretrained_policy_path=str(path / "base_policy"))
        
        # Recreate CLIP encoder
        clip_encoder = CLIPLanguageEncoder()
        
        # Wrap and return
        policy = cls(base_policy, clip_encoder)
        return policy.to(device)


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
def _save_training_state(
    state_path: Path,
    *,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "seed": seed,
        # RNG states (best-effort exact resume)
        "py_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    torch.save(payload, str(state_path))


def _load_training_state(
    state_path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> int:
    payload = torch.load(str(state_path), map_location="cpu", weights_only=False)

    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])

    # Restore RNG (best-effort)
    if "py_rng_state" in payload:
        random.setstate(payload["py_rng_state"])
    if "np_rng_state" in payload:
        np.random.set_state(payload["np_rng_state"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        try:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        except Exception:
            pass

    return int(payload.get("step", 0))



def main() -> None:
    # -------------------------
    # Config
    # -------------------------
    resume_from = os.environ.get("RESUME_FROM", "").strip()
    policy = None
    start_step = 0

    if resume_from:
        resume_dir = Path(resume_from)
        # resume_from points to .../pretrained_model
        policy = HybridCLIPDiffusionPolicy.from_pretrained(str(resume_dir), device)
        print(f"✓ Loaded hybrid policy from {resume_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(os.environ.get("OUT_DIR", "outputs/stepC_hybrid_clip_diffusion_pusht"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_steps = int(os.environ.get("STEPS", "10"))  # Default to 10 for easy testing!
    save_freq = int(os.environ.get("SAVE_FREQ", "5000"))
    seed = int(os.environ.get("SEED", "0"))
    instruction = os.environ.get("INSTRUCTION", "Push the T-shaped block to the target.")

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    print("=" * 80)
    print("STEP C FIXED — Hybrid CLIP + Language-Conditioned Diffusion")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"OUT_DIR: {out_dir}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"LR: {lr}")
    print(f"STEPS: {num_steps}")
    print(f"SAVE_FREQ: {save_freq}")
    print(f"SEED: {seed}")
    print(f"INSTRUCTION: {instruction!r}")
    print()

    # -------------------------
    # Create policy config with CORRECT settings
    # -------------------------
    env_cfg = make_env_config("pusht", task="PushT-v0")
    cfg = make_policy_config("diffusion")
    
    # CRITICAL: Configure for language conditioning
    cfg.use_language_cond = True
    cfg.language_cond_dim = 512  # Match CLIP output!
    
    # These don't actually work in 0.4.3, but set them anyway for completeness
    cfg.language_embedding_source = "clip"
    cfg.text_encoder_name = "openai/clip-vit-base-patch32"
    cfg.freeze_text_encoder = True

    # -------------------------
    # Build dataset FIRST (for ds_meta stats)
    # -------------------------
    fps = getattr(env_cfg, "fps", 10)
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon

    print("Loading dataset ...")
    dataset = LeRobotDataset(
        "lerobot/pusht",
        video_backend=None,
        delta_timestamps={
            "observation.image": [-(i / fps) for i in range(n_obs_steps - 1, -1, -1)],
            "observation.state": [-(i / fps) for i in range(n_obs_steps - 1, -1, -1)],
            "action": [(i / fps) for i in range(horizon)],
        },
    )
    ds_meta = dataset.meta
    print(f"✓ Dataset loaded (len={len(dataset)})")
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------
    # Create base policy WITH ds_meta ONLY (fixes normalization stats)
    # -------------------------
    print("Creating base diffusion policy with ds_meta ...")
    # CRITICAL: In LeRobot 0.4.3, make_policy accepts EITHER ds_meta OR env_cfg, not both!
    base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)
    print("  ✓ Created with ds_meta (normalization stats are correct!)")
    
    print("✓ Base policy created")
    print(f"  use_language_cond: {base_policy.diffusion.use_language_cond}")
    print(f"  language_cond_dim: {base_policy.diffusion.language_cond_dim}")
    print(f"  clip_text_encoder: {base_policy.diffusion.clip_text_encoder}")
    print()

    # -------------------------
    # Add manual CLIP encoder
    # -------------------------
    print("Loading CLIP encoder ...")
    clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32")
    print(f"✓ CLIP loaded (embedding_dim={clip_encoder.embedding_dim})")
    print()

    # -------------------------
    # Wrap in hybrid policy
    # -------------------------
    print("Creating hybrid CLIP policy ...")
    policy = HybridCLIPDiffusionPolicy(base_policy, clip_encoder).to(device)
    policy.train()
    
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    clip_params = sum(p.numel() for p in clip_encoder.parameters())
    
    print("✓ Hybrid policy created")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  CLIP params (frozen): {clip_params:,}")
    print()

    # -------------------------
    # Optimizer / scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=lr * 0.1,
    )

    # Save config
    with open(out_dir / "training_config.json", "w") as f:
        json.dump({
            "approach": "hybrid_manual_clip",
            "use_language_cond": True,
            "language_cond_dim": 512,
            "clip_model": "openai/clip-vit-base-patch32",
            "instruction": instruction,
            "batch_size": batch_size,
            "lr": lr,
            "steps": num_steps,
            "save_freq": save_freq,
            "seed": seed,
        }, f, indent=2)
    if resume_from:
        state_path = Path(resume_from) / "training_state.pt"
        if state_path.exists():
            start_step = _load_training_state(state_path, optimizer, scheduler)
            print(f"✓ Resumed optimizer/scheduler/RNG from {state_path}")
            print(f"  start_step={start_step}")
        else:
            print(f"WARNING: {state_path} not found. Resuming weights only (optimizer/scheduler restart).")


    # -------------------------
    # Train loop
    # -------------------------
    print("=" * 80)
    print("Training ...")
    print("=" * 80)

    step = start_step
    running = 0.0
    debug_once = False

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            batch = _to_device(batch, device)

            # Add action_is_pad if missing
            if "action_is_pad" not in batch:
                batch["action_is_pad"] = torch.zeros(
                    batch["action"].shape[:2],
                    dtype=torch.bool,
                    device=device,
                )

            # Add language instruction
            bsz = int(batch["action"].shape[0])
            batch["language"] = [instruction] * bsz

            if not debug_once:
                print("\nFirst batch:")
                for k in ["observation.state", "observation.image", "action", "action_is_pad"]:
                    if k in batch:
                        print(f"  {k}: {tuple(batch[k].shape)}")
                print(f"  language: {batch['language'][0]!r}")
                print()
                debug_once = True

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            out = policy(batch)
            
            # Extract loss
            if isinstance(out, tuple):
                loss = out[0]
            elif isinstance(out, dict) and "loss" in out:
                loss = out["loss"]
            else:
                raise RuntimeError(f"Unexpected output type: {type(out)}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach().item())
            step += 1

            if step % 100 == 0 or step == num_steps:
                avg = running / min(100, step)
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}/{num_steps} | loss={avg:.4f} | lr={cur_lr:.2e}")
                running = 0.0

            if step % save_freq == 0 or step == num_steps:
                ckpt_dir = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                policy.save_pretrained(str(ckpt_dir))

                _save_training_state(                       # <-- ADD THIS BLOCK
                    ckpt_dir / "training_state.pt",
                    step=step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    seed=seed,
                    extra={
                        "step_label": "C",
                        "use_language_cond": True,
                        "language_cond_dim": 512,
                        "clip_model": "openai/clip-vit-base-patch32",
                        "batch_size": batch_size,
                        "lr": lr,
                    },
                )

                print(f"\n✓ Saved checkpoint: {ckpt_dir}")
                print(f"  ✓ Saved training state: {ckpt_dir / 'training_state.pt'}\n")


    print("\n" + "=" * 80)
    print("DONE — Training complete!")
    print("=" * 80)
    final_dir = out_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"Final checkpoint: {final_dir}")
    print(f"\nThis policy uses:")
    print(f"  ✓ Real CLIP embeddings (512-dim, BPE tokenizer)")
    print(f"  ✓ use_language_cond=True in base policy")
    print(f"  ✓ Proper language conditioning throughout")


if __name__ == "__main__":
    main()