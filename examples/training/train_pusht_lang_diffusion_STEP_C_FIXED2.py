#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py
# ==============================================================================
"""
STEP C (BASELINE, RESUMABLE): Hybrid Manual-CLIP + Language-Conditioned Diffusion

Why "hybrid":
- LeRobot 0.4.3 sometimes doesn't instantiate its internal CLIP text encoder
  (clip_text_encoder=None), so we run CLIP ourselves and inject language_embedding.

What this script does:
- Builds a DiffusionPolicy with use_language_cond=True, language_cond_dim=512
- Freezes CLIPTextModel (feature extractor only)
- During training, encodes text -> (B,512) and adds batch["language_embedding"]
- Trains diffusion policy normally on lerobot/pusht dataset
- Saves:
  - checkpoint pretrained_model/ (contains base_policy + clip_info.json)
  - training_state.pt for true resume (optimizer/scheduler/RNG/step)

Run:
  # quick test
  STEPS=10 BATCH_SIZE=16 python examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py

  # full
  STEPS=100000 SAVE_FREQ=5000 BATCH_SIZE=64 python examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py

  # resume (weights + optimizer/scheduler/RNG)
  RESUME_FROM=outputs/stepC_hybrid_clip_diffusion_pusht/checkpoints/050000/pretrained_model \
    STEPS=100000 python examples/training/train_pusht_lang_diffusion_STEP_C_FIXED.py

Optional env vars:
  OUT_DIR=outputs/stepC_hybrid_clip_diffusion_pusht
  BATCH_SIZE=64
  LR=1e-4
  STEPS=100000
  SAVE_FREQ=5000
  SEED=0
  RESUME_FROM=/path/to/.../pretrained_model
  INSTRUCTION="Push the T-shaped block to the target."
  PROMPTS="a;b;c"    # optional prompt list; if provided, randomly sample per batch
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


# ------------------------------------------------------------------------------
# RNG / Resume helpers
# ------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# ------------------------------------------------------------------------------
# Manual CLIP encoder (frozen)
# ------------------------------------------------------------------------------

class CLIPLanguageEncoder(nn.Module):
    """
    Wraps CLIP text encoder to produce 512-dim embeddings.
    Frozen during training (feature extractor).
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        self.embedding_dim = int(self.text_encoder.config.hidden_size)  # usually 512 for ViT-B/32

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        out = self.text_encoder(**tokens)
        return out.pooler_output  # (B, 512)


# ------------------------------------------------------------------------------
# Hybrid wrapper: inject language_embedding, then call base policy
# ------------------------------------------------------------------------------

class HybridCLIPDiffusionPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module, clip_encoder: CLIPLanguageEncoder):
        super().__init__()
        self.base_policy = base_policy
        self.clip_encoder = clip_encoder

        # Basic assertions for safety
        assert bool(getattr(base_policy.diffusion, "use_language_cond", False)), "use_language_cond must be True"
        assert int(getattr(base_policy.diffusion, "language_cond_dim", -1)) == 512, "language_cond_dim must be 512"

    def forward(self, batch: dict) -> dict:
        if "language" in batch:
            texts = batch["language"]
        elif "language_text" in batch:
            texts = batch["language_text"]
        else:
            raise KeyError("Batch must contain 'language' or 'language_text'")

        device = next(self.base_policy.parameters()).device
        lang_embeddings = self.clip_encoder.encode(texts, device)  # (B, 512)
        batch["language_embedding"] = lang_embeddings
        return self.base_policy(batch)

    def save_pretrained(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save base policy weights/config
        self.base_policy.save_pretrained(str(path / "base_policy"))

        # Save CLIP metadata (weights are pretrained + frozen)
        (path / "clip_info.json").write_text(json.dumps({
            "model_name": self.clip_encoder.model_name,
            "embedding_dim": self.clip_encoder.embedding_dim,
            "frozen": True,
        }, indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: torch.device):
        """Load both components from a saved hybrid checkpoint folder."""
        path = Path(path)

        # Load base diffusion policy directly (avoid make_policy API differences)
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        base_dir = path / "base_policy"
        if not (base_dir / "config.json").exists():
            # Sometimes people point RESUME_FROM at base_policy already
            if (path / "config.json").exists():
                base_dir = path
            else:
                raise FileNotFoundError(f"Could not find config.json in {base_dir} (or {path}).")

        base_policy = DiffusionPolicy.from_pretrained(str(base_dir)).to(device)
        base_policy.eval()

        # Recreate CLIP encoder (weights are pretrained + frozen)
        clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32")

        # Wrap and return
        policy = cls(base_policy, clip_encoder)
        return policy.to(device)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(os.environ.get("OUT_DIR", "outputs/stepC_hybrid_clip_diffusion_pusht"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_steps = int(os.environ.get("STEPS", "10"))
    save_freq = int(os.environ.get("SAVE_FREQ", "5000"))
    seed = int(os.environ.get("SEED", "0"))
    resume_from = os.environ.get("RESUME_FROM", "").strip()

    instruction = os.environ.get("INSTRUCTION", "Push the T-shaped block to the target.")
    prompts_env = os.environ.get("PROMPTS", "").strip()
    prompt_list = [p.strip() for p in prompts_env.split(";") if p.strip()] if prompts_env else []
    # If PROMPTS provided, we sample prompts per batch; otherwise use INSTRUCTION constantly.

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    print("=" * 80)
    print("STEP C — Hybrid Manual CLIP (frozen) + Language-Conditioned Diffusion (RESUMABLE)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"OUT_DIR: {out_dir}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"LR: {lr}")
    print(f"STEPS: {num_steps}")
    print(f"SAVE_FREQ: {save_freq}")
    print(f"SEED: {seed}")
    if resume_from:
        print(f"RESUME_FROM: {resume_from}")
    print(f"INSTRUCTION: {instruction!r}")
    if prompt_list:
        print(f"PROMPTS: {len(prompt_list)} prompts (sampling per batch)")
    print()

    # -------------------------
    # Build dataset FIRST (for ds_meta stats)
    # -------------------------
    env_cfg = make_env_config("pusht", task="PushT-v0")
    cfg = make_policy_config("diffusion")

    # Configure language conditioning
    cfg.use_language_cond = True
    cfg.language_cond_dim = 512
    cfg.language_embedding_source = "clip"
    cfg.text_encoder_name = "openai/clip-vit-base-patch32"
    cfg.freeze_text_encoder = True

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
    print(f"  fps={fps} n_obs_steps={n_obs_steps} horizon={horizon}")
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
    # Create policy (fresh or resume)
    # -------------------------
    policy: HybridCLIPDiffusionPolicy
    start_step = 0

    if resume_from:
        resume_dir = Path(resume_from)
        policy = HybridCLIPDiffusionPolicy.from_pretrained(str(resume_dir), device)
        print(f"✓ Loaded hybrid policy from {resume_dir}")
    else:
        print("Creating base diffusion policy with ds_meta ...")
        base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)
        print("  ✓ Created with ds_meta (normalization stats are correct!)")
        print(f"  use_language_cond: {base_policy.diffusion.use_language_cond}")
        print(f"  language_cond_dim: {base_policy.diffusion.language_cond_dim}")
        print(f"  clip_text_encoder (expected None in 0.4.3): {getattr(base_policy.diffusion, 'clip_text_encoder', None)}")
        print()

        print("Loading CLIP encoder (frozen) ...")
        clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32").to(device)
        print(f"✓ CLIP loaded (embedding_dim={clip_encoder.embedding_dim})")
        print()

        policy = HybridCLIPDiffusionPolicy(base_policy, clip_encoder).to(device)

    policy.train()

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("✓ Policy ready")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
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

    # Resume optimizer/scheduler/RNG if available
    if resume_from:
        state_path = Path(resume_from) / "training_state.pt"
        if state_path.exists():
            start_step = _load_training_state(state_path, optimizer, scheduler)
            print(f"✓ Resumed optimizer/scheduler/RNG from {state_path}")
            print(f"  start_step={start_step}")
            print()
        else:
            print(f"WARNING: {state_path} not found. Resuming weights only (optimizer/scheduler restart).")
            print()

    # Save a run-level config snapshot
    (out_dir / "training_config.json").write_text(json.dumps({
        "step_label": "C",
        "approach": "hybrid_manual_clip_frozen",
        "use_language_cond": True,
        "language_cond_dim": 512,
        "clip_model": "openai/clip-vit-base-patch32",
        "instruction": instruction,
        "prompts_count": len(prompt_list),
        "batch_size": batch_size,
        "lr": lr,
        "steps": num_steps,
        "save_freq": save_freq,
        "seed": seed,
        "fps": fps,
        "n_obs_steps": n_obs_steps,
        "horizon": horizon,
        "resume_from": resume_from or None,
    }, indent=2))

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

            if "action_is_pad" not in batch:
                batch["action_is_pad"] = torch.zeros(
                    batch["action"].shape[:2],
                    dtype=torch.bool,
                    device=device,
                )

            bsz = int(batch["action"].shape[0])

            # Language: constant instruction OR sampled prompts
            if prompt_list:
                chosen = [random.choice(prompt_list) for _ in range(bsz)]
                batch["language"] = chosen
            else:
                batch["language"] = [instruction] * bsz

            if not debug_once:
                print("\nFirst batch:")
                for k in ["observation.state", "observation.image", "action", "action_is_pad"]:
                    if k in batch:
                        print(f"  {k}: {tuple(batch[k].shape)}")
                print(f"  language[0]: {batch['language'][0]!r}")
                debug_once = True
                print()

            optimizer.zero_grad(set_to_none=True)

            out = policy(batch)
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
                denom = 100 if (step % 100 == 0) else max(1, step % 100)
                avg = running / denom
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}/{num_steps} | loss={avg:.4f} | lr={cur_lr:.2e}")
                running = 0.0

            if step % save_freq == 0 or step == num_steps:
                ckpt_dir = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # Save model (base_policy + clip_info.json)
                policy.save_pretrained(str(ckpt_dir))

                # Save full training state for true resume
                _save_training_state(
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
                        "fps": fps,
                        "n_obs_steps": n_obs_steps,
                        "horizon": horizon,
                        "prompts_count": len(prompt_list),
                    },
                )

                print(f"\n✓ Saved checkpoint: {ckpt_dir}")
                print(f"  ✓ Saved training state: {ckpt_dir / 'training_state.pt'}\n")

    print("\n" + "=" * 80)
    print("DONE — Step C training complete")
    print("=" * 80)
    final_dir = out_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"Final checkpoint: {final_dir}")
    print("This policy uses:")
    print("  ✓ Real CLIP embeddings (512-dim, BPE tokenizer, frozen encoder)")
    print("  ✓ use_language_cond=True in base policy")
    print("  ✓ language_embedding injected via hybrid wrapper")


if __name__ == "__main__":
    main()
