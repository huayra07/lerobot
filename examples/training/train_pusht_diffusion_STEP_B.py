#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_pusht_diffusion_STEP_B.py
# ==============================================================================
"""
STEP B: Train a baseline (non-language) diffusion policy on PushT
with correct dataset metadata (ds_meta) so normalization stats are valid.

This version adds FULL resume support:
- Saves policy via policy.save_pretrained(...)
- Also saves training_state.pt containing:
  optimizer, scheduler, step, RNG states, config snapshot

Run:
  # Test with 10 steps:
  STEPS=10 BATCH_SIZE=16 python examples/training/train_pusht_diffusion_STEP_B.py

  # Train to 50k:
  STEPS=50000 SAVE_FREQ=5000 BATCH_SIZE=16 python examples/training/train_pusht_diffusion_STEP_B.py

  # Resume from a checkpoint "pretrained_model" directory:
  RESUME_FROM=outputs/stepB_diffusion_pusht/checkpoints/050000/pretrained_model \
    STEPS=100000 python examples/training/train_pusht_diffusion_STEP_B.py

Optional env vars:
  OUT_DIR=outputs/stepB_diffusion_pusht
  BATCH_SIZE=64
  LR=1e-4
  STEPS=10
  SAVE_FREQ=5000
  SEED=0
  RESUME_FROM=/path/to/.../pretrained_model
"""

import os
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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


def _try_load_policy_from_pretrained(
    pretrained_dir: Path,
    device: torch.device,
):
    """
    Try to load a policy directly from a local pretrained directory.
    Falls back to None if not possible.
    """
    # Most LeRobot versions provide policy.from_pretrained for local dirs,
    # but some older combinations may be flaky. We'll try and fall back.
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # type: ignore

        p = DiffusionPolicy.from_pretrained(str(pretrained_dir))
        p = p.to(device)
        return p
    except Exception:
        return None


def main() -> None:
    # -------------------------
    # Config
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(os.environ.get("OUT_DIR", "outputs/stepB_diffusion_pusht"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_steps = int(os.environ.get("STEPS", "10"))  # Default to 10 for easy testing!
    save_freq = int(os.environ.get("SAVE_FREQ", "5000"))
    seed = int(os.environ.get("SEED", "0"))
    resume_from = os.environ.get("RESUME_FROM", "").strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    print("=" * 80)
    print("STEP B — Baseline diffusion (NO language) on PushT")
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
    print()

    # -------------------------
    # Create env/policy config
    # -------------------------
    env_cfg = make_env_config("pusht", task="PushT-v0")
    cfg = make_policy_config("diffusion")

    # STEP B: explicitly disable language
    cfg.use_language_cond = False

    # -------------------------
    # Build dataset FIRST (for ds_meta stats)
    # -------------------------
    fps = getattr(env_cfg, "fps", 10)

    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon

    print("Loading dataset (video_backend=None) ...")
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
    print("✓ Dataset loaded")
    print(f"  len(dataset): {len(dataset)}")
    print(f"  fps: {fps} | n_obs_steps: {n_obs_steps} | horizon: {horizon}")
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
    # Create policy (optionally resume weights)
    # -------------------------
    policy = None
    start_step = 0

    if resume_from:
        resume_dir = Path(resume_from)
        state_path = resume_dir / "training_state.pt"

        # Try the clean way: load the full policy from the checkpoint folder
        policy = _try_load_policy_from_pretrained(resume_dir, device)

        if policy is None:
            print("WARNING: Could not load policy via DiffusionPolicy.from_pretrained. Falling back to fresh init.")
            policy = make_policy(cfg, ds_meta=ds_meta).to(device)
        else:
            print("✓ Loaded policy from pretrained checkpoint dir")

    if policy is None:
        print("Building diffusion policy with ds_meta (fixes normalization stats) ...")
        policy = make_policy(cfg, ds_meta=ds_meta).to(device)
        print("  ✓ Created with ds_meta (normalization stats are correct!)")

    policy.train()

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("✓ Policy ready")
    print(f"  total params: {total_params:,}")
    print(f"  trainable params: {trainable_params:,}")
    print(f"  use_language_cond: {getattr(policy.diffusion, 'use_language_cond', None)}")
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
        resume_dir = Path(resume_from)
        state_path = resume_dir / "training_state.pt"
        if state_path.exists():
            start_step = _load_training_state(state_path, optimizer, scheduler)
            print(f"✓ Resumed optimizer/scheduler/RNG from {state_path}")
            print(f"  start_step={start_step}")
            print()
        else:
            print(f"WARNING: {state_path} not found. Will resume weights only (optimizer/scheduler restart).")
            print()

    # Save training config snapshot (always)
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(
            {
                "step": "B",
                "policy": "diffusion",
                "use_language_cond": False,
                "batch_size": batch_size,
                "lr": lr,
                "steps": num_steps,
                "save_freq": save_freq,
                "seed": seed,
                "fps": fps,
                "n_obs_steps": n_obs_steps,
                "horizon": horizon,
                "resume_from": resume_from or None,
            },
            f,
            indent=2,
        )

    # -------------------------
    # Train loop
    # -------------------------
    print("=" * 80)
    print("Training ...")
    print("=" * 80)

    step = start_step
    running = 0.0
    debug_once = False

    # Fast-forward scheduler if we resumed weights only but no training_state.pt
    # (optional; leave as-is if you prefer restart schedule)
    # if resume_from and start_step > 0:
    #     for _ in range(start_step):
    #         scheduler.step()

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            batch = _to_device(batch, device)

            # Ensure action_is_pad exists
            if "action_is_pad" not in batch:
                batch["action_is_pad"] = torch.zeros(
                    batch["action"].shape[:2],
                    dtype=torch.bool,
                    device=device,
                )

            if not debug_once:
                print("First batch shapes:")
                for k in ["observation.state", "observation.image", "action", "action_is_pad"]:
                    if k in batch:
                        print(f"  {k}: {tuple(batch[k].shape)}")
                a = batch["action"]
                print(f"action min/max: {a.min().item():.4f} / {a.max().item():.4f}")
                print()
                debug_once = True

            optimizer.zero_grad(set_to_none=True)

            out = policy(batch)
            if isinstance(out, tuple):
                loss = out[0]
            elif isinstance(out, dict) and "loss" in out:
                loss = out["loss"]
            else:
                raise RuntimeError(f"Unexpected policy output type: {type(out)}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach().item())
            step += 1

            if step % 100 == 0 or step == num_steps:
                denom = 100 if step % 100 == 0 else (step % 100)
                denom = denom if denom != 0 else 100
                avg = running / denom
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}/{num_steps} | loss={avg:.4f} | lr={cur_lr:.2e}")
                running = 0.0

            if step % save_freq == 0 or step == num_steps:
                ckpt_dir = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # Save model
                policy.save_pretrained(str(ckpt_dir))

                # Save full training state
                _save_training_state(
                    ckpt_dir / "training_state.pt",
                    step=step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    seed=seed,
                    extra={
                        "step_label": "B",
                        "use_language_cond": False,
                        "batch_size": batch_size,
                        "lr": lr,
                        "fps": fps,
                        "n_obs_steps": n_obs_steps,
                        "horizon": horizon,
                    },
                )

                print(f"\n✓ Saved checkpoint: {ckpt_dir}")
                print(f"  ✓ Saved training state: {ckpt_dir / 'training_state.pt'}\n")

    print("\n" + "=" * 80)
    print("DONE — Step B training complete")
    print("=" * 80)
    final_dir = out_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()