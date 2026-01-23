#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/train_pusht_lang_diffusion_STEP_C.py
# ==============================================================================
"""
STEP C: Train a language-conditioned diffusion policy on PushT using CLIP
(BPE tokenizer + pretrained text encoder), avoiding "param" fake language embeddings.

Key points:
  - Dataset first + ds_meta=ds.meta (fixes normalization stats)
  - use_language_cond=True at policy construction time
  - attempts to choose a non-param language_embedding_source supported by your install
  - provides language strings in the batch every step

Run:
  python examples/training/train_pusht_lang_diffusion_STEP_C.py

Optional env vars:
  OUT_DIR=outputs/stepC_lang_clip_diffusion_pusht
  BATCH_SIZE=64
  LR=1e-4
  STEPS=100000
  SAVE_FREQ=5000
  SEED=0
  INSTRUCTION="Push the T-shaped block to the target."
  TEXT_ENCODER_NAME="openai/clip-vit-base-patch32"
  FREEZE_TEXT_ENCODER=1   (1=true, 0=false)
"""

import os
import json
import random
import re
import inspect
from pathlib import Path

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


def _discover_embedding_source_options(policy) -> list[str]:
    """
    Tries to discover which strings your LeRobot build checks for
    in diffusion._prepare_global_conditioning (or related code).

    Returns a list of unique options found in source code, e.g. ["param", "clip", ...]
    """
    opts: set[str] = set()
    try:
        dm = policy.diffusion
        # Try likely functions that branch on language_embedding_source
        candidates = []
        for name in ["_prepare_global_conditioning", "forward", "conditional_sample"]:
            if hasattr(dm, name):
                candidates.append(getattr(dm, name))

        for fn in candidates:
            try:
                src = inspect.getsource(fn)
            except Exception:
                continue

            # Match patterns like: if self.language_embedding_source == "clip":
            for m in re.finditer(r"language_embedding_source\s*==\s*['\"]([^'\"]+)['\"]", src):
                opts.add(m.group(1))

            # Match patterns like: in ("param","clip")
            for m in re.finditer(r"language_embedding_source\s*in\s*\(([^)]+)\)", src):
                chunk = m.group(1)
                for s in re.finditer(r"['\"]([^'\"]+)['\"]", chunk):
                    opts.add(s.group(1))
    except Exception:
        pass

    return sorted(opts)


def _pick_non_param_source(found: list[str]) -> str | None:
    """
    Pick a likely non-param source from discovered options.
    """
    if not found:
        return None

    # Preference order: actual text-encoder driven sources first
    preferred = [
        "clip",
        "text_encoder",
        "pretrained",
        "hf",
        "huggingface",
        "tokenizer",
        "text",
        "language",
        "from_text",
        "from_language",
        "sentence",
    ]
    for p in preferred:
        for opt in found:
            if opt.lower() == p:
                return opt
        for opt in found:
            if p in opt.lower():
                return opt

    # If only param exists, return None
    for opt in found:
        if opt.lower() != "param":
            return opt
    return None


def main() -> None:
    # -------------------------
    # Config
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(os.environ.get("OUT_DIR", "outputs/stepC_lang_clip_diffusion_pusht"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_steps = int(os.environ.get("STEPS", "100000"))
    save_freq = int(os.environ.get("SAVE_FREQ", "5000"))
    seed = int(os.environ.get("SEED", "0"))

    instruction = os.environ.get("INSTRUCTION", "Push the T-shaped block to the target.")
    text_encoder_name = os.environ.get("TEXT_ENCODER_NAME", "openai/clip-vit-base-patch32")
    freeze_text_encoder = os.environ.get("FREEZE_TEXT_ENCODER", "1").strip() not in ["0", "false", "False"]

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    print("=" * 80)
    print("STEP C — Language-conditioned diffusion (CLIP / BPE) on PushT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"OUT_DIR: {out_dir}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"LR: {lr}")
    print(f"STEPS: {num_steps}")
    print(f"SAVE_FREQ: {save_freq}")
    print(f"SEED: {seed}")
    print(f"INSTRUCTION: {instruction!r}")
    print(f"TEXT_ENCODER_NAME: {text_encoder_name}")
    print(f"FREEZE_TEXT_ENCODER: {freeze_text_encoder}")
    print()

    # -------------------------
    # Create env/policy config
    # -------------------------
    env_cfg = make_env_config("pusht", task="PushT-v0")
    cfg = make_policy_config("diffusion")

    # IMPORTANT: must be True at construction time so shapes match.
    cfg.use_language_cond = True

    # Use real CLIP model
    cfg.text_encoder_name = text_encoder_name
    cfg.freeze_text_encoder = freeze_text_encoder

    # This field exists in your cfg dump; setting it helps some internal paths.
    cfg.language_text = instruction

    # We will *try* a non-param source. If your build rejects it, we'll auto-adjust.
    # Start with the most likely:
    cfg.language_embedding_source = "clip"

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
    # Create policy WITH ds_meta
    # -------------------------
    print("Building language-conditioned diffusion policy with ds_meta ...")
    
    # Debug: check what ds_meta looks like
    print(f"  ds_meta type: {type(ds_meta)}")
    print(f"  ds_meta has 'info': {hasattr(ds_meta, 'info') if hasattr(ds_meta, '__dict__') else 'N/A'}")
    
    try:
        # Try with ds_meta first
        policy = make_policy(cfg, ds_meta=ds_meta, env_cfg=env_cfg).to(device)
        print("  ✓ Created with ds_meta")
    except Exception as e1:
        print(f"\n  ❌ Failed with ds_meta: {e1}")
        print("  Trying without ds_meta (using env_cfg only)...")
        try:
            # Try with just env_cfg
            policy = make_policy(cfg, env_cfg=env_cfg).to(device)
            print("  ✓ Created with env_cfg only")
        except Exception as e2:
            print(f"  ❌ Failed with env_cfg: {e2}")
            print("  Trying with language_embedding_source='param' to at least create policy...")
            # Last resort: use param to create policy
            cfg.language_embedding_source = "param"
            try:
                policy = make_policy(cfg, env_cfg=env_cfg).to(device)
                print("  ✓ Created with param (will try to switch to clip later)")
            except Exception as e3:
                print(f"  ❌ All attempts failed: {e3}")
                raise RuntimeError("Could not create policy with any method") from e3

    policy.train()

    found_opts = _discover_embedding_source_options(policy)
    picked = _pick_non_param_source(found_opts)

    print("✓ Policy created")
    print(f"  policy type: {type(policy)}")
    print(f"  diffusion.use_language_cond: {getattr(policy.diffusion, 'use_language_cond', None)}")
    print(f"  diffusion.language_cond_dim: {getattr(policy.diffusion, 'language_cond_dim', None)}")
    print(f"  diffusion.clip_text_encoder: {getattr(policy.diffusion, 'clip_text_encoder', None)}")
    print(f"  discovered language_embedding_source options: {found_opts if found_opts else '<<could not infer>>'}")
    print()

    # If the constructed policy ended up with param, and we discovered a better option, switch.
    # (Many builds read this attribute at runtime in conditioning prep.)
    if picked is not None:
        print(f"Setting policy.diffusion.language_embedding_source -> {picked!r}")
        try:
            policy.diffusion.language_embedding_source = picked
        except Exception as e:
            print("⚠️ Could not set diffusion.language_embedding_source:", e)

    # If CLIP encoder is still None, print a loud warning: training may not be "real language".
    if getattr(policy.diffusion, "clip_text_encoder", None) is None:
        print("\n" + "!" * 80)
        print("WARNING: diffusion.clip_text_encoder is still None.")
        print("This usually means your build is NOT constructing the CLIP text encoder path,")
        print("or your language_embedding_source is still effectively 'param'.")
        print("Training will run, but it may NOT actually be language-conditioned in the way you want.")
        print("!" * 80 + "\n")

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  total params: {total_params:,}")
    print(f"  trainable params: {trainable_params:,}")
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

    # Save training config
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(
            {
                "step": "C",
                "policy": "diffusion",
                "use_language_cond": True,
                "instruction": instruction,
                "text_encoder_name": text_encoder_name,
                "freeze_text_encoder": freeze_text_encoder,
                "batch_size": batch_size,
                "lr": lr,
                "steps": num_steps,
                "save_freq": save_freq,
                "seed": seed,
                "fps": fps,
                "n_obs_steps": n_obs_steps,
                "horizon": horizon,
                "discovered_language_embedding_source_options": found_opts,
                "picked_language_embedding_source": picked,
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

    step = 0
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

            # Provide language in the batch using multiple common keys.
            # Different internal versions read different keys.
            bsz = int(batch["action"].shape[0])
            lang_list = [instruction] * bsz
            batch["language"] = lang_list
            batch["language_text"] = lang_list

            if not debug_once:
                print("First batch shapes:")
                for k in ["observation.state", "observation.image", "action", "action_is_pad"]:
                    if k in batch:
                        print(f"  {k}: {tuple(batch[k].shape)}")
                a = batch["action"]
                print(f"action min/max: {a.min().item():.4f} / {a.max().item():.4f}")
                print("language example:", batch["language"][0])
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

            if step % 100 == 0:
                avg = running / 100.0
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}/{num_steps} | loss={avg:.4f} | lr={cur_lr:.2e}")
                running = 0.0

            if step % save_freq == 0:
                ckpt_dir = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(str(ckpt_dir))
                print(f"\n✓ Saved checkpoint: {ckpt_dir}\n")

    print("\n" + "=" * 80)
    print("DONE — Step C training complete")
    print("=" * 80)
    final_dir = out_dir / "checkpoints" / f"{num_steps:06d}" / "pretrained_model"
    print(f"Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()