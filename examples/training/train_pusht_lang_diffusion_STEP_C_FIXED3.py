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
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
import copy, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy




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

    # Always recover the saved step
    step = int(payload.get("step", 0))

    # Best-effort: optimizer/scheduler may not match across scripts
    try:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        print(f"[RESUME] Loaded optimizer/scheduler state (step={step})")
    except Exception as e:
        print(f"[RESUME WARNING] Could not load optimizer/scheduler state: {e}")
        print(f"[RESUME WARNING] Continuing with RESET optimizer/scheduler at step={step}")

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


def load_matching_weights(dst: torch.nn.Module, src: torch.nn.Module):
    dst_sd = dst.state_dict()
    src_sd = src.state_dict()

    filtered = {}
    skipped = []

    for k, v in src_sd.items():
        if k in dst_sd and v.shape == dst_sd[k].shape:
            filtered[k] = v
        else:
            # keep a short log of what didn't match
            if k in dst_sd:
                skipped.append((k, tuple(v.shape), tuple(dst_sd[k].shape)))

    missing, unexpected = dst.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped

# ------------------------------------------------------------------------------
# Manual CLIP encoder (frozen)
# ------------------------------------------------------------------------------
def freeze_all_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_by_name_(m: nn.Module, keywords: list[str]) -> list[str]:
    """Unfreezes params whose name contains any keyword. Returns list of unfrozen param names."""
    unfrozen = []
    for name, p in m.named_parameters():
        lname = name.lower()
        if any(k in lname for k in keywords):
            p.requires_grad = True
            unfrozen.append(name)
    return unfrozen

def count_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


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
    
def split_cond_base_params(policy: nn.Module, keys: list[str]):
    """
    Split trainable params into:
      - cond_params: language/cond/FiLM/gate related
      - base_params: everything else
    Always excludes CLIP encoder params.
    """
    cond_params, base_params = [], []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()

        # Never train CLIP
        if "clip_encoder" in lname or "text_encoder" in lname:
            continue

        if any(k in lname for k in keys):
            cond_params.append(p)
        else:
            base_params.append(p)

    return cond_params, base_params



    

def apply_freeze_mode_(policy: HybridCLIPDiffusionPolicy, freeze_mode: str) -> None:
    if freeze_mode == "lang_only":
        print("FREEZE_MODE=lang_only -> freezing everything except language-conditioning adapter layers")
        freeze_all_(policy.base_policy)
        # freeze_all_(policy)

        keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed"]
        unfrozen = unfreeze_by_name_(policy.base_policy, keys)
        print(f"Unfroze {len(unfrozen)} param tensors. Example names:")
        for n in unfrozen[:30]:
            print("  ", n)
        # Extra sanity: did we actually hit FiLM condition MLPs?
            # In modeling_diffusion.py, the FiLM layer is named "cond_encoder" inside DiffusionConditionalResidualBlock1d.
        hit_film = [n for n in unfrozen if "cond_encoder" in n.lower()]
        print(f"[SANITY] unfrozen params containing 'cond_encoder': {len(hit_film)}")
        for n in hit_film[:10]:
            print("   film:", n)

        if len(unfrozen) == 0:
            raise RuntimeError("FREEZE_MODE=lang_only unfroze 0 params. Your keywords missed everything.")
        if len(hit_film) == 0:
            print("[WARNING] FREEZE_MODE=lang_only did not unfreeze any 'cond_encoder' params.")
            print("          This might mean your name keywords don't match the actual model naming.")
            print("          You may be freezing the entire network and training nothing useful.")
    elif freeze_mode == "none":
        print("FREEZE_MODE=none -> training full policy (no freezing)")
    else:
        raise ValueError(f"Unknown FREEZE_MODE={freeze_mode}. Use 'lang_only' or 'none'.")

    tot, tr = count_params(policy)
    print(f"[PARAMS] total={tot:,} trainable={tr:,}")
    if tr == 0:
        raise RuntimeError("FREEZE_MODE resulted in 0 trainable params.")



# ------------------------------------------------------------------------------
# Hybrid wrapper: inject language_embedding, then call base policy
# ------------------------------------------------------------------------------

# class HybridCLIPDiffusionPolicy(nn.Module):
#     def __init__(self, base_policy: nn.Module, clip_encoder: CLIPLanguageEncoder):
#         super().__init__()
#         self.base_policy = base_policy
#         self.clip_encoder = clip_encoder
#         self.config = getattr(base_policy, "config", None)


#         # Basic assertions for safety
#         assert bool(getattr(base_policy.diffusion, "use_language_cond", False)), "use_language_cond must be True"
#         assert int(getattr(base_policy.diffusion, "language_cond_dim", -1)) == 512, "language_cond_dim must be 512"

#     def forward(self, batch: dict) -> dict:
#         if "language" in batch:
#             texts = batch["language"]
#         elif "language_text" in batch:
#             texts = batch["language_text"]
#         else:
#             raise KeyError("Batch must contain 'language' or 'language_text'")

#         device = next(self.base_policy.parameters()).device
#         lang_embeddings = self.clip_encoder.encode(texts, device)  # (B, 512)
#         batch["language_embedding"] = lang_embeddings
#         return self.base_policy(batch)

#     def save_pretrained(self, path: str) -> None:
#         path = Path(path)
#         path.mkdir(parents=True, exist_ok=True)

#         # Save base policy weights/config
#         self.base_policy.save_pretrained(str(path / "base_policy"))

#         # Save CLIP metadata (weights are pretrained + frozen)
#         (path / "clip_info.json").write_text(json.dumps({
#             "model_name": self.clip_encoder.model_name,
#             "embedding_dim": self.clip_encoder.embedding_dim,
#             "frozen": True,
#         }, indent=2))

#     @classmethod
#     def from_pretrained(cls, path: str, device: torch.device):
#         """Load both components from a saved hybrid checkpoint folder."""
#         path = Path(path)

#         # Load base diffusion policy directly (avoid make_policy API differences)
#         from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

#         base_dir = path / "base_policy"
#         if not (base_dir / "config.json").exists():
#             # Sometimes people point RESUME_FROM at base_policy already
#             if (path / "config.json").exists():
#                 base_dir = path
#             else:
#                 raise FileNotFoundError(f"Could not find config.json in {base_dir} (or {path}).")

#         base_policy = DiffusionPolicy.from_pretrained(str(base_dir)).to(device)
#         base_policy.eval()

#         # Recreate CLIP encoder (weights are pretrained + frozen)
#         clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32")

#         # Wrap and return
#         policy = cls(base_policy, clip_encoder)
#         return policy.to(device)

class HybridCLIPDiffusionPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module, clip_encoder: CLIPLanguageEncoder):
        super().__init__()
        self.base_policy = base_policy
        self.clip_encoder = clip_encoder
        self.config = getattr(base_policy, "config", None)

        # Make sure CLIP lives on the same device in eval
        # (we’ll also move it in from_pretrained, but this is harmless)
        try:
            dev = next(self.base_policy.parameters()).device
            self.clip_encoder = self.clip_encoder.to(dev)
        except Exception:
            pass

        # These assertions are fine IF base_policy has .diffusion like your training build.
        # If you ever load a different policy class without .diffusion, this will crash.
        # So make them conditional.
        if hasattr(base_policy, "diffusion"):
            assert bool(getattr(base_policy.diffusion, "use_language_cond", False)), "use_language_cond must be True"
            assert int(getattr(base_policy.diffusion, "language_cond_dim", -1)) == 512, "language_cond_dim must be 512"
            # print("[RESUME] use_language_cond:", policy.base_policy.diffusion.use_language_cond)
            print("[RESUME] use_language_cond:", self.base_policy.diffusion.use_language_cond)


    def train(self, mode: bool = True):
        super().train(mode)
        self.base_policy.train(mode)
        self.clip_encoder.eval()  # always keep CLIP in eval
        return self

    def reset(self, *args, **kwargs):
        # lerobot_eval2 rollout calls this
        if hasattr(self.base_policy, "reset"):
            return self.base_policy.reset(*args, **kwargs)
        return None
        
    def eval(self):
        super().eval()
        self.base_policy.eval()
        return self

    def _get_texts(self, batch: dict):
        # Accept either key; otherwise fallback to config.language_text replicated to batch size
        if "language" in batch:
            texts = batch["language"]
        elif "language_text" in batch:
            texts = batch["language_text"]
        else:
            # infer batch size B from any tensor-like value
            B = None
            for v in batch.values():
                if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 0:
                    B = int(v.shape[0])
                    break
            if B is None:
                B = 1
            default_text = getattr(self.config, "language_text", "Push the T-shaped block to the target.")
            texts = [default_text] * B
        return texts

    def _inject(self, batch: dict):
        device = next(self.base_policy.parameters()).device
        texts = self._get_texts(batch)
        batch = dict(batch)  # don’t mutate caller dict
        batch["language_embedding"] = self.clip_encoder.encode(texts, device)  # (B, 512)
        return batch

    # def forward(self, batch: dict) -> dict:
    #     batch = self._inject(batch)
    #     # after batch["language_embedding"] = lang_embeddings
    #     if not hasattr(self, "_printed_once"):
    #         self._printed_once = True
    #         print("[HYBRID] injected language_embedding:",
    #             "language_embedding" in batch,
    #             tuple(batch["language_embedding"].shape),
    #             "dtype", batch["language_embedding"].dtype,
    #             "device", batch["language_embedding"].device)

    #     return self.base_policy(batch)
    def forward(self, batch: dict) -> dict:
        batch = self._inject(batch)

        if not hasattr(self, "_printed_forward_once"):
            self._printed_forward_once = True
            has = "language_embedding" in batch
            print("[HYBRID] injected language_embedding:", has)
            if has:
                t = batch["language_embedding"]
                print("  shape:", tuple(t.shape), "dtype:", t.dtype, "device:", t.device)
            else:
                print("  keys:", list(batch.keys())[:20])

        return self.base_policy(batch)

    # ---- common action method names that rollout might call ----
    @torch.no_grad()
    def select_action(self, batch, noise=None):
        # always work on a copy
        batch2 = dict(batch)

        # inject embedding into batch2
        batch2 = self._inject(batch2)   # or inline compute + batch2["language_embedding"]=...

        if not hasattr(self, "_printed_select_once"):
            self._printed_select_once = True
            print("[HYBRID] injected language_embedding:",
                "language_embedding" in batch2,
                (tuple(batch2["language_embedding"].shape) if "language_embedding" in batch2 else None))

        # IMPORTANT: call base_policy on batch2 (not batch)
        return self.base_policy.select_action(batch2, noise=noise)

    # def select_action(self, obs: dict, *args, **kwargs):
    #     if not hasattr(self, "_logged_once"):
    #         self._logged_once = True
    #         print("[HYBRID] select_action called. example text:", self._get_texts(obs)[0])
    #     if not hasattr(self, "_emb_logged_once"):
    #         self._emb_logged_once = True
    #         device = next(self.base_policy.parameters()).device
    #         e1 = self.clip_encoder.encode(["AAAAA AAAAA"], device)
    #         e2 = self.clip_encoder.encode(["BBBBB BBBBB"], device)
    #         print("[HYBRID] emb norms:", float(e1.norm()), float(e2.norm()))
    #         cos = torch.nn.functional.cosine_similarity(e1, e2).item()
    #         print("[HYBRID] emb cosine(AAAA,BBBB):", cos)
    #     # inside HybridCLIPDiffusionPolicy.select_action (or forward path right before calling base_policy)
    #     if not hasattr(self, "_delta_once"):
    #         self._delta_once = True
    #         # compute two embeddings for the SAME obs
    #         device = next(self.base_policy.parameters()).device
    #         e1 = self.clip_encoder.encode(["AAAAA AAAAA"], device)
    #         e2 = self.clip_encoder.encode(["BBBBB BBBBB"], device)
    #         print("[HYBRID] emb delta norm:", float((e1-e2).norm()))

        
    #     if not hasattr(self, "_action_delta_once"):
    #         self._action_delta_once = True
    #         device = next(self.base_policy.parameters()).device

    #         # helper: attach a language_embedding into obs/batch
    #         def run_with_text(text):
    #             o = copy.deepcopy(obs)
    #             emb = self.clip_encoder.encode([text], device)  # (1,512)

    #             # depending on what base_policy.select_action expects, try BOTH:
    #             o["language_embedding"] = emb
    #             o["language_text"] = [text]  # harmless if ignored
    #             return self.base_policy.select_action(o, **kwargs)

    #         a1 = run_with_text("AAAAA AAAAA")
    #         a2 = run_with_text("BBBBB BBBBB")

    #         # convert to tensor if needed
    #         t1 = torch.as_tensor(a1).float().flatten()
    #         t2 = torch.as_tensor(a2).float().flatten()
    #         print("[HYBRID] action A:", t1[:8].detach().cpu().tolist())
    #         print("[HYBRID] action B:", t2[:8].detach().cpu().tolist())
    #         print("[HYBRID] action delta norm:", float((t1 - t2).norm()))

    #     return self.base_policy.select_action(obs, **kwargs)

    #     # obs = self._inject(obs)
    #     # return self.base_policy.select_action(obs, *args, **kwargs)

    def act(self, obs: dict, *args, **kwargs):
        obs = self._inject(obs)
        if hasattr(self.base_policy, "act"):
            return self.base_policy.act(obs, *args, **kwargs)
        return self.base_policy.select_action(obs, *args, **kwargs)

    def predict_action(self, obs: dict, *args, **kwargs):
        obs = self._inject(obs)
        if hasattr(self.base_policy, "predict_action"):
            return self.base_policy.predict_action(obs, *args, **kwargs)
        return self.base_policy.select_action(obs, *args, **kwargs)

    def save_pretrained(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.base_policy.save_pretrained(str(path / "base_policy"))
        (path / "clip_info.json").write_text(json.dumps({
            "model_name": self.clip_encoder.model_name,
            "embedding_dim": self.clip_encoder.embedding_dim,
            "frozen": True,
        }, indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: torch.device):
        path = Path(path)
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        base_dir = path / "base_policy"
        if not (base_dir / "config.json").exists():
            if (path / "config.json").exists():
                base_dir = path
            else:
                raise FileNotFoundError(f"Could not find config.json in {base_dir} (or {path}).")

        base_policy = DiffusionPolicy.from_pretrained(str(base_dir)).to(device)
        base_policy.eval()

        clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32").to(device)
        policy = cls(base_policy, clip_encoder).to(device)
        policy.eval()
        return policy
    
def _resolve_policy_dir(p: str) -> str:
    """
    Accept either:
    - .../pretrained_model
    - .../pretrained_model/base_policy
    and return the folder that actually contains config.json.
    """
    pp = Path(p)
    if (pp / "config.json").exists():
        return str(pp)
    if (pp / "base_policy" / "config.json").exists():
        return str(pp / "base_policy")
    return str(pp)  # fallback (will error clearly if wrong)

def _resolve_hybrid_root(p: str) -> str:
    pp = Path(p)
    if (pp / "base_policy" / "config.json").exists():
        return str(pp)  # hybrid root
    if (pp / "config.json").exists():
        return str(pp.parent)  # if user passed .../base_policy
    return str(pp)


def hash_text_embedding(text: str, dim: int) -> torch.Tensor:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    emb = torch.randn(1, dim, generator=g, dtype=torch.float32)
    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
    return emb.squeeze(0)  # (dim,)

def make_collate_with_language(dim: int, text_key: str = "language"):
    def collate(batch_list):
        out = default_collate(batch_list)

        texts = []
        for ex in batch_list:
            t = ex.get(text_key, None) or ex.get("language_text", None) or ""
            texts.append(t)

        emb = torch.stack([hash_text_embedding(t, dim) for t in texts], dim=0)  # (B, dim)
        out["language_embedding"] = emb
        return out
    return collate

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
    init_from = os.environ.get("INIT_FROM", "").strip()
    freeze_mode = os.environ.get("FREEZE_MODE", "lang_only").strip()
    unfreeze_step = int(os.environ.get("UNFREEZE_STEP", "2000"))

    # FREEZE_MODE: "lang_only" (stage1) or "none" (stage2)


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
        resume_dir = _resolve_hybrid_root(resume_from)  # <-- add this
        policy = HybridCLIPDiffusionPolicy.from_pretrained(str(resume_dir), device)
        print("Policy class:", type(policy))

        # apply_freeze_mode_(policy, freeze_mode)

        print(f"✓ Loaded hybrid policy from {resume_dir}")
    # else:
    #     print("Creating base diffusion policy with ds_meta ...")
    #     base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)
    #     print("  ✓ Created with ds_meta (normalization stats are correct!)")
    #     print(f"  use_language_cond: {base_policy.diffusion.use_language_cond}")
    #     print(f"  language_cond_dim: {base_policy.diffusion.language_cond_dim}")
    #     print(f"  clip_text_encoder (expected None in 0.4.3): {getattr(base_policy.diffusion, 'clip_text_encoder', None)}")
    #     print()

    #     print("Loading CLIP encoder (frozen) ...")
    #     clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32").to(device)
    #     print(f"✓ CLIP loaded (embedding_dim={clip_encoder.embedding_dim})")
    #     print()

    #     policy = HybridCLIPDiffusionPolicy(base_policy, clip_encoder).to(device)
    # else:
    #     if init_from:
    #         print(f"Loading BASE init policy from: {init_from}")
    #         base_policy = DiffusionPolicy.from_pretrained(init_from).to(device)
    #         base_policy.train()
    #         print("  ✓ Loaded BASE weights")
    #     else:
    #         print("Creating base diffusion policy with ds_meta ...")
    #         base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)
    #         print("  ✓ Created with ds_meta (normalization stats are correct!)")
    else:
        if init_from:
            init_dir = _resolve_policy_dir(init_from)
            print(f"INIT_FROM provided. Loading source weights from: {init_dir}")
            src = DiffusionPolicy.from_pretrained(init_dir).to(device)

            src_has_lang = hasattr(src, "diffusion") and bool(getattr(src.diffusion, "use_language_cond", False))
            print(f"[INIT] source use_language_cond = {src_has_lang}")

            if src_has_lang:
                # Source already has language modules → use it directly
                base_policy = src
                base_policy.train()
                print("  ✓ Using source policy directly (already language-enabled)")
            # else:
            #     # Source is a BASE (no-language) policy → build StepC arch, then transplant weights
            #     print("  Source is BASE (no-language). Building StepC policy with ds_meta, then loading BASE weights (strict=False).")
            #     base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)  # StepC arch (language-enabled)
            #     missing, unexpected = base_policy.load_state_dict(src.state_dict(), strict=False)
            #     base_policy.train()
            #     print(f"  ✓ Loaded BASE weights into StepC (strict=False). missing={len(missing)} unexpected={len(unexpected)}")
            else:
                # Source is BASE (no-language) policy → build StepC arch, then transplant compatible weights
                print("  Source is BASE (no-language). Building StepC policy with ds_meta, then loading matching weights.")
                base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)  # StepC arch (language-enabled)

                missing, unexpected, skipped = load_matching_weights(base_policy, src)
                base_policy.train()

                print(f"  ✓ Loaded matching weights into StepC.")
                print(f"    skipped(shape mismatch): {len(skipped)}")
                print(f"    missing(after load): {len(missing)} unexpected: {len(unexpected)}")
                if skipped:
                    print("    example skipped:", skipped[:3])

        else:
            print("Creating base diffusion policy with ds_meta ...")
            base_policy = make_policy(cfg, ds_meta=ds_meta).to(device)
            base_policy.train()
            print("  ✓ Created with ds_meta (normalization stats are correct!)")


        print(f"  use_language_cond: {getattr(base_policy.diffusion, 'use_language_cond', None)}")
        print(f"  language_cond_dim: {getattr(base_policy.diffusion, 'language_cond_dim', None)}")
        print()

        print("Loading CLIP encoder (frozen) ...")
        clip_encoder = CLIPLanguageEncoder("openai/clip-vit-base-patch32").to(device)
        print(f"✓ CLIP loaded (embedding_dim={clip_encoder.embedding_dim})")
        print()

        policy = HybridCLIPDiffusionPolicy(base_policy, clip_encoder).to(device)
        # apply_freeze_mode_(policy, freeze_mode)
        # -------- STAGE FREEZING ----------
        # if freeze_mode == "lang_only":
        #     print("FREEZE_MODE=lang_only -> freezing everything except language-conditioning adapter layers")
        #     freeze_all_(policy.base_policy)

        #     # These keywords are the best generic “catch” for language-conditioning in diffusion policies
        #     keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed"]
        #     unfrozen = unfreeze_by_name_(policy.base_policy, keys)

        #     # Always keep anything explicitly in the wrapper trainable (CLIP is frozen anyway)
        #     # (CLIPTextModel params are already requires_grad=False in your encoder)
        #     print(f"Unfroze {len(unfrozen)} param tensors. Example names:")
        #     for n in unfrozen[:30]:
        #         print("  ", n)

        # elif freeze_mode == "none":
        #     print("FREEZE_MODE=none -> training full policy (no freezing)")
        #     # do nothing
        # else:
        #     raise ValueError(f"Unknown FREEZE_MODE={freeze_mode}. Use 'lang_only' or 'none'.")
        # # After freeze/unfreeze decisions, sanity check trainable params
        # tot, tr = count_params(policy)
        # print(f"[PARAMS] total={tot:,} trainable={tr:,}")
        # if tr == 0:
        #     raise RuntimeError("FREEZE_MODE resulted in 0 trainable params. Check unfreeze keywords / model names.")




    policy.train()
    apply_freeze_mode_(policy, freeze_mode)
    # >>> STAGE 1 GOES HERE (freeze + unfreeze) <<<
    # (either use apply_freeze_mode_ OR paste the explicit Stage1 block)

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
        # state_path = Path(resume_from) / "training_state.pt"
        resume_root = Path(_resolve_hybrid_root(resume_from))
        state_path = resume_root / "training_state.pt"
        # if freeze_mode == "lang_only":
        #     print("[RESUME] Re-applying FREEZE_MODE=lang_only")
        #     freeze_all_(policy.base_policy)
        #     keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed"]
        #     unfrozen = unfreeze_by_name_(policy.base_policy, keys)
        #     print(f"[RESUME] unfroze {len(unfrozen)} param tensors")
        # elif freeze_mode == "none":
        #     print("[RESUME] FREEZE_MODE=none (no freezing)")
        # else:
        #     raise ValueError(f"Unknown FREEZE_MODE={freeze_mode}")
            
        if state_path.exists():
            start_step = _load_training_state(state_path, optimizer, scheduler)
            print(f"✓ Resumed optimizer/scheduler/RNG from {state_path}")
            print(f"  start_step={start_step}")
            print()
        else:
            print(f"WARNING: {state_path} not found. Resuming weights only (optimizer/scheduler restart).")
            print()
        # -------------------------
        # STAGE RESUME LOGIC (IMPORTANT)
        # -------------------------
        # If we resumed past UNFREEZE_STEP, the "if step == unfreeze_step" block will never run.
        # So we must configure stage2 immediately.
        if start_step >= unfreeze_step:
            print("[RESUME] NOTE: start_step>=UNFREEZE_STEP so we rebuild optimizer/scheduler for STAGE2; optimizer state (Adam moments) is reset.")

            print(f"[RESUME] start_step={start_step} >= UNFREEZE_STEP={unfreeze_step} -> configuring STAGE2 now")
            
            # Unfreeze everything in base policy
            for p in policy.base_policy.parameters():
                p.requires_grad = True

            # Rebuild optimizer with param groups (cond vs base)
            keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed", "cond_alpha"]
            cond_params, base_params = [], []
            for name, p in policy.base_policy.named_parameters():
                lname = name.lower()
                if any(k in lname for k in keys):
                    cond_params.append(p)
                else:
                    base_params.append(p)

            print("[RESUME STAGE2] cond_params:", sum(p.numel() for p in cond_params),
                "base_params:", sum(p.numel() for p in base_params))

            optimizer = torch.optim.AdamW(
                [
                    {"params": cond_params, "lr": lr, "weight_decay": 1e-6},
                    {"params": base_params, "lr": lr * 0.1, "weight_decay": 1e-6},
                ]
            )

            # Scheduler for remaining steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, (num_steps - start_step)),
                eta_min=lr * 0.1,
            )

            tot, tr = count_params(policy)
            print(f"[RESUME STAGE2 PARAMS] total={tot:,} trainable={tr:,}")
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
    stage2_configured = (start_step >= unfreeze_step)

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

            if step % 500 == 0:
                with torch.no_grad():
                    emb = policy.clip_encoder.encode(batch["language"], device)
                    print(f"[LANG_EMB] step={step} mean={emb.mean().item():.4f} std={emb.std().item():.4f} norm={emb.norm(dim=-1).mean().item():.4f}")


            if not debug_once:
                print("\nFirst batch:")
                for k in ["observation.state", "observation.image", "action", "action_is_pad"]:
                    if k in batch:
                        print(f"  {k}: {tuple(batch[k].shape)}")
                print(f"  language[0]: {batch['language'][0]!r}")
                debug_once = True
                print()
            # -------------------------
            # DEBUG: does language affect action?
            # -------------------------
            if os.environ.get("DEBUG_LANG_EFFECT", "0") == "1" and step == start_step:
                policy.eval()
                with torch.no_grad():
                    # Build two batches identical except language text
                    # bA = dict(batch)
                    # bB = dict(batch)

                    # bA["language"] = ["AAAAA AAAAA"] * bsz
                    # bB["language"] = ["BBBBB BBBBB"] * bsz

                    obs_keys = [k for k in batch.keys() if k.startswith("observation.")]
                    obsA = {k: batch[k] for k in obs_keys}
                    obsB = {k: batch[k] for k in obs_keys}
                    # Fix image shape: (B, T, C, H, W) -> (B, C, H, W)
                    if "observation.image" in obsA and obsA["observation.image"].ndim == 5:
                        obsA["observation.image"] = obsA["observation.image"][:, -1]  # last frame
                        obsB["observation.image"] = obsB["observation.image"][:, -1]

                    # Fix state shape similarly if needed: (B, T, D) -> (B, D)
                    if "observation.state" in obsA and obsA["observation.state"].ndim == 3:
                        obsA["observation.state"] = obsA["observation.state"][:, -1]
                        obsB["observation.state"] = obsB["observation.state"][:, -1]
                    obsA["language"] = ["AAAAA AAAAA"] * bsz
                    obsB["language"] = ["BBBBB BBBBB"] * bsz

                    aA = policy.select_action(obsA)
                    aB = policy.select_action(obsB)


                    tA = torch.as_tensor(aA).float().to(device)
                    tB = torch.as_tensor(aB).float().to(device)
                    delta = (tA - tB).norm().item()

                    print(f"[DEBUG_LANG_EFFECT] ||action(A)-action(B)|| = {delta:.6f}")
                    print("  If ~0.0 => language is ignored")
                    print("  If huge / saturating => language dominates or scaling is off")

                policy.train()

            # -------------------------
            # STAGE2: switch optimizer exactly when step reaches UNFREEZE_STEP
            # (do this BEFORE optimizer.zero_grad/forward/backward)
            # -------------------------
            # -------------------------
            # STAGE 2: once step hits unfreeze_step, train full policy
            # - cond params get lr
            # - base params get lr*0.1
            # -------------------------
            if (not stage2_configured) and (step >= unfreeze_step):
                stage2_configured = True
                print(f"\n[STAGE2] switching at step={step} (unfreeze full policy)")

                # Unfreeze everything
                for p in policy.parameters():
                    p.requires_grad = True

                # But keep CLIP frozen
                for p in policy.clip_encoder.parameters():
                    p.requires_grad = False

                # Re-split params
                keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed", "cond_alpha"]

                cond_params, base_params = split_cond_base_params(policy, keys)

                print("[STAGE2] cond scalars:", sum(p.numel() for p in cond_params),
                    "base scalars:", sum(p.numel() for p in base_params))

                assert len(cond_params) > 0, "Stage2 cond_params empty"
                assert len(base_params) > 0, "Stage2 base_params empty (keywords too broad?)"

                optimizer = torch.optim.AdamW(
                    [
                        {"params": cond_params, "lr": lr, "weight_decay": 1e-6},
                        {"params": base_params, "lr": lr * 0.1, "weight_decay": 1e-6},
                    ]
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, (num_steps - step)),
                    eta_min=lr * 0.1,
                )

            # if (not stage2_configured) and (step >= unfreeze_step):
            #     stage2_configured = True
            #     print(f"\n[STAGE2] Unfreezing full policy at step={step}")

            #     # Unfreeze base policy params
            #     for p in policy.base_policy.parameters():
            #         p.requires_grad = True

            #     # Rebuild optimizer with param groups
            #     keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed", "cond_alpha"]
            #     # cond_params, base_params = [], []
            #     # for name, p in policy.base_policy.named_parameters():
            #     #     lname = name.lower()
            #     #     if any(k in lname for k in keys):
            #     #         cond_params.append(p)
            #     #     else:
            #     #         base_params.append(p)
            #     cond_params, base_params = [], []
            #     for name, p in policy.named_parameters():
            #         if not p.requires_grad:
            #             continue
            #         lname = name.lower()

            #         # skip CLIP encoder params
            #         if "clip_encoder" in lname or "text_encoder" in lname:
            #             continue

            #         if any(k in lname for k in keys):
            #             cond_params.append(p)
            #         else:
            #             base_params.append(p)

            #     print("[STAGE2] cond_params:", sum(p.numel() for p in cond_params),
            #         "base_params:", sum(p.numel() for p in base_params))

            #     optimizer = torch.optim.AdamW(
            #         [
            #             {"params": cond_params, "lr": lr, "weight_decay": 1e-6},
            #             {"params": base_params, "lr": lr * 0.1, "weight_decay": 1e-6},
            #         ]
            #     )

            #     # Rebuild scheduler to match remaining steps
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer,
            #         T_max=max(1, (num_steps - step)),
            #         eta_min=lr * 0.1,
            #     )

            #     tot, tr = count_params(policy)
            #     print(f"[STAGE2 PARAMS] total={tot:,} trainable={tr:,}\n")

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

            
            # if (not stage2_configured) and (step >= unfreeze_step):
            #     stage2_configured = True
            #     print(f"\n[STAGE2] Unfreezing full policy at step={step}")

            #     # Unfreeze base policy params
            #     for p in policy.base_policy.parameters():
            #         p.requires_grad = True

            #     # Rebuild optimizer with param groups
            #     keys = ["language", "lang", "cond", "film", "proj", "project", "adapter", "embed", "cond_alpha"]

            #     cond_params, base_params = [], []
            #     for name, p in policy.base_policy.named_parameters():
            #         lname = name.lower()
            #         if any(k in lname for k in keys):
            #             cond_params.append(p)
            #         else:
            #             base_params.append(p)
            #     print("[STAGE2] cond_params:", sum(p.numel() for p in cond_params),
            #     "base_params:", sum(p.numel() for p in base_params))

            #     optimizer = torch.optim.AdamW(
            #         [
            #             {"params": cond_params, "lr": lr, "weight_decay": 1e-6},
            #             {"params": base_params, "lr": lr * 0.1, "weight_decay": 1e-6},
            #         ]
            #     )

            #     # Rebuild scheduler so it matches the remaining steps
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer,
            #         T_max=(num_steps - step),
            #         eta_min=lr * 0.1,
            #     )

            #     tot, tr = count_params(policy)
            #     print(f"[STAGE2 PARAMS] total={tot:,} trainable={tr:,}\n")

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
