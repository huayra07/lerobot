#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/eval_pusht_STEP_B_or_C.py
# ==============================================================================
"""
Evaluate PushT diffusion checkpoints for:
- STEP B baseline diffusion (no language)
- STEP C FIXED hybrid CLIP language-conditioned diffusion

It will:
- Create PushT env
- Maintain an observation history window (n_obs_steps)
- Query the policy for an action each step
- Postprocess action into env action space
- Track success/reward/steps
- Save results JSON

Usage examples:

# Evaluate Step B checkpoint
CKPT_EVAL=outputs/stepB_diffusion_pusht/checkpoints/050000/pretrained_model \
  N_EVAL=20 MAX_STEPS=300 \
  python examples/training/eval_pusht_STEP_B_or_C.py

# Evaluate Step C (hybrid CLIP) checkpoint
CKPT_EVAL=outputs/stepC_hybrid_clip_diffusion_pusht/checkpoints/050000/pretrained_model \
  N_EVAL=20 MAX_STEPS=300 \
  INSTRUCTION="Push the T-shaped block to the target." \
  python examples/training/eval_pusht_STEP_B_or_C.py

Optional env vars:
  DEVICE=cuda|cpu
  SEED=0
  ACTION_MODE=neg1_1|0_1|raw        (default: neg1_1)
  HISTORY_MODE=last                 (default: last)  # currently only "last" supported
  N_OBS_STEPS=2                     (default: inferred from policy config if possible, else 2)
  SUCCESS_KEY=is_success|success    (default: auto-detect)
  SUCCESS_REWARD=...                (default: None; only used if no success key exists)
  OUT_JSON=...                      (default: outputs/<ckpt_parent>/eval_results_*.json)
"""

import os
import json
import time
import random
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# CLIP only needed for Step C hybrid (if base policy needs language_embedding)
from transformers import CLIPTokenizer, CLIPTextModel

# STATE_MODE = os.getenv("STATE_MODE", "raw")  # raw | 0_1 | neg1_1
# print("STATE_MODE:", STATE_MODE)
DEBUG = int(os.getenv("DEBUG", "0"))



# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def to_torch_obs_image(img: np.ndarray) -> torch.Tensor:
#     """
#     Convert env image to torch tensor with shape (3, H, W).
#     Resizes to 84x84 and normalizes to [0, 1].
#     """
#     if img is None:
#         raise ValueError("Observation image is None")
    
#     t = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
#     t = t / 255.0
#     # Resize to 84x84 if your model expects that
#     t = F.interpolate(t.unsqueeze(0), size=(96, 96), mode="bilinear", align_corners=False).squeeze(0)
#     return t.contiguous()
def to_torch_obs_image(img: np.ndarray) -> torch.Tensor:
    if img is None:
        raise ValueError("Observation image is None")
    if DEBUG:
        print("[to_torch_obs_image input]", type(img), img.dtype, img.shape)

    t = torch.from_numpy(img).permute(2, 0, 1).float()
    if DEBUG:
        print("[torch pre /255]", t.shape, t.dtype, t.min().item(), t.max().item())

    t = t / 255.0
    if DEBUG:
        print("[torch post /255]", t.shape, t.dtype, t.min().item(), t.max().item())

    if t.shape[1] == 96 and t.shape[2] == 96:
        t = t[:, 6:90, 6:90]
    if DEBUG:
        print("[torch post crop]", t.shape, t.dtype, t.min().item(), t.max().item())

    return t.contiguous()



def to_torch_obs_state(state: np.ndarray) -> torch.Tensor:
    """
    Convert env state to torch tensor with shape (state_dim,)
    """
    if state is None:
        raise ValueError("Observation state is None")
    return torch.from_numpy(np.asarray(state)).float().contiguous()


def infer_success(info: dict, success_key: str | None, reward: float, success_reward: float | None):
    """
    Returns: (success: bool, used: str)
    """
    if info is None:
        info = {}

    # explicit success key requested
    if success_key:
        if success_key in info:
            return bool(info[success_key]), f"info[{success_key}]"
        return False, f"missing info[{success_key}]"

    # auto-detect common keys
    for k in ["is_success", "success", "task_success"]:
        if k in info:
            return bool(info[k]), f"info[{k}]"

    # fallback to reward threshold if provided
    if success_reward is not None:
        return (reward >= success_reward), f"reward>={success_reward}"

    return False, "no_success_signal"


# def postprocess_action(action: np.ndarray, env, mode: str) -> np.ndarray:
#     """
#     Map model action to env action space.

#     mode:
#       - "neg1_1": action assumed in [-1, 1]
#       - "0_1": action assumed in [0, 1]
#       - "raw": no scaling, only clip to bounds if possible
#     """
#     a = np.asarray(action, dtype=np.float32)

#     # Flatten common shapes: (horizon, act_dim) -> take first action
#     if a.ndim == 2:
#         a = a[0]
#     if a.ndim != 1:
#         a = a.reshape(-1)

#     # If env has Box bounds, scale accordingly.
#     # low = getattr(getattr(env, "action_space", None), "low", None)
#     # high = getattr(getattr(env, "action_space", None), "high", None)

#     # if low is None or high is None:
#     #     # no bounds available -> return as-is
#     #     return a

#     # low = np.asarray(low, dtype=np.float32)
#     # high = np.asarray(high, dtype=np.float32)
#     low = np.array([96.0, 71.0], dtype=np.float32)
#     high = np.array([375.0, 449.0], dtype=np.float32)

#     if mode == "neg1_1":
#         a = (a + 1.0) * 0.5  # -> [0,1]
#         a = low + a * (high - low)
#     elif mode == "0_1":
#         a = low + a * (high - low)
#     elif mode == "raw":
#         pass
#     else:
#         raise ValueError(f"Unknown ACTION_MODE: {mode}")

    

#     low = np.asarray(low, dtype=np.float32)
#     high = np.asarray(high, dtype=np.float32)
#     a = np.clip(a, low, high)
#     return a
# def postprocess_action(action: np.ndarray, env, mode: str) -> np.ndarray:
#     a = np.asarray(action, dtype=np.float32)
#     if a.ndim == 2:
#         a = a[0]
#     a = a.reshape(-1)

#     # Workspace bounds inferred from dataset (if that's truly what training used)
#     # data_low  = np.array([96.0, 71.0], dtype=np.float32)
#     # data_high = np.array([375.0, 449.0], dtype=np.float32)
#     data_low  = np.array([33.0, 36.0], dtype=np.float32)
#     data_high = np.array([511.0, 490.0], dtype=np.float32)

#     if mode == "neg1_1":
#         a01 = (a + 1.0) * 0.5
#         a = data_low + a01 * (data_high - data_low)
#         a = np.clip(a, data_low, data_high)   # keep inside workspace
#     elif mode == "0_1":
#         a = data_low + a * (data_high - data_low)
#         a = np.clip(a, data_low, data_high)
#     elif mode == "raw":
#         # raw means env units; do NOT clamp to workspace
#         pass
#     else:
#         raise ValueError(f"Unknown ACTION_MODE: {mode}")

#     # Final always-safe clip to environment bounds
#     env_low  = np.asarray(env.action_space.low, dtype=np.float32)
#     env_high = np.asarray(env.action_space.high, dtype=np.float32)
#     a = np.clip(a, env_low, env_high)
#     return a.astype(np.float32)
def postprocess_action(action: np.ndarray, env, mode: str) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32)
    if a.ndim >= 2:
        a = a[0]
    a = a.reshape(-1)[:2]

    env_low  = np.asarray(env.action_space.low, dtype=np.float32)
    env_high = np.asarray(env.action_space.high, dtype=np.float32)

    if mode == "neg1_1":
        a01 = (a + 1.0) * 0.5
        a = env_low + a01 * (env_high - env_low)
    elif mode == "0_1":
        a = env_low + a * (env_high - env_low)
    elif mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown ACTION_MODE: {mode}")

    a = np.clip(a, env_low, env_high)
    return a.astype(env.action_space.dtype)


def build_batch_from_history(policy, imgs, sts, device):
    """
    imgs: list[torch.Tensor] each (3,84,84)
    sts:  list[torch.Tensor] each (2,)
    Returns batch with shapes matching policy expectation.
    """
    cfg = getattr(policy, "config", None)
    input_shapes = getattr(cfg, "input_shapes", {}) if cfg is not None else {}

    img_shape = input_shapes.get("observation.image", None)
    st_shape  = input_shapes.get("observation.state", None)

    # Default fallbacks: latest-only
    obs_image = imgs[-1].unsqueeze(0)   # (1,3,84,84)
    obs_state = sts[-1].unsqueeze(0)    # (1,2)

    # If policy expects history + camera dim: (B,S,N,C,H,W)
    if isinstance(img_shape, (list, tuple)) and len(img_shape) == 5:
        # img_shape is likely [S, N, C, H, W] without batch
        obs_image = torch.stack(imgs, dim=0)          # (S,3,H,W)
        obs_image = obs_image.unsqueeze(0).unsqueeze(2)  # (1,S,1,3,H,W)

    # If policy expects state history: (B,S,D)
    if isinstance(st_shape, (list, tuple)) and len(st_shape) == 2:
        obs_state = torch.stack(sts, dim=0).unsqueeze(0)  # (1,S,2)

    return {
        "observation.image": obs_image.to(device, non_blocking=True),
        "observation.state": obs_state.to(device, non_blocking=True),
    }


# ------------------------------------------------------------------------------
# CLIP encoder for Step C hybrid
# ------------------------------------------------------------------------------

class CLIPLanguageEncoder(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        # CLIP ViT-B/32 pooled dim is 512
        self.embedding_dim = int(self.text_encoder.config.hidden_size)

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
# Load policy
# ------------------------------------------------------------------------------

# def load_base_policy(pretrained_dir: Path, device: torch.device):
#     """
#     Load a LeRobot DiffusionPolicy saved via .save_pretrained().

#     We try a couple known patterns:
#       - DiffusionPolicy.from_pretrained(path)
#       - Auto class from factory (if available)
#     """
#     try:
#         from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
#         p = DiffusionPolicy.from_pretrained(str(pretrained_dir))
#         return p.to(device)
#     except Exception:
#         pass

#     # Fallback: try factory loader if your version supports it
#     try:
#         from lerobot.policies.factory import make_policy
#         try:
#             p = make_policy(pretrained_policy_path=str(pretrained_dir))
#             return p.to(device)
#         except TypeError:
#             raise RuntimeError(
#                 "Could not load policy. Your LeRobot version may not support this loader path.\n"
#                 f"Tried DiffusionPolicy.from_pretrained({pretrained_dir}) and make_policy(pretrained_policy_path=...)."
#             )
#     except Exception as e:
#         raise RuntimeError(f"Failed to load policy from {pretrained_dir}: {e}") from e

# def load_base_policy(pretrained_dir, device):
#     """
#     Load a policy from a LeRobot checkpoint directory in a version-robust way.

#     Handles API differences across LeRobot versions:
#       - DiffusionPolicy.from_pretrained(...)
#       - make_policy(pretrained_* = ...)
#     """
#     import inspect
#     from pathlib import Path

#     pretrained_dir = Path(pretrained_dir)

#     # 1) Try DiffusionPolicy.from_pretrained if available in this install
#     try:
#         from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
#         if hasattr(DiffusionPolicy, "from_pretrained"):
#             p = DiffusionPolicy.from_pretrained(str(pretrained_dir))
#             p = p.to(device).eval()
#             return p
#     except Exception as e:
#         last_err = e
#     else:
#         last_err = None

#     # 2) Fallback: call make_policy with whatever arg name this version expects
#     from lerobot.policies import make_policy
#     sig = inspect.signature(make_policy)
#     params = sig.parameters

#     # candidate kwarg names used across versions
#     candidates = [
#         "pretrained_policy_path",
#         "pretrained_policy_dir",
#         "pretrained_model_path",
#         "pretrained_model_dir",
#         "pretrained_dir",
#         "pretrained_path",
#         "checkpoint_path",
#         "checkpoint",
#         "path",
#     ]

#     kw = None
#     for k in candidates:
#         if k in params:
#             kw = {k: str(pretrained_dir)}
#             break

#     if kw is None:
#         # print something helpful
#         raise RuntimeError(
#             "Could not find a compatible argument name for make_policy().\n"
#             f"make_policy signature is: {sig}\n"
#             "Edit load_base_policy() to pass the checkpoint path using the correct kwarg."
#         )

#     try:
#         p = make_policy(**kw)
#         if hasattr(p, "to"):
#             p = p.to(device)
#         p.eval()
#         return p
#     except Exception as e:
#         raise RuntimeError(
#             f"Failed to load policy from {pretrained_dir}.\n"
#             f"make_policy signature: {sig}\n"
#             f"Tried kwargs: {kw}\n"
#             f"Earlier from_pretrained error: {repr(last_err)}\n"
#             f"make_policy error: {repr(e)}"
#         )
def load_base_policy(pretrained_dir, device):
    """
    Load a diffusion policy checkpoint from a pretrained_model folder.
    Works with newer LeRobot repos that do NOT expose lerobot.policies.make_policy.
    """
    from pathlib import Path
    pretrained_dir = Path(pretrained_dir)

    # Some checkpoints are nested (Step C had .../pretrained_model/base_policy)
    # If config.json isn't here, try base_policy subdir.
    if not (pretrained_dir / "config.json").exists():
        candidate = pretrained_dir / "base_policy"
        if (candidate / "config.json").exists():
            pretrained_dir = candidate

    # Load diffusion policy directly
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    try:
        policy = DiffusionPolicy.from_pretrained(str(pretrained_dir))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DiffusionPolicy.from_pretrained from: {pretrained_dir}\n"
            f"Original error: {repr(e)}"
        )

    policy = policy.to(device)
    policy.eval()

    # === ADD THIS DIAGNOSTIC BLOCK ===
    print("\n" + "="*80)
    print("POLICY CONFIGURATION DIAGNOSTICS")
    print("="*80)

    # Check config
    if hasattr(policy, 'config'):
        cfg = policy.config
        print(f"Policy config type: {type(cfg)}")
        if hasattr(cfg, 'input_shapes'):
            print(f"  input_shapes: {cfg.input_shapes}")
        if hasattr(cfg, 'crop_shape'):
            print(f"  crop_shape: {cfg.crop_shape}")
        if hasattr(cfg, 'normalize_mode'):
            print(f"  normalize_mode: {cfg.normalize_mode}")

    # Check normalizer
    if hasattr(policy, 'normalize_inputs'):
        print("✓ Policy has normalize_inputs method")
    if hasattr(policy, 'normalizer'):
        print(f"✓ Policy has normalizer: {type(policy.normalizer)}")
        if hasattr(policy.normalizer, 'mode'):
            print(f"  Normalizer mode: {policy.normalizer.mode}")
        if hasattr(policy.normalizer, 'stats'):
            print(f"  Normalizer has stats for keys: {list(policy.normalizer.stats.keys())}")

    # Check what the RGB encoder expects
    if hasattr(policy, 'diffusion') and hasattr(policy.diffusion, 'rgb_encoder'):
        encoder = policy.diffusion.rgb_encoder
        print(f"RGB encoder type: {type(encoder).__name__}")
        if hasattr(encoder, 'in_shape'):
            print(f"  Expected input shape: {encoder.in_shape}")
        # Check first conv layer
        first_layer = None
        if hasattr(encoder, 'model') and hasattr(encoder.model, 'conv1'):
            first_layer = encoder.model.conv1
        elif hasattr(encoder, 'backbone') and len(list(encoder.backbone.children())) > 0:
            first_layer = list(encoder.backbone.children())[0]
        
        if first_layer is not None and hasattr(first_layer, 'weight'):
            weight_shape = first_layer.weight.shape
            print(f"  First conv layer weight shape: {weight_shape}")
            print(f"  → Expects input: (B, {weight_shape[1]}, H, W)")

    print("="*80 + "\n")
    # === END DIAGNOSTIC BLOCK ===
    return policy



def policy_needs_language(policy) -> bool:
    try:
        return bool(getattr(policy.diffusion, "use_language_cond", False))
    except Exception:
        return False


def try_read_policy_n_obs_steps(policy, default: int = 2) -> int:
    # best-effort; different versions store this differently
    for attr in ["n_obs_steps", "num_obs_steps"]:
        if hasattr(policy, attr):
            v = getattr(policy, attr)
            if isinstance(v, int) and v > 0:
                return v
    # sometimes stored in config
    cfg = getattr(policy, "config", None)
    if cfg is not None and hasattr(cfg, "n_obs_steps"):
        v = getattr(cfg, "n_obs_steps")
        if isinstance(v, int) and v > 0:
            return v
    return default


# ------------------------------------------------------------------------------
# Environment creation
# ------------------------------------------------------------------------------

def make_pusht_env():
    """
    Create PushT env directly with gymnasium.
    We need the IMAGE-BASED version for our policy!
    """
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    # Import gym_pusht to register the environment
    try:
        import gym_pusht
    except ImportError:
        pass
    
    # Try to create the IMAGE-BASED version explicitly
    env_ids = [
        "gym_pusht/PushT-v0",
        "PushT-v0",
    ]
    
    for env_id in env_ids:
        try:
            # Try to force render_mode='rgb_array' for image observations
            # env = gym.make(env_id, render_mode='rgb_array', obs_type='pixels')
            env = gym.make(env_id, render_mode="rgb_array", obs_type="pixels_agent_pos")

            return env
        except Exception:
            try:
                # Try without obs_type
                env = gym.make(env_id, render_mode='rgb_array')
                return env
            except Exception:
                try:
                    # Try just the env_id
                    env = gym.make(env_id)
                    return env
                except Exception:
                    continue
    
    raise RuntimeError(f"Could not create PushT environment. Tried: {env_ids}")


# ------------------------------------------------------------------------------
# Main eval
# ------------------------------------------------------------------------------

def main():
    ckpt = os.environ.get("CKPT_EVAL", "").strip()
    if not ckpt:
        raise ValueError("Set CKPT_EVAL to a pretrained_model directory.")

    device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    n_eval = int(os.environ.get("N_EVAL", "20"))
    max_steps = int(os.environ.get("MAX_STEPS", "300"))
    seed = int(os.environ.get("SEED", "0"))

    action_mode = os.environ.get("ACTION_MODE", "neg1_1")
    history_mode = os.environ.get("HISTORY_MODE", "last")

    instruction = os.environ.get("INSTRUCTION", "Push the T-shaped block to the target.")
    success_key = os.environ.get("SUCCESS_KEY", "").strip() or None
    success_reward = os.environ.get("SUCCESS_REWARD", "").strip()
    success_reward = float(success_reward) if success_reward else None

    ckpt_dir = Path(ckpt)
    # Step C hybrid saves base policy in subdir "base_policy"
    base_policy_dir = ckpt_dir / "base_policy"
    if base_policy_dir.exists():
        base_dir = base_policy_dir
        hybrid = True
    else:
        base_dir = ckpt_dir
        hybrid = False

    set_seed(seed)

    print("=" * 80)
    print("Evaluating PushT checkpoint")
    print("=" * 80)
    print(f"CKPT_EVAL: {ckpt_dir}")
    print(f"Resolved base policy dir: {base_dir}")
    print(f"Device: {device}")
    print(f"Episodes: {n_eval}")
    print(f"Max steps: {max_steps}")
    print(f"ACTION_MODE: {action_mode}")
    print(f"HISTORY_MODE: {history_mode}")
    print(f"INSTRUCTION: {instruction!r}")
    print()

    # Load policy
    policy = load_base_policy(base_dir, device)
    print("Has normalizer:", hasattr(policy, "normalize_inputs") or hasattr(policy, "normalizer") or hasattr(policy, "input_normalizer"))

    policy.eval()

    # DEBUG: check what rgb encoder expects
    print("DEBUG: rgb_encoder expects input:", getattr(policy.diffusion.rgb_encoder, "in_shape", None))
    w = next(policy.diffusion.rgb_encoder.parameters())
    print("DEBUG: first rgb_encoder param shape:", w.shape)

    needs_lang = policy_needs_language(policy)
    n_obs_steps = int(os.environ.get("N_OBS_STEPS", "0")) or try_read_policy_n_obs_steps(policy, default=2)

    print("Policy loaded:")
    print(f"  type: {type(policy)}")
    print(f"  use_language_cond: {needs_lang}")
    print(f"  n_obs_steps: {n_obs_steps}")
    print()

    # If language conditioning is enabled but base policy didn't create its own CLIP,
    # we will provide language_embedding via a CLIP encoder (Step C hybrid behavior).
    clip_encoder = None
    if needs_lang:
        # Determine CLIP model name if clip_info.json exists
        clip_model_name = "openai/clip-vit-base-patch32"
        clip_info = ckpt_dir / "clip_info.json"
        if clip_info.exists():
            try:
                clip_model_name = json.loads(clip_info.read_text()).get("model_name", clip_model_name)
            except Exception:
                pass

        clip_encoder = CLIPLanguageEncoder(clip_model_name).to(device)
        print(f"Using manual CLIP encoder for language_embedding: {clip_model_name}")
        print(f"  embedding_dim: {clip_encoder.embedding_dim}")
        print()

    # Create env
    env = make_pusht_env()
    print("action_space low/high:", env.action_space.low, env.action_space.high)
    print("random action example:", env.action_space.sample())
    if hasattr(env.observation_space, "spaces") and "agent_pos" in env.observation_space.spaces:
        ap_space = env.observation_space.spaces["agent_pos"]
        print("agent_pos space low/high:", ap_space.low, ap_space.high, "dtype:", ap_space.dtype)


    print("DEBUG unwrapped attrs contain agent_pos?:", hasattr(env.unwrapped, "agent_pos"))
    for name in ["agent_pos", "_agent_pos", "state", "_state", "robot_pos", "cursor_pos"]:
        if hasattr(env.unwrapped, name):
            v = getattr(env.unwrapped, name)
            print("  ", name, type(v), v if np.isscalar(v) else (np.array(v).shape, np.array(v)[:5]))

    results = {
        "ckpt_eval": str(ckpt_dir),
        "base_policy_dir": str(base_dir),
        "device": str(device),
        "n_eval": n_eval,
        "max_steps": max_steps,
        "action_mode": action_mode,
        "history_mode": history_mode,
        "n_obs_steps": n_obs_steps,
        "instruction": instruction,
        "episodes": [],
    }

    t0 = time.time()
    successes = 0
    rewards = []

    # DEBUG: Check what the environment actually returns
    if n_eval > 0:
        test_obs = env.reset(seed=seed) if "seed" in env.reset.__code__.co_varnames else env.reset()
        print("DEBUG: Environment observation check:")
        print(f"  Type: {type(test_obs)}")
        if isinstance(test_obs, tuple):
            print(f"  Tuple length: {len(test_obs)}")
            print(f"  First element type: {type(test_obs[0])}")
            if hasattr(test_obs[0], 'shape'):
                print(f"  First element shape: {test_obs[0].shape}")
            elif isinstance(test_obs[0], dict):
                print(f"  First element keys: {list(test_obs[0].keys())}")
        elif hasattr(test_obs, 'shape'):
            print(f"  Shape: {test_obs.shape}")
        elif isinstance(test_obs, dict):
            print(f"  Keys: {list(test_obs.keys())}")
            for k, v in test_obs.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print()

    for ep in range(1, n_eval + 1):
        obs = env.reset(seed=seed + ep) if "seed" in env.reset.__code__.co_varnames else env.reset()
        policy.reset()

        # Maintain obs history
        img_hist = deque(maxlen=n_obs_steps)
        state_hist = deque(maxlen=n_obs_steps)
        def unpack_obs(o, env=None):
            # If tuple, extract first element
            if isinstance(o, tuple):
                o = o[0]

            # Dict obs: (pixels, agent_pos)
            if isinstance(o, dict):
                img = None
                for k in ["pixels", "image", "rgb", "observation.image"]:
                    if k in o and o[k] is not None:
                        img = o[k]
                        break

                st = None
                for k in ["agent_pos", "state", "observation.state"]:
                    if k in o and o[k] is not None:
                        st = o[k]
                        break

                if img is None:
                    raise ValueError(f"Could not find image in obs keys: {list(o.keys())}")
                if st is None:
                    raise ValueError(f"Could not find state in obs keys: {list(o.keys())}")

                return img, st

            # Numpy obs fallback (pixels-only or flattened)
            if isinstance(o, np.ndarray):
                if o.shape == (96, 96, 3):
                    if env is None:
                        return o, np.zeros(2, dtype=np.float32)
                    st = getattr(env.unwrapped, "agent_pos", None) or getattr(env.unwrapped, "_agent_pos", None)
                    if st is None:
                        raise ValueError("Pixels-only obs but could not find env.unwrapped.agent_pos (or _agent_pos).")
                    return o, np.asarray(st, dtype=np.float32)

                if o.ndim == 1:
                    if o.shape[0] == 27650:
                        img = o[:27648].reshape(96, 96, 3)
                        st = o[27648:]
                        return img, st
                    raise ValueError(f"Unexpected flattened obs shape: {o.shape}")

                raise ValueError(f"Unexpected array obs shape: {o.shape}")

            raise ValueError(f"Unexpected obs type: {type(o)}, shape: {getattr(o, 'shape', 'N/A')}")


        # Push first obs into history enough times to fill window
        # def unpack_obs(o, env):
        #     """
        #     Extract image and state from PushT observation.
            
        #     PushT can return:
        #     - Tuple (obs, info) from reset()
        #     - Array directly (flattened obs)
        #     - Dict with "pixels" and "agent_pos"
        #     """
        #     # If tuple, extract first element
        #     if isinstance(o, tuple):
        #         o = o[0]
            
        #     # If dict, extract image and state
        #     if isinstance(o, dict):
        #         img = o.get("pixels") or o.get("image") or o.get("rgb")
        #         st = o.get("agent_pos") or o.get("state")
                
        #         # Try LeRobot-style keys as fallback
        #         if img is None:
        #             img = o.get("observation.image")
        #         if st is None:
        #             st = o.get("observation.state")
                
        #         if img is None:
        #             raise ValueError(f"Could not find image in obs keys: {list(o.keys())}")
        #         if st is None:
        #             raise ValueError(f"Could not find state in obs keys: {list(o.keys())}")
                
        #         return img, st
            
        #     # # If numpy array, need to check what format it is
        #     # if isinstance(o, np.ndarray):
        #     #     # PushT observation space is Box(low=0, high=255, shape=(96, 96, 3))
        #     #     if o.shape == (96, 96, 3):
        #     #         # Just the image, no state available
        #     #         # Use zeros for state as placeholder
        #     #         return o, np.zeros(2, dtype=np.float32)
        #     #     elif len(o.shape) == 1:
        #     #         # Flattened observation - need to reshape
        #     #         # Typical PushT: 96*96*3 + 2 = 27650 elements
        #     #         if o.shape[0] == 27650:
        #     #             img = o[:27648].reshape(96, 96, 3)
        #     #             st = o[27648:]
        #     #             return img, st
        #     # If numpy array, need to check what format it is
        #     if isinstance(o, np.ndarray):
        #         # Pixels-only
        #         if o.shape == (96, 96, 3):
        #             img = o
        #             # grab real state from env
        #             st = getattr(env.unwrapped, "agent_pos", None)
        #             if st is None:
        #                 st = getattr(env.unwrapped, "_agent_pos", None)
        #             if st is None:
        #                 raise ValueError("Pixels-only obs but could not find env.unwrapped.agent_pos (or _agent_pos).")
        #             st = np.asarray(st, dtype=np.float32)
        #             return img, st

        #         # Flattened observation (image + state packed into 1D)
        #         if o.ndim == 1:
        #             if o.shape[0] == 27650:
        #                 img = o[:27648].reshape(96, 96, 3)
        #                 st  = o[27648:]
        #                 return img, st
        #             else:
        #                 raise ValueError(f"Unexpected flattened obs shape: {o.shape}")

        #         # Anything else is unexpected
        #         raise ValueError(f"Unexpected array obs shape: {o.shape}")

        img0, st0 = unpack_obs(obs, env)
        for _ in range(n_obs_steps):
            img_hist.append(img0)
            state_hist.append(st0)

        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}
        success_any = False
        success_used = "no_success_signal"

        while not done and steps < max_steps:
            if history_mode != "last":
                raise ValueError("Only HISTORY_MODE=last is supported in this script.")

            # --- Build history tensors (S = n_obs_steps) ---
            imgs = []
            sts = []
            for x_img, x_st in zip(list(img_hist), list(state_hist)):
                img_t = to_torch_obs_image(np.asarray(x_img))
                st_t  = to_torch_obs_state(np.asarray(x_st))

                # IMPORTANT: force shapes to be (3,H,W) and (state_dim,)
                # because some helpers may already return a batch dim.
                if img_t.dim() == 4 and img_t.shape[0] == 1:
                    img_t = img_t.squeeze(0)  # (3,H,W)
                if st_t.dim() == 2 and st_t.shape[0] == 1:
                    st_t = st_t.squeeze(0)    # (state_dim,)

                imgs.append(img_t)
                sts.append(st_t)

            # Stack time: (S,3,H,W) -> (B=1,S,3,H,W) -> (B=1,S,N=1,3,H,W)
            # obs_image = torch.stack(imgs, dim=0)              # (S,3,H,W)
            # obs_image = obs_image.unsqueeze(0)  # (1,S,1,3,H,W)

            # # Stack time: (S,state_dim) -> (B=1,S,state_dim)
            # obs_state = torch.stack(sts, dim=0).unsqueeze(0)  # (1,S,state_dim)
            # This LeRobot version wants ONLY the latest obs (no history dims)
            # obs_image = imgs[-1].unsqueeze(0) 
            # obs_state = st_raw.unsqueeze(0)                # (1,3,84,84)

            st_raw = sts[-1]
            if ep == 1 and steps < 5:
                print("DEBUG st_raw:", st_raw.detach().cpu().numpy())
                                 # torch tensor (2,)
            # # obs_state = (st_raw / 256.0) - 1.0                # (2,) in [-1,1]
            # if STATE_MODE == "raw":
            #     obs_state = st_raw
            # elif STATE_MODE == "0_1":
            #     obs_state = st_raw / 512.0
            # elif STATE_MODE == "neg1_1":
            #     obs_state = (st_raw / 256.0) - 1.0
            # else:
            #     raise ValueError(f"Unknown STATE_MODE={STATE_MODE}")
            # obs_state = obs_state.unsqueeze(0)                # (1,2)
            # CORRECT: MIN_MAX normalization to match training
            # obs_state = sts[-1] / 512.0  # [0, 512] → [0, 1]
            # obs_state = obs_state.unsqueeze(0)  # (1, 2)

            # if ep == 1 and steps < 5:
            #     print("DEBUG st_raw:", st_raw.detach().cpu().numpy())
            #     print("DEBUG st_norm:", obs_state.detach().cpu().numpy())



            # batch = {
            #     "observation.image": obs_image.to(device, non_blocking=True),
            #     "observation.state": obs_state.to(device, non_blocking=True),
            # }
            batch = build_batch_from_history(policy, imgs, sts, device)

            # --- Debug what we're feeding the model (first step only) ---
            if ep == 1 and steps == 0:
                print("DEBUG image fed:", batch["observation.image"].shape,
                    "min/max", float(batch["observation.image"].min()), float(batch["observation.image"].max()))
                print("DEBUG state fed:", batch["observation.state"].shape,
                    "vals", batch["observation.state"].detach().cpu().numpy())

            # get current obs (obs may be tuple in gymnasium)
            # img, st = unpack_obs(obs)

            # obs_image = to_torch_obs_image(np.asarray(img)).unsqueeze(0)  # (1,3,84,84)
            # obs_state = to_torch_obs_state(np.asarray(st)).unsqueeze(0)   # (1,state_dim)

            # batch = {
            #     "observation.image": obs_image.to(device, non_blocking=True),
            #     "observation.state": obs_state.to(device, non_blocking=True),
            # }


            # --- Language conditioning (Step C only) ---
            if needs_lang:
                # raw text is useful for debugging/logging; embedding is what matters
                batch["language"] = [instruction]

                if clip_encoder is not None:
                    # should be (B, 512) for clip-vit-base-patch32
                    batch["language_embedding"] = clip_encoder.encode([instruction], device)

            # --- Debug (first step only) ---
            if ep == 1 and steps == 0:
                print("DEBUG batch shapes:",
                    batch["observation.image"].shape,
                    batch["observation.state"].shape,
                    "lang_emb" if "language_embedding" in batch else "no_lang_emb")

            # --- Select action ---
            with torch.no_grad():
                if hasattr(policy, "select_action"):
                    act = policy.select_action(batch)
                else:
                    out = policy(batch)
                    act = out["action"] if isinstance(out, dict) and "action" in out else out
            if ep == 1 and steps == 0:
                a = act.detach().cpu().numpy() if hasattr(act, "detach") else np.asarray(act)
                print("DEBUG raw model act:", a, "min/max:", float(a.min()), float(a.max()))


            # act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else np.asarray(act)
            # if ep == 1 and steps < 5:
            #     a = act_np
            #     if a.ndim == 2:
            #         a = a[0]
            #     a = np.asarray(a).reshape(-1)
            #     print("DEBUG act_np:", a, "min", a.min(), "max", a.max(), "mean", a.mean())
            # a = np.asarray(act_np).reshape(-1)[:2]
            # env_a = postprocess_action(a, env, action_mode)
            # print("act_np:", a, " -> env_act:", env_a,
            #     "in_range:", np.all(env_a >= env.action_space.low) and np.all(env_a <= env.action_space.high))
            act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else np.asarray(act)

            # flatten to 1D action vector (first element if batched)
            a = np.asarray(act_np)
            if a.ndim >= 2:
                a = a[0]
            a = a.reshape(-1)[:2].astype(np.float32)

            if ep == 1 and steps < 5:
                print("DEBUG act_np:", a, "min", a.min(), "max", a.max(), "mean", a.mean())

            env_act = postprocess_action(a, env, action_mode)

            env_act = np.asarray(env_act, dtype=env.action_space.dtype)
            env_act = np.clip(env_act, env.action_space.low, env.action_space.high)


            in_range = np.all(env_act >= env.action_space.low) and np.all(env_act <= env.action_space.high)
            if ep == 1 and steps < 5:
                print("act_np:", a, "-> env_act:", env_act, "in_range:", in_range)
                print("env_act shape/dtype:", np.asarray(env_act).shape, np.asarray(env_act).dtype)




            # --- Step env ---
            # policy_mode = os.getenv("POLICY_MODE", "policy")  # policy | random
            # action_mode = os.getenv("ACTION_MODE", "neg1_1")

            # if policy_mode == "random":
            #     env_act = env.action_space.sample().astype(env.action_space.dtype)
            # else:
            #     act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else np.asarray(act)
            #     a = np.asarray(act_np)
            #     if a.ndim >= 2:
            #         a = a[0]
            #     a = a.reshape(-1)[:2].astype(np.float32)

            #     env_act = postprocess_action(a, env, action_mode)

            # hard clip + dtype match
            # env_act = postprocess_action(a, env, action_mode)
            # env_act = np.asarray(env_act, dtype=env.action_space.dtype)
            # env_act = np.clip(env_act, env.action_space.low, env.action_space.high)
            print("env_act:", env_act, "contains:", env.action_space.contains(env_act))
            print("low/high:", env.action_space.low, env.action_space.high, "dtype:", env.action_space.dtype)


            if ep == 1 and steps < 5:
                print("FINAL env_act:", env_act,
                    "contains:", env.action_space.contains(env_act),
                    "low/high:", env.action_space.low, env.action_space.high,
                    "dtype:", env_act.dtype)


            step_out = env.step(env_act)
            if len(step_out) == 4:
                obs, r, done, info = step_out
                terminated, truncated = False, False
            elif len(step_out) == 5:
                obs, r, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                raise ValueError(f"Unexpected env.step output length: {len(step_out)}")


            info = info or {}

            # track success if it appears at ANY step
            for k in ("is_success", "success", "task_success"):
                if k in info and bool(info[k]):
                    success_any = True
                    success_used = f"any_step_info[{k}]"
                    break

            # img_new, st_new = unpack_obs(obs, env)



            # if ep == 1 and steps < 5:
            #     x = np.asarray(img_new)
            #     print("[RAW IMG step]", x.dtype, x.shape, "min/max", float(x.min()), float(x.max()))



            # Unpack step output immediately -> NEW obs
            

            # after you unpack the next obs:
            img_new, st_new = unpack_obs(obs, env)

            # if ep == 1 and steps < 5:
            #     x = np.asarray(img_new)
            #     print("[RAW IMG step]", x.dtype, x.shape, "min/max", float(x.min()), float(x.max()))


            # img0, st0 = unpack_obs(obs, env)

            # --- DEBUG raw env obs (right after unpack) ---
            # if ep == 1:   # or just steps == 0 if you want less spam
            #     x = np.asarray(img0)
            #     print("[RAW IMG0]", type(img0), "np", x.dtype, x.shape,
            #         "min/max", float(x.min()), float(x.max()))
            #     s = np.asarray(st0)
            #     print("[RAW ST0 ]", type(st0), "np", s.dtype, s.shape, "vals", s)
            # ---------------------------------------------

            # st_new_np = np.asarray(st_new, dtype=np.float32).reshape(-1)
            # if ep == 1 and steps < 20:
            #     print("DEBUG st_new range:", st_new_np, "min", st_new_np.min(), "max", st_new_np.max())
            #     assert np.all(st_new_np >= -1e-3) and np.all(st_new_np <= 512 + 1e-3), "agent_pos out of expected [0,512]!"

            # if ep == 1 and steps < 5:
            #     print("DEBUG post_state(raw):", np.asarray(st_new))
# for debug above to the -----------

            # last_info = info or {}
            # ep_reward += float(r)
            # steps += 1

            # # --- Update history with NEW obs ---
            # img_new, st_new = unpack_obs(obs, env)
            # img_hist.append(img_new)
            # state_hist.append(st_new)
            # NOW unpack the NEW observation


            last_info = info or {}
            ep_reward += float(r)
            steps += 1

            # Update history with NEW obs
            img_hist.append(img_new)
            state_hist.append(st_new)



        # while not done and steps < max_steps:
        #     if history_mode != "last":
        #         raise ValueError("Only HISTORY_MODE=last is supported in this script.")
        #     img, st = unpack_obs(obs)

        #     obs_image = to_torch_obs_image(np.asarray(img)).unsqueeze(0)  # (1,3,84,84)
        #     obs_state = to_torch_obs_state(np.asarray(st)).unsqueeze(0)  # (1,state_dim)

        #     batch = {
        #         "observation.image": obs_image.to(device, non_blocking=True),
        #         "observation.state": obs_state.to(device, non_blocking=True),
        #     }

        #     if needs_lang:
        #         batch["language"] = [instruction]
        #         if clip_encoder is not None:
        #             batch["language_embedding"] = clip_encoder.encode([instruction], device)  # (1,512)

        #     # Build tensors for history window
        #     imgs = [to_torch_obs_image(np.asarray(x)) for x in list(img_hist)]  # (3,84,84) each
        #     sts  = [to_torch_obs_state(np.asarray(x)) for x in list(state_hist)] # (state_dim,) each

        #     obs_image = torch.stack(imgs, dim=0)          # (S,3,84,84)
        #     obs_image = obs_image.unsqueeze(0)            # (1,S,3,84,84)
        #     obs_image = obs_image.unsqueeze(2)            # (1,S,1,3,84,84)  <-- KEEP THIS

        #     obs_state = torch.stack(sts, dim=0).unsqueeze(0)  # (1,S,state_dim)

        #     batch = {
        #         "observation.image": obs_image.to(device, non_blocking=True),
        #         "observation.state": obs_state.to(device, non_blocking=True),
        #     }

        #     # imgs = [to_torch_obs_image(np.asarray(x)) for x in list(img_hist)]   # each (3, H, W)
        #     # sts = [to_torch_obs_state(np.asarray(x)) for x in list(state_hist)]  # each (state_dim,)

        #     # # Stack over time
        #     # obs_image = torch.stack(imgs, dim=0)        # (S, 3, H, W)
        #     # obs_state = torch.stack(sts, dim=0)         # (S, state_dim)

        #     # # Add batch dim and camera dim (LeRobot expects camera dim!)
        #     # # obs_image = obs_image.unsqueeze(0).unsqueeze(2)   # (B=1, S, N=1, 3, H, W)
        #     # obs_image = obs_image.unsqueeze(0)   # (B=1, S, 3, H, W) - NO camera dim

        #     # obs_state = obs_state.unsqueeze(0)                # (B=1, S, state_dim)

        #     # batch = {
        #     #     "observation.image": obs_image.to(device, non_blocking=True),
        #     #     "observation.state": obs_state.to(device, non_blocking=True),
        #     # }
        #     # Build batch with shapes matching LeRobot diffusion expectations
        #     # imgs = [to_torch_obs_image(np.asarray(x)) for x in list(img_hist)]  # each (3,84,84)
        #     # sts  = [to_torch_obs_state(np.asarray(x)) for x in list(state_hist)]

        #     # obs_image = torch.stack(imgs, dim=0)          # (T, 3, H, W)
        #     # obs_image = obs_image.unsqueeze(0)            # (1, T, 3, H, W)
        #     # # obs_image = obs_image.unsqueeze(2)            # (1, T, 1, 3, H, W)  <-- camera dim N=1

        #     # obs_state = torch.stack(sts, dim=0).unsqueeze(0)  # (1, T, state_dim)

        #     # batch = {
        #     #     "observation.image": obs_image.to(device, non_blocking=True),
        #     #     "observation.state": obs_state.to(device, non_blocking=True),
        #     # }


        #     # Language
        #     # if needs_lang:
        #     #     # Provide both raw language and embedding (embedding is what matters)
        #     #     batch["language"] = [instruction]
        #     #     if clip_encoder is not None:
        #     #         lang_emb = clip_encoder.encode([instruction], device)  # (1, 512)
        #     #         batch["language_embedding"] = lang_emb

        #     # Select action
        #     with torch.no_grad():
        #         if hasattr(policy, "select_action"):
        #             # DEBUG print (only first step of first episode)
        #             if ep == 1 and steps == 0:
        #                 print("DEBUG batch shapes:",
        #                     batch["observation.image"].shape,
        #                     batch["observation.state"].shape)

        #             act = policy.select_action(batch)
        #         else:
        #             out = policy(batch)
        #             # Some versions return dict with action, some return tensor
        #             if isinstance(out, dict) and "action" in out:
        #                 act = out["action"]
        #             else:
        #                 act = out

            # if isinstance(act, torch.Tensor):
            #     act_np = act.detach().cpu().numpy()
            # else:
            #     act_np = np.asarray(act)

            # # Postprocess and step env
            # env_act = postprocess_action(act_np, env, action_mode)
            # step_out = env.step(env_act)

            # # handle gym/gymnasium step signatures
            # if len(step_out) == 4:
            #     obs, r, done, info = step_out
            # elif len(step_out) == 5:
            #     obs, r, terminated, truncated, info = step_out
            #     done = bool(terminated or truncated)
            # else:
            #     raise ValueError(f"Unexpected env.step output length: {len(step_out)}")

            # last_info = info or {}
            # ep_reward += float(r)
            # steps += 1

            # # Update history
            # img, st = unpack_obs(obs)
            # img_hist.append(img)
            # state_hist.append(st)

        # success, used = infer_success(last_info, success_key, ep_reward, success_reward)
        if success_key is None and success_reward is None:
            # prefer any-step success if available; otherwise fallback to last_info inference
            if success_any:
                success, used = True, success_used
            else:
                success, used = infer_success(last_info, None, ep_reward, None)
        else:
            # if user forced a key or reward threshold, keep your existing behavior
            success, used = infer_success(last_info, success_key, ep_reward, success_reward)

        successes += int(success)
        rewards.append(ep_reward)

        results["episodes"].append({
            "episode": ep,
            "success": bool(success),
            "success_signal": used,
            "reward": ep_reward,
            "steps": steps,
        })

        print(f"Episode {ep:3d}/{n_eval}: {'✓ SUCCESS' if success else '✗ FAIL'} | steps={steps:3d} | reward={ep_reward:8.2f} | ({used})")

    dt = time.time() - t0
    success_rate = successes / max(1, n_eval)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    results["summary"] = {
        "successes": successes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "time_sec": dt,
    }

    # Default output JSON path near checkpoint
    out_json_env = os.environ.get("OUT_JSON", "").strip()
    if out_json_env:
        out_json = Path(out_json_env)
    else:
        # Put under outputs/<run>/eval_results_*.json if possible
        # ckpt_dir = .../checkpoints/XXXXXX/pretrained_model
        run_root = ckpt_dir.parents[2] if len(ckpt_dir.parts) >= 3 else ckpt_dir
        out_json = run_root / f"eval_results_{ckpt_dir.parents[1].name}_{action_mode}_T{n_obs_steps}_{history_mode}.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("EVAL DONE")
    print("=" * 80)
    print(f"Success rate: {successes}/{n_eval} = {success_rate*100:.1f}%")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()