#!/usr/bin/env python3
"""
FIXED v2: PushT Evaluation Script for Language-Conditioned Diffusion Policy
============================================================================

This fixes the dimension mismatch in global conditioning by properly formatting
observations with temporal dimensions.

Key fix: Keep temporal dimension as (B, T, C, H, W) for images, not (B*T, C, H, W)
The policy's RGB encoder handles temporal processing internally.
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
from transformers import CLIPTokenizer, CLIPTextModel


# ==============================================================================
# UTILITIES
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch_obs_image(img: np.ndarray) -> torch.Tensor:
    """Convert env image to torch tensor (3, H, W), resize to 84x84."""
    if img is None:
        raise ValueError("Observation image is None")
    
    t = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
    t = t / 255.0
    t = F.interpolate(
        t.unsqueeze(0),
        size=(84, 84),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    return t.contiguous()


def to_torch_obs_state(state: np.ndarray) -> torch.Tensor:
    """Convert env state to torch tensor."""
    if state is None:
        raise ValueError("Observation state is None")
    return torch.from_numpy(np.asarray(state)).float().contiguous()


# ==============================================================================
# CLIP LANGUAGE ENCODER
# ==============================================================================

class CLIPLanguageEncoder(torch.nn.Module):
    """CLIP encoder for language instructions."""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

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


# ==============================================================================
# POLICY LOADING
# ==============================================================================

def load_base_policy(pretrained_dir: Path, device: torch.device):
    """Load LeRobot DiffusionPolicy from checkpoint."""
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        p = DiffusionPolicy.from_pretrained(str(pretrained_dir))
        return p.to(device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load policy from {pretrained_dir}: {e}"
        ) from e


def policy_needs_language(policy) -> bool:
    """Check if policy expects language conditioning."""
    try:
        return bool(getattr(policy.diffusion, "use_language_cond", False))
    except Exception:
        return False


def try_read_policy_n_obs_steps(policy, default: int = 2) -> int:
    """Infer n_obs_steps from policy config."""
    for attr in ["n_obs_steps", "num_obs_steps"]:
        if hasattr(policy, attr):
            v = getattr(policy, attr)
            if isinstance(v, int) and v > 0:
                return v
    cfg = getattr(policy, "config", None)
    if cfg is not None and hasattr(cfg, "n_obs_steps"):
        v = getattr(cfg, "n_obs_steps")
        if isinstance(v, int) and v > 0:
            return v
    return default


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

def make_pusht_env():
    """Create PushT environment with image observations."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    try:
        import gym_pusht
    except ImportError:
        pass
    
    env_ids = ["gym_pusht/PushT-v0", "PushT-v0"]
    
    for env_id in env_ids:
        try:
            env = gym.make(env_id, render_mode='rgb_array', obs_type='pixels')
            return env
        except Exception:
            try:
                env = gym.make(env_id, render_mode='rgb_array')
                return env
            except Exception:
                try:
                    env = gym.make(env_id)
                    return env
                except Exception:
                    continue
    
    raise RuntimeError(f"Could not create PushT environment. Tried: {env_ids}")


def unpack_obs(o):
    """
    Extract image and state from PushT observation.
    
    PushT can return:
    - Tuple (obs, info) from reset()
    - Array directly (flattened obs)
    - Dict with "pixels" and "agent_pos"
    """
    # If tuple, extract first element
    if isinstance(o, tuple):
        o = o[0]
    
    # If dict, extract image and state
    if isinstance(o, dict):
        img = o.get("pixels") or o.get("image") or o.get("rgb")
        st = o.get("agent_pos") or o.get("state")
        
        # Try LeRobot-style keys as fallback
        if img is None:
            img = o.get("observation.image")
        if st is None:
            st = o.get("observation.state")
        
        if img is None:
            raise ValueError(f"Could not find image in obs keys: {list(o.keys())}")
        if st is None:
            raise ValueError(f"Could not find state in obs keys: {list(o.keys())}")
        
        return img, st
    
    # If numpy array, check format
    if isinstance(o, np.ndarray):
        # PushT observation: Box(0, 255, (96, 96, 3))
        if o.shape == (96, 96, 3):
            # Just the image, no state available
            return o, np.zeros(2, dtype=np.float32)
        elif len(o.shape) == 1:
            # Flattened observation: 96*96*3 + 2 = 27650 elements
            if o.shape[0] == 27650:
                img = o[:27648].reshape(96, 96, 3)
                st = o[27648:]
                return img, st
            else:
                raise ValueError(f"Unexpected flattened obs shape: {o.shape}")
        else:
            raise ValueError(f"Unexpected array obs shape: {o.shape}")
    
    raise ValueError(f"Unexpected obs type: {type(o)}, shape: {getattr(o, 'shape', 'N/A')}")


# ==============================================================================
# ACTION POSTPROCESSING
# ==============================================================================

def postprocess_action(action: np.ndarray, env, mode: str) -> np.ndarray:
    """
    Map model action to env action space.
    
    mode:
      - "neg1_1": action in [-1, 1]
      - "0_1": action in [0, 1]
      - "raw": no scaling
    """
    a = np.asarray(action, dtype=np.float32)

    # Flatten: (horizon, act_dim) -> take first action
    if a.ndim == 2:
        a = a[0]
    if a.ndim != 1:
        a = a.reshape(-1)

    # Get env bounds
    low = getattr(getattr(env, "action_space", None), "low", None)
    high = getattr(getattr(env, "action_space", None), "high", None)

    if low is None or high is None:
        return a

    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    if mode == "neg1_1":
        a = (a + 1.0) * 0.5  # -> [0,1]
        a = low + a * (high - low)
    elif mode == "0_1":
        a = low + a * (high - low)
    elif mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown ACTION_MODE: {mode}")

    a = np.clip(a, low, high)
    return a


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def main():
    # Parse environment variables
    ckpt = os.environ.get("CKPT_EVAL", "").strip()
    if not ckpt:
        raise ValueError("Set CKPT_EVAL to a pretrained_model directory.")

    device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    n_eval = int(os.environ.get("N_EVAL", "20"))
    max_steps = int(os.environ.get("MAX_STEPS", "300"))
    seed = int(os.environ.get("SEED", "0"))

    action_mode = os.environ.get("ACTION_MODE", "neg1_1")

    instruction = os.environ.get("INSTRUCTION", "Push the T-shaped block to the target.")
    
    ckpt_dir = Path(ckpt)
    base_policy_dir = ckpt_dir / "base_policy"
    if base_policy_dir.exists():
        base_dir = base_policy_dir
    else:
        base_dir = ckpt_dir

    set_seed(seed)

    print("=" * 80)
    print("Evaluating PushT Language-Conditioned Diffusion Policy")
    print("=" * 80)
    print(f"CKPT_EVAL: {ckpt_dir}")
    print(f"Base policy dir: {base_dir}")
    print(f"Device: {device}")
    print(f"Episodes: {n_eval}")
    print(f"Max steps: {max_steps}")
    print(f"ACTION_MODE: {action_mode}")
    print(f"INSTRUCTION: {instruction!r}")
    print()

    # Load policy
    policy = load_base_policy(base_dir, device)
    policy.eval()

    needs_lang = policy_needs_language(policy)
    n_obs_steps = int(os.environ.get("N_OBS_STEPS", "0")) or try_read_policy_n_obs_steps(policy, default=2)

    print("Policy loaded:")
    print(f"  Type: {type(policy).__name__}")
    print(f"  Language conditioning: {needs_lang}")
    print(f"  n_obs_steps: {n_obs_steps}")
    print()

    # Load CLIP encoder if needed
    clip_encoder = None
    if needs_lang:
        clip_model_name = "openai/clip-vit-base-patch32"
        clip_info = ckpt_dir / "clip_info.json"
        if clip_info.exists():
            try:
                clip_model_name = json.loads(clip_info.read_text()).get(
                    "model_name", clip_model_name
                )
            except Exception:
                pass

        clip_encoder = CLIPLanguageEncoder(clip_model_name).to(device)
        print(f"CLIP encoder loaded: {clip_model_name}")
        print(f"  Embedding dim: {clip_encoder.embedding_dim}")
        print()

    # Create environment
    env = make_pusht_env()
    print(f"Environment: PushT")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Run evaluation
    results = {
        "ckpt_eval": str(ckpt_dir),
        "n_eval": n_eval,
        "max_steps": max_steps,
        "action_mode": action_mode,
        "n_obs_steps": n_obs_steps,
        "instruction": instruction,
        "episodes": [],
    }

    t0 = time.time()
    successes = 0
    rewards = []

    for ep in range(1, n_eval + 1):
        # Reset environment
        obs = env.reset(seed=seed + ep) if "seed" in env.reset.__code__.co_varnames else env.reset()

        # Initialize observation history
        img_hist = deque(maxlen=n_obs_steps)
        state_hist = deque(maxlen=n_obs_steps)

        img0, st0 = unpack_obs(obs)
        for _ in range(n_obs_steps):
            img_hist.append(img0)
            state_hist.append(st0)

        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Build observation tensors
            imgs = [to_torch_obs_image(np.asarray(x)) for x in list(img_hist)]
            sts = [to_torch_obs_state(np.asarray(x)) for x in list(state_hist)]

            # Stack over time dimension
            obs_image = torch.stack(imgs, dim=0)  # (T, C, H, W)
            obs_state = torch.stack(sts, dim=0)   # (T, state_dim)

            # =================================================================
            # KEY FIX: Keep temporal dimension, add batch dimension
            # =================================================================
            # Policy expects:
            #   observation.image: (B, T, C, H, W)
            #   observation.state: (B, T, state_dim)
            
            # Add batch dimension
            obs_image_batched = obs_image.unsqueeze(0)  # (1, T, C, H, W)
            obs_state_batched = obs_state.unsqueeze(0)  # (1, T, state_dim)
            
            # Build batch dictionary
            batch = {
                "observation.image": obs_image_batched.to(device, non_blocking=True),
                "observation.state": obs_state_batched.to(device, non_blocking=True),
            }
            # =================================================================

            # Add language conditioning if needed
            if needs_lang:
                batch["language"] = [instruction]
                if clip_encoder is not None:
                    lang_emb = clip_encoder.encode([instruction], device)
                    batch["language_embedding"] = lang_emb

            # Debug print (first step only)
            if ep == 1 and steps == 0:
                print("DEBUG: Batch shapes:")
                print(f"  observation.image: {batch['observation.image'].shape}")
                print(f"  observation.state: {batch['observation.state'].shape}")
                if needs_lang:
                    print(f"  language_embedding: {batch.get('language_embedding', torch.empty(0)).shape}")
                print()

            # Predict action
            with torch.no_grad():
                try:
                    if hasattr(policy, "select_action"):
                        act = policy.select_action(batch)
                    else:
                        out = policy(batch)
                        if isinstance(out, dict) and "action" in out:
                            act = out["action"]
                        else:
                            act = out
                except Exception as e:
                    print(f"\nERROR during policy forward pass:")
                    print(f"  Error: {e}")
                    print(f"  Batch keys: {list(batch.keys())}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: {v.shape}")
                    raise

            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)

            # Postprocess and step
            env_act = postprocess_action(act_np, env, action_mode)
            step_out = env.step(env_act)

            # Handle step output
            if len(step_out) == 4:
                obs, r, done, info = step_out
            elif len(step_out) == 5:
                obs, r, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                raise ValueError(f"Unexpected env.step output length: {len(step_out)}")

            ep_reward += float(r)
            steps += 1

            # Update history
            img, st = unpack_obs(obs)
            img_hist.append(img)
            state_hist.append(st)

        # Check success
        success = bool(info.get("is_success", False))
        successes += int(success)
        rewards.append(ep_reward)

        results["episodes"].append({
            "episode": ep,
            "success": success,
            "reward": ep_reward,
            "steps": steps,
        })

        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"Ep {ep:3d}/{n_eval}: {status} | steps={steps:3d} | reward={ep_reward:8.2f}")

    # Summary
    dt = time.time() - t0
    success_rate = successes / max(1, n_eval)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    results["summary"] = {
        "successes": successes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "time_sec": dt,
    }

    # Save results
    out_json = ckpt_dir.parent.parent / f"eval_results_{ckpt_dir.parent.name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Success rate: {successes}/{n_eval} = {success_rate*100:.1f}%")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Time: {dt:.1f}s")
    print(f"Saved: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()