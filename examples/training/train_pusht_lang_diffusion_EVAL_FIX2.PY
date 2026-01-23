#!/usr/bin/env python3
"""
Fixed evaluation script for PushT diffusion checkpoints.

BUG FIX: Properly reshapes observation tensors to match LeRobot's expected format.
The key issue was that LeRobot's DiffusionRgbEncoder expects (B*T, C, H, W) internally,
but the policy's _prepare_global_conditioning handles the reshape.

Your input should be: (B, T, C, H, W) - which is what you were providing!
The actual issue is likely in how the policy config was set up during training.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch_obs_image(img: np.ndarray, target_size=(84, 84)) -> torch.Tensor:
    """
    Convert env image to torch tensor with shape (3, H, W).
    Normalizes to [0, 1] and resizes.
    """
    if img is None:
        raise ValueError("Observation image is None")
    
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0, 1]
    
    # Resize if needed
    if t.shape[1:] != target_size:
        t = F.interpolate(
            t.unsqueeze(0), 
            size=target_size, 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0)
    
    return t.contiguous()


def to_torch_obs_state(state: np.ndarray) -> torch.Tensor:
    """Convert env state to torch tensor."""
    if state is None:
        raise ValueError("Observation state is None")
    return torch.from_numpy(np.asarray(state)).float().contiguous()


def infer_success(info: dict, success_key: str | None, reward: float, success_reward: float | None):
    """Returns: (success: bool, used: str)"""
    if info is None:
        info = {}

    if success_key and success_key in info:
        return bool(info[success_key]), f"info[{success_key}]"

    # Auto-detect common keys
    for k in ["is_success", "success", "task_success"]:
        if k in info:
            return bool(info[k]), f"info[{k}]"

    # Fallback to reward threshold
    if success_reward is not None:
        return (reward >= success_reward), f"reward>={success_reward}"

    return False, "no_success_signal"


def postprocess_action(action: np.ndarray, env, mode: str) -> np.ndarray:
    """
    Map model action to env action space.
    
    mode:
      - "neg1_1": action in [-1, 1]
      - "0_1": action in [0, 1]
      - "raw": no scaling
    """
    a = np.asarray(action, dtype=np.float32)
    
    # Take first action if batched (horizon, act_dim)
    if a.ndim == 2:
        a = a[0]
    if a.ndim != 1:
        a = a.reshape(-1)
    
    low = getattr(getattr(env, "action_space", None), "low", None)
    high = getattr(getattr(env, "action_space", None), "high", None)
    
    if low is None or high is None:
        return a
    
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    
    if mode == "neg1_1":
        a = (a + 1.0) * 0.5  # [-1,1] -> [0,1]
        a = low + a * (high - low)
    elif mode == "0_1":
        a = low + a * (high - low)
    elif mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown ACTION_MODE: {mode}")
    
    return np.clip(a, low, high)


class CLIPLanguageEncoder(torch.nn.Module):
    """CLIP text encoder for language conditioning."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Freeze
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()
        
        self.embedding_dim = int(self.text_encoder.config.hidden_size)
    
    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode texts to CLIP embeddings."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        out = self.text_encoder(**tokens)
        return out.pooler_output  # (B, 512)


def load_base_policy(pretrained_dir: Path, device: torch.device):
    """Load a DiffusionPolicy from pretrained directory."""
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(pretrained_dir))
        return policy.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load policy from {pretrained_dir}: {e}") from e


def policy_needs_language(policy) -> bool:
    """Check if policy expects language conditioning."""
    try:
        return bool(getattr(policy.diffusion, "use_language_cond", False))
    except Exception:
        return False


def try_read_policy_n_obs_steps(policy, default: int = 2) -> int:
    """Try to infer n_obs_steps from policy config."""
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
            # Try with explicit render_mode for images
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


def unpack_obs(obs):
    """
    Extract (image, state) from PushT observation.
    
    Handles different observation formats:
    - Tuple (obs, info) from reset()
    - Dict with "pixels"/"agent_pos"
    - Raw array
    """
    # Handle tuple from reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Dict format
    if isinstance(obs, dict):
        img = obs.get("pixels") or obs.get("image") or obs.get("rgb")
        state = obs.get("agent_pos") or obs.get("state")
        
        if img is None:
            img = obs.get("observation.image")
        if state is None:
            state = obs.get("observation.state")
        
        if img is None:
            raise ValueError(f"No image in obs keys: {list(obs.keys())}")
        if state is None:
            raise ValueError(f"No state in obs keys: {list(obs.keys())}")
        
        return img, state
    
    # Array format
    if isinstance(obs, np.ndarray):
        if obs.shape == (96, 96, 3):
            # Just image, no state
            return obs, np.zeros(2, dtype=np.float32)
        elif len(obs.shape) == 1:
            # Flattened: 96*96*3 + 2 = 27650
            if obs.shape[0] == 27650:
                img = obs[:27648].reshape(96, 96, 3)
                state = obs[27648:]
                return img, state
            else:
                raise ValueError(f"Unexpected flattened obs shape: {obs.shape}")
        else:
            raise ValueError(f"Unexpected array obs shape: {obs.shape}")
    
    raise ValueError(f"Unexpected obs type: {type(obs)}")


def main():
    # Parse env vars
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
    
    # Check for hybrid model (Step C)
    base_policy_dir = ckpt_dir / "base_policy"
    if base_policy_dir.exists():
        base_dir = base_policy_dir
        hybrid = True
    else:
        base_dir = ckpt_dir
        hybrid = False
    
    set_seed(seed)
    
    print("=" * 80)
    print("PushT Evaluation - FIXED VERSION")
    print("=" * 80)
    print(f"CKPT: {ckpt_dir}")
    print(f"Base policy: {base_dir}")
    print(f"Device: {device}")
    print(f"Episodes: {n_eval}, Max steps: {max_steps}")
    print(f"Action mode: {action_mode}")
    print(f"Instruction: {instruction!r}")
    print()
    
    # Load policy
    policy = load_base_policy(base_dir, device)
    policy.eval()
    
    needs_lang = policy_needs_language(policy)
    n_obs_steps = int(os.environ.get("N_OBS_STEPS", "0")) or try_read_policy_n_obs_steps(policy, default=2)
    
    print(f"Policy type: {type(policy).__name__}")
    print(f"Language conditioning: {needs_lang}")
    print(f"n_obs_steps: {n_obs_steps}")
    print()
    
    # Setup language encoder if needed
    clip_encoder = None
    if needs_lang:
        clip_model_name = "openai/clip-vit-base-patch32"
        clip_info = ckpt_dir / "clip_info.json"
        if clip_info.exists():
            try:
                clip_model_name = json.loads(clip_info.read_text()).get("model_name", clip_model_name)
            except Exception:
                pass
        
        clip_encoder = CLIPLanguageEncoder(clip_model_name).to(device)
        print(f"CLIP encoder: {clip_model_name} (dim={clip_encoder.embedding_dim})")
        print()
    
    # Create environment
    env = make_pusht_env()
    
    print(f"Environment: {env}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    # Test environment observation
    if n_eval > 0:
        test_obs = env.reset(seed=seed) if "seed" in env.reset.__code__.co_varnames else env.reset()
        test_img, test_state = unpack_obs(test_obs)
        print(f"Test observation:")
        print(f"  Image shape: {test_img.shape}, dtype: {test_img.dtype}")
        print(f"  State shape: {test_state.shape}, dtype: {test_state.dtype}")
        print()
    
    # Run evaluation
    results = {
        "ckpt_eval": str(ckpt_dir),
        "base_policy_dir": str(base_dir),
        "device": str(device),
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
        last_info = {}
        
        while not done and steps < max_steps:
            # Build observation tensors
            imgs = [to_torch_obs_image(np.asarray(x)) for x in list(img_hist)]
            sts = [to_torch_obs_state(np.asarray(x)) for x in list(state_hist)]
            
            obs_image = torch.stack(imgs, dim=0)  # (T, 3, H, W)
            obs_state = torch.stack(sts, dim=0)   # (T, state_dim)
            
            # Add batch dimension - THIS IS THE KEY FIX
            # LeRobot expects (B, T, C, H, W) for images
            obs_image = obs_image.unsqueeze(0)  # (1, T, 3, H, W)
            obs_state = obs_state.unsqueeze(0)  # (1, T, state_dim)
            
            # Prepare batch
            batch = {
                "observation.image": obs_image.to(device, non_blocking=True),
                "observation.state": obs_state.to(device, non_blocking=True),
            }
            
            # Add language if needed
            if needs_lang:
                batch["language"] = [instruction]
                if clip_encoder is not None:
                    lang_emb = clip_encoder.encode([instruction], device)
                    batch["language_embedding"] = lang_emb
            
            # Debug first step
            if ep == 1 and steps == 0:
                print(f"First step batch shapes:")
                print(f"  observation.image: {batch['observation.image'].shape}")
                print(f"  observation.state: {batch['observation.state'].shape}")
                if "language_embedding" in batch:
                    print(f"  language_embedding: {batch['language_embedding'].shape}")
                print()
            
            # Get action
            with torch.no_grad():
                if hasattr(policy, "select_action"):
                    act = policy.select_action(batch)
                else:
                    out = policy(batch)
                    if isinstance(out, dict) and "action" in out:
                        act = out["action"]
                    else:
                        act = out
            
            # Convert to numpy
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            
            # Postprocess and execute
            env_act = postprocess_action(act_np, env, action_mode)
            step_out = env.step(env_act)
            
            # Parse step output
            if len(step_out) == 4:
                obs, r, done, info = step_out
            elif len(step_out) == 5:
                obs, r, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                raise ValueError(f"Unexpected env.step output length: {len(step_out)}")
            
            last_info = info or {}
            ep_reward += float(r)
            steps += 1
            
            # Update history
            img, st = unpack_obs(obs)
            img_hist.append(img)
            state_hist.append(st)
        
        # Evaluate success
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
        
        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"Ep {ep:3d}/{n_eval}: {status} | steps={steps:3d} | reward={ep_reward:8.2f} | ({used})")
    
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
    out_json_env = os.environ.get("OUT_JSON", "").strip()
    if out_json_env:
        out_json = Path(out_json_env)
    else:
        run_root = ckpt_dir.parents[2] if len(ckpt_dir.parts) >= 3 else ckpt_dir
        out_json = run_root / f"eval_results_{ckpt_dir.parents[1].name}_{action_mode}_T{n_obs_steps}.json"
    
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Success rate: {successes}/{n_eval} = {success_rate*100:.1f}%")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Time: {dt:.1f}s")
    print(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()