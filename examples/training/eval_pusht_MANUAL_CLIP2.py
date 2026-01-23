#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/eval_pusht_MANUAL_CLIP2.py
# ==============================================================================
"""
Evaluate the manually CLIP-integrated diffusion policy.

Env vars:
  CKPT_EVAL     Path to checkpoint pretrained_model dir
  N_EVAL        Number of eval episodes (default 20)
  DEBUG         1/0 (default 0)

  ACTION_MODE   auto | 0_1 | neg1_1 | raw   (default auto)
               - auto   : infer from action range each step (robust for debugging)
               - 0_1    : assume policy outputs in [0,1], map -> [0,512]
               - neg1_1 : assume policy outputs in [-1,1], map -> [0,512]
               - raw    : assume policy outputs already in env units

  N_OBS_STEPS   observation history length to MAINTAIN (default 2)

  HISTORY_MODE  last | concat (default last)
               - last   : feed only the most recent obs to the policy (4D image, 2D state)
               - concat : feed history by concatenating along channel/state dims:
                          image: (B, T*C, H, W), state: (B, T*D)
                          This keeps conv2d happy (still 4D).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
from transformers import CLIPTextModel, CLIPTokenizer

from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.envs.factory import make_env, make_env_config


# -----------------------------
# CLIP-conditioned wrapper
# -----------------------------
class CLIPConditionedPolicy(nn.Module):
    """Wrapper that adds CLIP text conditioning to any policy."""

    def __init__(self, base_policy, clip_model_name="openai/clip-vit-base-patch32", policy_dim=128):
        super().__init__()
        self.base_policy = base_policy

        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        for p in self.text_encoder.parameters():
            p.requires_grad = False

        clip_dim = self.text_encoder.config.hidden_size
        self.lang_proj = nn.Sequential(
            nn.Linear(clip_dim, policy_dim),
            nn.ReLU(),
            nn.Linear(policy_dim, policy_dim),
        )

    def encode_text(self, text_list):
        tokens = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(self.text_encoder.device)

        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            text_embeds = outputs.pooler_output

        return self.lang_proj(text_embeds)

    def forward(self, batch):
        if "language" in batch:
            batch["language_embedding"] = self.encode_text(batch["language"])
        return self.base_policy(batch)

    def predict_action(self, batch, language=None):
        self.base_policy.eval()
        batch = dict(batch)

        if language is not None:
            if isinstance(language, str):
                language = [language]
            batch["language"] = language

        with torch.no_grad():
            if hasattr(self.base_policy, "select_action"):
                if "language" in batch:
                    batch["language_embedding"] = self.encode_text(batch["language"])
                return self.base_policy.select_action(batch)

            out = self.forward(batch)
            if isinstance(out, dict) and "action" in out:
                return out["action"]
            return out


# -----------------------------
# Helpers
# -----------------------------
def unwrap_env_bundle(x):
    seen = set()
    while isinstance(x, dict):
        obj_id = id(x)
        if obj_id in seen:
            break
        seen.add(obj_id)

        for key in ["env", "pusht", "environment"]:
            if key in x:
                x = x[key]
                break
        else:
            if len(x) == 1:
                x = list(x.values())[0]
            else:
                break

    while hasattr(x, "env"):
        x = x.env
    return x


def _first_bool(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return bool(np.array(x).flatten()[0])
    return bool(x)


def _to_tensor(v, device):
    """
    Convert numpy arrays to torch tensors with proper image formatting.
    Returns:
      - state: (D,) or (B,D)
      - image: (C,H,W) or (B,C,H,W)
    """
    t = torch.from_numpy(v).to(device)

    if t.ndim == 1:
        return t.float()  # (D,)
    if t.ndim == 2:
        return t.float()  # (B,D)
    if t.ndim == 3:
        # HWC -> CHW if last dim looks like channels
        if t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        t = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
        return t  # (C,H,W)
    if t.ndim == 4:
        # BHWC -> BCHW
        if t.shape[-1] in (1, 3, 4):
            t = t.permute(0, 3, 1, 2)
        t = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
        return t  # (B,C,H,W)

    return t.float()


def _extract_state_image(obs):
    # PushT typical dict: agent_pos, pixels
    state = obs.get("agent_pos", obs.get("state", obs.get("observation.state", None)))
    image = obs.get("pixels", obs.get("image", obs.get("observation.image", None)))
    return state, image


def _infer_action_mode(a):
    """
    Infer action mode from a single action vector a (shape (A,)).
    Treat small negatives as still [0,1]-ish output (common with imperfect squashing).
    """
    amin = float(np.min(a))
    amax = float(np.max(a))

    if amin >= -1.2 and amax <= 1.2:
        # clearly [-1,1] only if it uses meaningful negative range
        if amin < -0.35:
            return "neg1_1"
        return "0_1"

    return "raw"


def _scale_action(a, mode):
    if mode == "0_1":
        return np.clip(a * 512.0, 0.0, 512.0)
    if mode == "neg1_1":
        return np.clip((a + 1.0) * 0.5 * 512.0, 0.0, 512.0)
    return a.copy()


def _jsonify(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


def _build_policy_inputs(state_hist, image_hist, history_mode):
    """
    Build policy inputs that keep image 4D and state 2D.

    state_hist items: (D,)
    image_hist items: (C,H,W)
    """
    if history_mode == "concat":
        # concat state along feature dim -> (T*D,)
        s = torch.cat(list(state_hist), dim=0).unsqueeze(0)  # (1, T*D)
        # concat images along channel dim -> (T*C, H, W)
        img = torch.cat(list(image_hist), dim=0).unsqueeze(0)  # (1, T*C, H, W)
        return s, img

    # default: last
    s = state_hist[-1].unsqueeze(0)      # (1, D)
    img = image_hist[-1].unsqueeze(0)    # (1, C, H, W)
    return s, img


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate_policy(policy, env, instruction, n_episodes, max_steps, device, debug, action_mode, n_obs_steps, history_mode):
    policy.eval()
    if hasattr(policy, "base_policy"):
        policy.base_policy.eval()

    # PushT often wrapped as vector env even when num_envs=1
    is_vector = True

    successes = []
    rewards = []
    lengths = []

    print("\n" + "=" * 60)
    print("Starting evaluation")
    print(f"  Episodes: {n_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Instruction: '{instruction}'")
    print(f"  N_OBS_STEPS (maintained): {n_obs_steps}")
    print(f"  HISTORY_MODE (fed to policy): {history_mode}")
    print(f"  ACTION_MODE: {action_mode}")
    print("=" * 60 + "\n")

    for ep in range(n_episodes):
        obs, info = env.reset()

        if isinstance(info, dict):
            info = {k: (v[0] if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 else v)
                    for k, v in info.items()}

        if not isinstance(obs, dict):
            raise RuntimeError("Expected dict observation from PushT.")

        state_np, image_np = _extract_state_image(obs)
        if state_np is None or image_np is None:
            raise RuntimeError(f"Obs keys: {list(obs.keys())} (could not find state/image)")

        # history buffers store single-step tensors (no batch)
        state_hist = deque(maxlen=n_obs_steps)
        image_hist = deque(maxlen=n_obs_steps)

        state_t = _to_tensor(state_np, device)
        img_t = _to_tensor(image_np, device)

        # If batched, take first item
        state_item = state_t[0] if state_t.ndim == 2 else state_t  # (D,)
        img_item = img_t[0] if img_t.ndim == 4 else img_t          # (C,H,W)

        for _ in range(n_obs_steps):
            state_hist.append(state_item)
            image_hist.append(img_item)

        ep_reward = 0.0
        ep_success = False

        for step in range(max_steps):
            obs_state, obs_img = _build_policy_inputs(state_hist, image_hist, history_mode)

            batch = {
                "observation.state": obs_state,  # (1,D) or (1,T*D)
                "observation.image": obs_img,    # (1,C,H,W) or (1,T*C,H,W)
            }

            if debug and ep == 0 and step == 0:
                print("First batch shapes (fed to policy):")
                for k, v in batch.items():
                    print(f"  {k}: {tuple(v.shape)} {v.dtype}")

            act = policy.predict_action(batch, language=instruction)
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()

            # Normalize action shapes to (A,)
            if act.ndim == 3:
                act = act[:, 0, :]   # (B,A)
            elif act.ndim == 2 and act.shape[0] > 2:
                act = act[0]         # (A,)
            a = act[0] if act.ndim == 2 else act  # (A,)

            mode = action_mode
            if mode == "auto":
                mode = _infer_action_mode(a)

            a_scaled = _scale_action(a, mode)

            if debug and ep == 0 and step < 5:
                print(f"\n[step {step}] raw={a} range=({a.min():.3f},{a.max():.3f}) mode={mode}")
                print(f"[step {step}] scaled={a_scaled} range=({a_scaled.min():.1f},{a_scaled.max():.1f})")

            a_env = np.expand_dims(a_scaled, axis=0) if is_vector else a_scaled
            obs, reward, terminated, truncated, info = env.step(a_env)

            if isinstance(reward, np.ndarray):
                reward = float(reward[0])
            if isinstance(terminated, np.ndarray):
                terminated = bool(terminated[0])
            if isinstance(truncated, np.ndarray):
                truncated = bool(truncated[0])
            if isinstance(info, dict):
                info = {k: (v[0] if isinstance(v, (list, tuple, np.ndarray)) and hasattr(v, "__len__") and len(v) > 0 else v)
                        for k, v in info.items()}

            ep_reward += float(reward)

            if "success" in info and _first_bool(info["success"]):
                ep_success = True
            if "is_success" in info and _first_bool(info["is_success"]):
                ep_success = True

            done = bool(terminated or truncated)
            if done:
                break

            if not isinstance(obs, dict):
                raise RuntimeError("Expected dict observation from PushT.")

            state_np, image_np = _extract_state_image(obs)
            state_t = _to_tensor(state_np, device)
            img_t = _to_tensor(image_np, device)

            state_item = state_t[0] if state_t.ndim == 2 else state_t
            img_item = img_t[0] if img_t.ndim == 4 else img_t

            state_hist.append(state_item)
            image_hist.append(img_item)

        successes.append(bool(ep_success))
        rewards.append(float(ep_reward))
        lengths.append(int(step + 1))

        status = "✓ SUCCESS" if ep_success else "✗ FAIL"
        print(f"Episode {ep+1:2d}/{n_episodes}: {status} | Steps: {step+1:3d} | Reward: {ep_reward:6.2f}")

    success_rate = float(np.mean(successes) * 100.0)
    avg_reward = float(np.mean(rewards))
    avg_len = float(np.mean(lengths))

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Success Rate:    {success_rate:.1f}% ({sum(successes)}/{n_episodes})")
    print(f"Average Reward:  {avg_reward:.2f}")
    print(f"Average Length:  {avg_len:.1f} steps")
    print("=" * 60 + "\n")

    return {
        "success_rate": success_rate,
        "successes": successes,
        "avg_reward": avg_reward,
        "avg_length": avg_len,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "action_mode": str(action_mode),
        "n_obs_steps": int(n_obs_steps),
        "history_mode": str(history_mode),
    }


def main():
    ckpt_path = Path(os.environ.get("CKPT_EVAL", "outputs/manual_clip_diffusion_pusht/checkpoints/065000/pretrained_model"))
    n_eval = int(os.environ.get("N_EVAL", "20"))
    debug = bool(int(os.environ.get("DEBUG", "0")))
    action_mode = os.environ.get("ACTION_MODE", "auto").strip().lower()
    n_obs_steps = int(os.environ.get("N_OBS_STEPS", "2"))
    history_mode = os.environ.get("HISTORY_MODE", "last").strip().lower()

    if action_mode not in {"auto", "0_1", "neg1_1", "raw"}:
        raise ValueError("ACTION_MODE must be one of: auto | 0_1 | neg1_1 | raw")

    if history_mode not in {"last", "concat"}:
        raise ValueError("HISTORY_MODE must be one of: last | concat")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🎯 Evaluating CLIP-Conditioned Diffusion Policy")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Episodes: {n_eval}")
    print(f"Debug: {debug}")
    print(f"ACTION_MODE: {action_mode}")
    print(f"N_OBS_STEPS: {n_obs_steps}")
    print(f"HISTORY_MODE: {history_mode}\n")

    instruction = "Push the T-shaped block to the target."

    base_policy_path = ckpt_path / "base_policy"
    clip_proj_path = ckpt_path / "clip_projector.pt"

    print("Loading policy components...")
    print(f"  Base policy: {base_policy_path}")
    print(f"  CLIP projector: {clip_proj_path}")

    if not base_policy_path.exists():
        raise FileNotFoundError(f"Base policy not found: {base_policy_path}")
    if not clip_proj_path.exists():
        raise FileNotFoundError(f"CLIP projector not found: {clip_proj_path}")

    # Load base policy
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        base_policy = DiffusionPolicy.from_pretrained(str(base_policy_path))
        print("✓ Base policy loaded (from_pretrained)")
    except Exception as e:
        print(f"  from_pretrained failed: {e}")
        from safetensors.torch import load_file

        policy_config = make_policy_config("diffusion")
        env_config = make_env_config("pusht", task="PushT-v0")
        base_policy = make_policy(policy_config, env_cfg=env_config)

        weights_path = base_policy_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"No model.safetensors in {base_policy_path}")

        state_dict = load_file(str(weights_path))
        base_policy.load_state_dict(state_dict)
        print("✓ Base policy loaded (manual safetensors)")

    policy = CLIPConditionedPolicy(base_policy).to(device)

    projector_state = torch.load(clip_proj_path, map_location=device)
    policy.lang_proj.load_state_dict(projector_state["lang_proj"])

    print("✓ Policy loaded successfully")
    print(f"  CLIP model: {projector_state.get('clip_model_name', 'openai/clip-vit-base-patch32')}\n")

    print("Creating PushT environment...")
    env_cfg = make_env_config("pusht", task="PushT-v0")
    env_bundle = make_env(env_cfg)
    env = unwrap_env_bundle(env_bundle)
    print("✓ Environment created\n")

    results = evaluate_policy(
        policy=policy,
        env=env,
        instruction=instruction,
        n_episodes=n_eval,
        max_steps=300,
        device=device,
        debug=debug,
        action_mode=action_mode,
        n_obs_steps=n_obs_steps,
        history_mode=history_mode,
    )

    results_path = ckpt_path.parent.parent.parent / f"eval_results_{ckpt_path.parent.name}_{action_mode}_T{n_obs_steps}_{history_mode}.json"
    with open(results_path, "w") as f:
        json.dump(_jsonify(results), f, indent=2)

    print(f"✓ Results saved to: {results_path}")
    env.close()


if __name__ == "__main__":
    main()
