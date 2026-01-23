# examples/training/eval_pusht_NO_LANG.py
"""
Test if model works WITHOUT language conditioning.
This helps diagnose if the base model learned the task at all.
"""
import os
import json
import numpy as np
import torch
from pathlib import Path

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config, make_env


def unwrap_env_bundle(x):
    seen = set()
    while isinstance(x, dict):
        obj_id = id(x)
        if obj_id in seen:
            break
        seen.add(obj_id)
        for k in ("env", "gym_env", "wrapped_env", "pusht"):
            if k in x and x[k] is not None:
                x = x[k]
                break
        else:
            if len(x) == 1:
                x = next(iter(x.values()))
            else:
                break
    return x


def _first_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return bool(x.reshape(-1)[0]) if x.size else False
    if isinstance(x, (list, tuple)):
        return bool(x[0]) if len(x) else False
    return bool(x)


def rollout_one(env, policy, max_steps: int = 300):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out

    if hasattr(policy, "reset") and callable(getattr(policy, "reset")):
        try:
            policy.reset()
        except Exception:
            pass

    device = next(policy.parameters()).device
    done = False
    step = 0
    info = {}

    while not done and step < max_steps:
        batch = {}

        # image
        if "pixels" in obs:
            img = obs["pixels"]
            if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).float()
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            if tuple(img.shape[1:]) != (384, 384):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False
                )
            else:
                img = img.unsqueeze(0)
            batch["observation.image"] = img.to(device)

        # state
        if "agent_pos" in obs:
            state = obs["agent_pos"]
            if isinstance(state, np.ndarray) and state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            state = torch.from_numpy(state).float().unsqueeze(0)
            batch["observation.state"] = state.to(device)

        # NO LANGUAGE - testing baseline

        with torch.no_grad():
            if hasattr(policy, "select_action"):
                action = policy.select_action(batch)
            elif hasattr(policy, "act"):
                action = policy.act(batch)
            else:
                outp = policy(batch)
                action = outp["action"] if isinstance(outp, dict) and "action" in outp else outp

            if isinstance(action, dict) and "action" in action:
                action = action["action"]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 1:
                action = action[None, :]

        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = _first_bool(terminated) or _first_bool(truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done_flag, info = step_out
            done = _first_bool(done_flag)
        else:
            raise TypeError(f"Unexpected step output: {type(step_out)}")
        step += 1

    # Extract success
    success = False
    if isinstance(info, dict):
        if "is_success" in info:
            success = _first_bool(info["is_success"])
        elif "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, dict) and "is_success" in fi:
                success = _first_bool(fi["is_success"])

    return bool(success)


def main():
    ckpt = os.environ.get("CKPT_EVAL", "outputs/lang_v1_ft_5k_from_100k/checkpoints/005000/pretrained_model")
    
    print(f"Loading checkpoint: {ckpt}")
    
    # FORCE language OFF to test base model
    cfg = make_policy_config(
        "diffusion",
        pretrained_path=ckpt,
        use_language_cond=False,  # ← DISABLE LANGUAGE
    )

    print(f"✓ Created policy config (NO LANGUAGE):")
    print(f"  use_language_cond={cfg.use_language_cond}")
    print()

    env_cfg = make_env_config("pusht", task="PushT-v0")
    env_bundle = make_env(env_cfg)
    env = unwrap_env_bundle(env_bundle)

    policy = make_policy(cfg, env_cfg=env_cfg)
    policy.eval()

    print("Device:", next(policy.parameters()).device)
    print()

    N = int(os.environ.get("N_EVAL", "20"))
    successes = []
    
    print(f"Running {N} episodes WITHOUT language conditioning...")
    for i in range(N):
        ok = rollout_one(env, policy)
        successes.append(ok)
        print(f"ep {i:03d} success={ok}")

    sr = float(np.mean(successes)) if successes else 0.0
    print(f"\n{'='*60}")
    print(f"Success rate over {N} eps (NO LANGUAGE): {sr:.3f} ({sr*100:.1f}%)")
    print(f"{'='*60}")
    
    if sr > 0.0:
        print("\n✅ Model learned something! It can solve the task without language.")
        print("   This means your base training worked.")
    else:
        print("\n❌ Model failed completely. Base training might have issues.")
        print("   Try training longer or check hyperparameters.")


if __name__ == "__main__":
    main()