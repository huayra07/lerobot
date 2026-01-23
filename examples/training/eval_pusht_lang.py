# examples/training/eval_pusht_lang.py
import os
import json
import numpy as np
import torch
from pathlib import Path

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config, make_env


def unwrap_env_bundle(x):
    """
    LeRobot make_env sometimes returns nested dict bundles.
    Keep unwrapping until we get something that has .reset().
    """
    seen = set()
    while isinstance(x, dict):
        obj_id = id(x)
        if obj_id in seen:
            break
        seen.add(obj_id)

        # try common keys first
        for k in ("env", "gym_env", "wrapped_env", "pusht"):
            if k in x and x[k] is not None:
                x = x[k]
                break
        else:
            # if only one value, unwrap it
            if len(x) == 1:
                x = next(iter(x.values()))
            else:
                break
    return x


def _first_bool(x):
    """Vector-env friendly bool extraction (terminated/truncated often arrays)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return bool(x.reshape(-1)[0]) if x.size else False
    if isinstance(x, (list, tuple)):
        return bool(x[0]) if len(x) else False
    return bool(x)


def rollout_one(env, policy, language_text: str, max_steps: int = 300, debug: bool = False):
    # reset() can return obs OR (obs, info)
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
    else:
        obs, _info = out, {}

    # IMPORTANT: diffusion policies often keep internal history buffers
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

        # -------- image: env gives pixels (B,H,W,C) uint8, policy wants (B,C,384,384) float --------
        if "pixels" in obs:
            img = obs["pixels"]  # usually np.ndarray
            if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[0] == 1:
                img = img[0]  # (H,W,C)

            if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

            img = torch.from_numpy(img).float()  # (H,W,C) or (C,H,W)
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)  # (C,H,W)
            elif img.dim() == 2:
                img = img.unsqueeze(0)

            # resize to (384,384) and add batch dim
            if tuple(img.shape[1:]) != (384, 384):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(384, 384),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                img = img.unsqueeze(0)

            batch["observation.image"] = img.to(device)

        elif "observation.image" in obs:
            v = obs["observation.image"]
            batch["observation.image"] = v.to(device) if isinstance(v, torch.Tensor) else v

        # -------- state: env gives agent_pos (B,2), policy wants (B,2) float --------
        if "agent_pos" in obs:
            state = obs["agent_pos"]
            if isinstance(state, np.ndarray) and state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            state = torch.from_numpy(state).float().unsqueeze(0)  # (1,2)
            batch["observation.state"] = state.to(device)

        elif "observation.state" in obs:
            v = obs["observation.state"]
            batch["observation.state"] = v.to(device) if isinstance(v, torch.Tensor) else v

        # -------- language: keep as string; CLIP wrapper handles tokenization --------
        batch["language"] = language_text

        # -------- action --------
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

            # SyncVectorEnv expects (B,A). If policy gave (A,), add batch dim.
            if action.ndim == 1:
                action = action[None, :]

            if debug and step == 0:
                print(f"  action shape: {action.shape}, env type: {type(env).__name__}")

        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = _first_bool(terminated) or _first_bool(truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done_flag, info = step_out
            done = _first_bool(done_flag)
        else:
            raise TypeError(f"env.step returned unexpected type/len: {type(step_out)} {step_out}")

        step += 1

    # DEBUG: Print info structure when episode ends
    if debug and done:
        print(f"\n🔍 DEBUG: Episode ended at step {step}")
        print(f"  done={done}, info type={type(info)}")
        if isinstance(info, dict):
            print(f"  info keys: {list(info.keys())}")
            for k, v in info.items():
                if isinstance(v, (list, tuple, dict)):
                    print(f"    {k}: {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else '?'})")
                    if isinstance(v, dict):
                        print(f"      sub-keys: {list(v.keys())}")
                    elif isinstance(v, (list, tuple)) and len(v) > 0:
                        print(f"      first element type: {type(v[0])}")
                        if isinstance(v[0], dict):
                            print(f"      first element keys: {list(v[0].keys())}")
                else:
                    print(f"    {k}: {type(v).__name__} = {v}")
        print()

    # success signals vary a lot; try common keys
    success = False
    if isinstance(info, dict):
        for k in ("success", "task_success", "is_success"):
            if k in info:
                success = _first_bool(info[k])
                break
        # gymnasium vector env sometimes stores final_info
        if not success and "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, (list, tuple)) and len(fi) and isinstance(fi[0], dict):
                for k in ("success", "task_success", "is_success"):
                    if k in fi[0]:
                        success = _first_bool(fi[0][k])
                        break

    return bool(success)


def main():
    ckpt = os.environ.get("CKPT_EVAL", "outputs/train/example_pusht_diffusion")
    ckpt = str(ckpt)

    # ---- read checkpoint config.json (if present) ----
    config_path = Path(ckpt) / "config.json"
    saved = {}
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            saved = json.load(f)

        print("\nCheckpoint language config:")
        for key in ("use_language_cond", "language_cond_dim", "language_embedding_source", "text_encoder_name"):
            print(f"  {key}: {saved.get(key, 'NOT FOUND')}")
        print()

    # ---- IMPORTANT FIX: do NOT use language_embedding_source='embedding' (unsupported) ----
    # If checkpoint doesn't specify these, default to the CLIP path that your codebase supports.
    use_language_cond = bool(saved.get("use_language_cond", True))
    language_cond_dim = int(saved.get("language_cond_dim", 128))

    # Force a supported source. (Your error came from setting this to "embedding".)
    language_embedding_source = saved.get("language_embedding_source", None)
    if language_embedding_source in (None, "NOT FOUND", "embedding"):
        language_embedding_source = "clip"

    text_encoder_name = saved.get("text_encoder_name", None)
    if not text_encoder_name or text_encoder_name == "NOT FOUND":
        text_encoder_name = "openai/clip-vit-base-patch32"

    cfg = make_policy_config(
        "diffusion",
        pretrained_path=ckpt,
        use_language_cond=use_language_cond,
        language_cond_dim=language_cond_dim,
        language_embedding_source=language_embedding_source,
        text_encoder_name=text_encoder_name,
        freeze_text_encoder=True,
    )

    print(f"✓ Created policy config:")
    print(f"  use_language_cond={cfg.use_language_cond}")
    print(f"  language_cond_dim={cfg.language_cond_dim}")
    print(f"  language_embedding_source={cfg.language_embedding_source}")
    print(f"  text_encoder_name={getattr(cfg, 'text_encoder_name', None)}")
    print()

    # env
    env_cfg = make_env_config("pusht", task="PushT-v0")
    env_bundle = make_env(env_cfg)
    env = unwrap_env_bundle(env_bundle)
    if not hasattr(env, "reset"):
        raise TypeError(f"Unwrapped env is still not an env. Got type={type(env)} repr={repr(env)[:300]}")

    # policy
    policy = make_policy(cfg, env_cfg=env_cfg)
    policy.eval()
    if hasattr(policy, "diffusion"):
        policy.diffusion.eval()

    device = next(policy.parameters()).device
    print("Loaded ckpt:", ckpt)
    print("Device:", device)
    print("Has select_action:", hasattr(policy, "select_action"))
    print("Policy training mode:", policy.training)
    if hasattr(policy, "diffusion"):
        print("Diffusion training mode:", policy.diffusion.training)

    # quick env sanity
    test_out = env.reset()
    test_obs = test_out[0] if isinstance(test_out, tuple) else test_out
    if isinstance(test_obs, dict):
        print("\nEnvironment observation keys:", list(test_obs.keys()))
        if "pixels" in test_obs:
            print("  pixels:", getattr(test_obs["pixels"], "shape", None), getattr(test_obs["pixels"], "dtype", None))
        if "agent_pos" in test_obs:
            print("  agent_pos:", getattr(test_obs["agent_pos"], "shape", None))
    print()

    # evaluation
    N = int(os.environ.get("N_EVAL", "20"))
    # IMPORTANT: Use actual PushT instruction, not meaningless placeholders
    # Common PushT instructions: "push the T-shaped block to the target goal"
    instruction = os.environ.get("PUSHT_INSTRUCTION", "push the T-shaped block to the target goal")
    successes = []
    for i in range(N):
        ok = rollout_one(env, policy, instruction, debug=(i == 0))
        successes.append(ok)
        print(f"ep {i:03d} success={ok}")

    sr = float(np.mean(successes)) if successes else 0.0
    print(f"\nSuccess rate over {N} eps: {sr:.3f}")


if __name__ == "__main__":
    main()