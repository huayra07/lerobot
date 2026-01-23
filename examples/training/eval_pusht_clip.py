# ==============================================================================
# FILE PATH: examples/training/eval_pusht_clip.py
# ==============================================================================
"""
Eval CLIP-based language-conditioned diffusion policy on PushT.
Built on your existing eval_pusht_SIMPLE_LANG.py style.

Usage:
    # Evaluate specific checkpoint
    CKPT_EVAL=outputs/clip_lang_diffusion_pusht/checkpoints/050000 python eval_pusht_clip.py
    
    # Different instruction
    INSTRUCTION="move the block to the goal" python eval_pusht_clip.py
    
    # More episodes
    N_EVAL=50 python eval_pusht_clip.py
"""
import os
import json
import numpy as np
import torch
from pathlib import Path

from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config, make_env


def unwrap_env_bundle(x):
    """Unwrap environment to get the actual PushT env"""
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
    """Extract first boolean from various types"""
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
    """
    Roll out one episode with CLIP language conditioning.
    
    Returns:
        success (bool): Whether episode succeeded
    """
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

        # Image observation
        if "pixels" in obs:
            img = obs["pixels"]
            if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).float()
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            # Resize if needed
            if tuple(img.shape[1:]) != (384, 384):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False
                )
            else:
                img = img.unsqueeze(0)
            batch["observation.image"] = img.to(device)

        # State observation (agent position)
        if "agent_pos" in obs:
            state = obs["agent_pos"]
            if isinstance(state, np.ndarray) and state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            state = torch.from_numpy(state).float().unsqueeze(0)
            batch["observation.state"] = state.to(device)

        # Language - CLIP expects a string (will tokenize internally)
        batch["language"] = language_text

        # Get action from policy
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

        # Step environment
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

    # Debug: print info structure when episode ends
    if debug and done:
        print(f"\n🔍 DEBUG: Episode ended at step {step}")
        print(f"  done={done}, info type={type(info)}")
        if isinstance(info, dict):
            print(f"  info keys: {list(info.keys())}")
            for k, v in info.items():
                print(f"    {k}: {type(v)} = {v}")
        print()

    # Extract success from info
    success = False
    if isinstance(info, dict):
        if "is_success" in info:
            success = _first_bool(info["is_success"])
        elif "success" in info:
            success = _first_bool(info["success"])
        elif "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, dict) and "is_success" in fi:
                success = _first_bool(fi["is_success"])
        
        # PushT specific: sometimes success is in dist_to_goal
        if not success and "dist_to_goal" in info:
            success = info["dist_to_goal"] < 0.02

    return bool(success)


def main():
    # Configuration from environment variables
    ckpt = os.environ.get(
        "CKPT_EVAL", 
        "outputs/clip_lang_diffusion_pusht/checkpoints/100000/pretrained_model"
    )
    
    # Load checkpoint config to get training instruction
    config_path = Path(ckpt) / "config.json"
    training_instruction = "Push the T-shaped block to the target."  # default
    
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        # Extract language config
        print("Checkpoint language config:")
        if "language_text" in saved_config:
            training_instruction = saved_config["language_text"]
            print(f"  language_text: {training_instruction}")
        if "use_language_cond" in saved_config:
            print(f"  use_language_cond: {saved_config['use_language_cond']}")
        if "language_cond_dim" in saved_config:
            print(f"  language_cond_dim: {saved_config['language_cond_dim']}")
        if "language_embedding_source" in saved_config:
            print(f"  language_embedding_source: {saved_config['language_embedding_source']}")
        if "text_encoder_name" in saved_config:
            print(f"  text_encoder_name: {saved_config['text_encoder_name']}")
        print()
    
    # Allow override via environment variable
    instruction = os.environ.get("INSTRUCTION", training_instruction)
    n_eval = int(os.environ.get("N_EVAL", "20"))
    debug = os.environ.get("DEBUG", "0") == "1"
    
    print(f"Checkpoint: {ckpt}")
    print(f"Instruction: '{instruction}'")
    print(f"Episodes: {n_eval}")
    if debug:
        print("Debug mode: ON")
    print()
    
    # Create policy config - load from checkpoint
    print("Creating policy config...")
    cfg = make_policy_config(
        "diffusion",
        pretrained_path=ckpt,
        use_language_cond=True,
        language_cond_dim=512,  # CLIP uses 512
        language_embedding_source="clip",
        text_encoder_name="openai/clip-vit-base-patch32",
    )

    print(f"✓ Created policy config:")
    print(f"  use_language_cond={cfg.use_language_cond}")
    print(f"  language_cond_dim={cfg.language_cond_dim}")
    print(f"  language_embedding_source={cfg.language_embedding_source}")
    print(f"  text_encoder_name={cfg.text_encoder_name}")
    print()

    # Create environment
    print("Creating environment...")
    env_cfg = make_env_config("pusht", task="PushT-v0")
    env_bundle = make_env(env_cfg)
    env = unwrap_env_bundle(env_bundle)
    print("✓ Environment ready")
    print()

    # Create policy
    print("Loading policy...")
    policy = make_policy(cfg, env_cfg=env_cfg)
    policy.eval()
    
    device = next(policy.parameters()).device
    print(f"Device: {device}")
    
    # Verify CLIP is loaded
    state_dict = policy.state_dict()
    clip_keys = [k for k in state_dict.keys() 
                 if "clip" in k.lower() or "text_encoder" in k.lower()]
    print(f"CLIP parameters found: {len(clip_keys)}")
    if len(clip_keys) > 0:
        print(f"  Sample: {clip_keys[0]}")
    
    if len(clip_keys) < 10:
        print("\n⚠️  WARNING: Expected 100+ CLIP params, found", len(clip_keys))
        print("   This checkpoint may not be properly CLIP-conditioned!")
    print()

    # Run evaluation
    print("=" * 70)
    print("Running evaluation...")
    print("=" * 70)
    
    successes = []
    for i in range(n_eval):
        success = rollout_one(env, policy, instruction, debug=(debug and i == 0))
        successes.append(success)
        status = "✓" if success else "✗"
        print(f"ep {i:03d} text='{instruction}' success={status}")

    # Results
    sr = float(np.mean(successes)) if successes else 0.0
    print()
    print(f"{'='*70}")
    print(f"Success rate over {n_eval} eps: {sr:.3f} ({sr*100:.1f}%)")
    print(f"{'='*70}")
    
    # Interpretation
    print("\n📊 Interpretation:")
    if sr >= 0.80:
        print("🎉 EXCELLENT - Model learned the task well!")
    elif sr >= 0.50:
        print("👍 GOOD - Model is learning, may benefit from more training")
    elif sr >= 0.20:
        print("📈 MODERATE - Model shows some learning, needs more training")
    elif sr > 0:
        print("⚠️  POOR - Model barely learned")
        print("   → Try training for more steps (currently trained for how many?)")
        print("   → Verify instruction matches training")
    else:
        print("❌ FAILED - Model didn't learn")
        print("   Possible issues:")
        print("   1. Not enough training steps (need 50k-100k)")
        print("   2. Wrong instruction (must match training exactly)")
        print("   3. CLIP not properly integrated")
        print("   4. Architecture mismatch between train/eval")
        print(f"\n   Try debug mode: DEBUG=1 N_EVAL=1 CKPT_EVAL={ckpt} python {__file__}")


if __name__ == "__main__":
    main()