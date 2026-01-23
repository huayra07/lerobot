#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: examples/training/eval_pusht_MANUAL_CLIP.py
# ==============================================================================
"""
Evaluate the manually CLIP-integrated diffusion policy.

Usage:
    # Test WITHOUT action scaling (if policy was trained on raw [0,512] actions)
    CKPT_EVAL=outputs/manual_clip_diffusion_pusht/checkpoints/065000/pretrained_model \
    N_EVAL=5 \
    DEBUG=1 \
    SCALE_ACTIONS=0 \
    python examples/training/eval_pusht_MANUAL_CLIP.py
    
    # Test WITH action scaling (if policy was trained on [-1,1] normalized actions)
    CKPT_EVAL=outputs/manual_clip_diffusion_pusht/checkpoints/065000/pretrained_model \
    N_EVAL=5 \
    DEBUG=1 \
    SCALE_ACTIONS=1 \
    python examples/training/eval_pusht_MANUAL_CLIP.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer

from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.envs.factory import make_env, make_env_config


# Copy the CLIPConditionedPolicy class from training
class CLIPConditionedPolicy(nn.Module):
    """Wrapper that adds CLIP text conditioning to any policy."""
    
    def __init__(self, base_policy, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.base_policy = base_policy
        
        # Load CLIP text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Freeze CLIP
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Projection layer
        clip_dim = self.text_encoder.config.hidden_size
        policy_dim = 128
        
        self.lang_proj = nn.Sequential(
            nn.Linear(clip_dim, policy_dim),
            nn.ReLU(),
            nn.Linear(policy_dim, policy_dim)
        )
    
    def encode_text(self, text_list):
        """Encode text using CLIP."""
        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            text_embeds = outputs.pooler_output
        
        lang_features = self.lang_proj(text_embeds)
        return lang_features
    
    def forward(self, batch):
        """Forward pass with language conditioning."""
        if "language" in batch:
            text_list = batch["language"]
            lang_features = self.encode_text(text_list)
            batch["language_embedding"] = lang_features
        
        return self.base_policy(batch)
    
    def predict_action(self, observation, language=None):
        """Predict action (for evaluation)."""
        # Ensure base policy is in eval mode
        self.base_policy.eval()
        
        # Merge observation dict into batch (don't nest it)
        batch = dict(observation)
        
        if language is not None:
            if isinstance(language, str):
                language = [language]
            batch["language"] = language
        
        with torch.no_grad():
            # Call base policy's select_action method if available
            if hasattr(self.base_policy, 'select_action'):
                # Add language embedding if provided
                if "language" in batch:
                    text_list = batch["language"]
                    lang_features = self.encode_text(text_list)
                    batch["language_embedding"] = lang_features
                return self.base_policy.select_action(batch)
            else:
                # Fall back to forward
                output = self.forward(batch)
                if isinstance(output, dict) and "action" in output:
                    return output["action"]
                return output


def unwrap_env_bundle(x):
    """Unwrap environment to get the actual PushT env"""
    seen = set()
    while isinstance(x, dict):
        obj_id = id(x)
        if obj_id in seen:
            break
        seen.add(obj_id)
        
        # Try different possible keys
        for key in ['env', 'pusht', 'environment']:
            if key in x:
                x = x[key]
                break
        else:
            # No known key found, take the first value if there's only one key
            if len(x) == 1:
                x = list(x.values())[0]
            else:
                break
    
    # Handle wrapped envs with .env attribute
    while hasattr(x, 'env'):
        x = x.env
    
    return x


def _first_bool(x):
    """Get first boolean value from an array/tensor."""
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return bool(x.flatten()[0])
    return bool(x)


def evaluate_policy(policy, env, instruction, n_episodes=20, max_steps=300, device="cuda", debug=False, scale_actions=True):
    """
    Evaluate policy on PushT environment.
    
    Args:
        scale_actions: If True, scale actions from [-1,1] to [0,512]. 
                      If False, use raw policy outputs (assumes policy outputs [0,512] already)
    """
    # Ensure policy is in eval mode
    policy.eval()
    if hasattr(policy, 'base_policy'):
        policy.base_policy.eval()
    
    successes = []
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation")
    print(f"  Episodes: {n_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Action scaling: {'ENABLED ([-1,1] -> [0,512])' if scale_actions else 'DISABLED (raw actions)'}")
    if debug:
        print(f"  Policy training mode: {policy.training}")
        if hasattr(policy, 'base_policy'):
            print(f"  Base policy training mode: {policy.base_policy.training}")
            print(f"  Base policy has select_action: {hasattr(policy.base_policy, 'select_action')}")
    print(f"{'='*60}\n")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        # Handle vectorized environment - obs might be batched
        if isinstance(info, dict) and any(isinstance(v, (list, tuple)) for v in info.values()):
            # Info is batched, take first element
            info = {k: v[0] if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 else v 
                    for k, v in info.items()}
        
        episode_reward = 0.0
        episode_success = False
        
        for step in range(max_steps):
            # Prepare observation for policy - match training format
            if isinstance(obs, dict):
                # Build batch dict with proper keys
                batch = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        # Convert to tensor
                        tensor = torch.from_numpy(v).to(device)
                        
                        # Handle different tensor dimensions
                        if tensor.ndim == 1:
                            # State vector: add batch dimension
                            tensor = tensor.unsqueeze(0)  # (state_dim,) -> (1, state_dim)
                            # Convert to float32
                            tensor = tensor.float()
                        elif tensor.ndim == 2:
                            # Already batched state: keep as is
                            tensor = tensor.float()
                        elif tensor.ndim == 3:
                            # Image without batch: HWC -> CHW, then add batch
                            if tensor.shape[-1] in [1, 3, 4]:  # Last dim is likely channels
                                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
                            tensor = tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
                            # Convert uint8 to float32 and normalize to [0, 1]
                            if tensor.dtype == torch.uint8:
                                tensor = tensor.float() / 255.0
                            else:
                                tensor = tensor.float()
                        elif tensor.ndim == 4:
                            # Already batched image: BHWC -> BCHW
                            if tensor.shape[-1] in [1, 3, 4]:  # Last dim is channels
                                tensor = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                            # Convert uint8 to float32 and normalize to [0, 1]
                            if tensor.dtype == torch.uint8:
                                tensor = tensor.float() / 255.0
                            else:
                                tensor = tensor.float()
                        
                        batch[k] = tensor
                    else:
                        batch[k] = v
                
                # Ensure we have the right keys - map from env format to policy format
                obs_tensor = {}
                if 'agent_pos' in batch:
                    obs_tensor['observation.state'] = batch['agent_pos']
                elif 'state' in batch:
                    obs_tensor['observation.state'] = batch['state']
                
                if 'pixels' in batch:
                    obs_tensor['observation.image'] = batch['pixels']
                elif 'image' in batch:
                    obs_tensor['observation.image'] = batch['image']
                
                # If obs already has observation.* keys, use them directly
                for k in batch:
                    if k.startswith('observation.'):
                        obs_tensor[k] = batch[k]
            else:
                obs_tensor = {"observation.state": torch.from_numpy(obs).unsqueeze(0).to(device)}
            
            if debug and ep == 0 and step == 0:
                print(f"  First observation keys: {list(obs_tensor.keys())}")
                for k, v in obs_tensor.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            
            # Get action from policy with language
            with torch.no_grad():
                action = policy.predict_action(obs_tensor, language=instruction)
            
            # Convert action to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Handle action shape from policy
            # Policy returns (B, horizon, action_dim) or (horizon, action_dim)
            # We want just the first action: (B, action_dim) or (action_dim,)
            if action.ndim == 3:
                # Shape: (B, horizon, action_dim) -> take first timestep
                action = action[:, 0, :]  # (B, action_dim)
            elif action.ndim == 2 and action.shape[0] > 2:  # Likely (horizon, action_dim)
                action = action[0]  # (action_dim,)
            
            # Get the actual 2D action vector for processing
            if action.ndim == 2:
                action_1d = action[0]  # Unbatch
            else:
                action_1d = action
            
            # DEBUG: Show raw policy output
            if debug and ep == 0 and step < 5:
                print(f"\n  === Step {step} Action Processing ===")
                print(f"  Raw policy output: {action_1d}")
                print(f"  Raw range: [{action_1d.min():.4f}, {action_1d.max():.4f}]")
            
            # Apply action scaling if enabled
            if scale_actions:
                # Scale from [0, 1] to [0, 512]
                # Policy outputs are in [0, 1] range, need to scale to pixel coordinates
                action_scaled = action_1d * 512.0
                action_scaled = np.clip(action_scaled, 0.0, 512.0)
                if debug and ep == 0 and step < 5:
                    print(f"  After scaling [0,1]->[0,512]: {action_scaled}")
            else:
                # No scaling - use raw policy output
                action_scaled = action_1d.copy()
                if debug and ep == 0 and step < 5:
                    print(f"  No scaling applied (raw): {action_scaled}")
            
            # Handle batch dimension for vectorized environment  
            if action_scaled.ndim == 1:
                action_for_env = np.expand_dims(action_scaled, axis=0)  # (action_dim,) -> (1, action_dim)
            else:
                action_for_env = action_scaled
            
            if debug and ep == 0 and step < 5:
                print(f"  Final action to env: {action_for_env[0]}")
                print(f"  Final range: [{action_for_env.min():.4f}, {action_for_env.max():.4f}]")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_for_env)
            
            # Handle vectorized environment outputs (unbatch them)
            if isinstance(reward, np.ndarray) and reward.ndim > 0:
                reward = reward[0]
            if isinstance(terminated, np.ndarray) and terminated.ndim > 0:
                terminated = terminated[0]
            if isinstance(truncated, np.ndarray) and truncated.ndim > 0:
                truncated = truncated[0]
            
            # Handle info dict from vectorized env
            if isinstance(info, dict):
                info = {k: v[0] if isinstance(v, (list, tuple, np.ndarray)) and hasattr(v, '__len__') and len(v) > 0 else v 
                        for k, v in info.items()}
            
            done = terminated or truncated
            
            episode_reward += float(reward)
            
            # Check for success
            if "success" in info:
                if _first_bool(info["success"]):
                    episode_success = True
            elif "is_success" in info:
                if _first_bool(info["is_success"]):
                    episode_success = True
            
            if debug and ep == 0 and step < 5:
                print(f"  Reward: {reward:.4f}, Done: {done}")
            
            if done:
                break
        
        successes.append(episode_success)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        status = "✓ SUCCESS" if episode_success else "✗ FAIL"
        print(f"Episode {ep+1:2d}/{n_episodes}: {status} | Steps: {step+1:3d} | Reward: {episode_reward:6.2f}")
    
    # Compute statistics
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Success Rate:    {success_rate:.1f}% ({sum(successes)}/{n_episodes})")
    print(f"Average Reward:  {avg_reward:.2f}")
    print(f"Average Length:  {avg_length:.1f} steps")
    print(f"{'='*60}\n")
    
    return {
        "success_rate": success_rate,
        "successes": successes,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    # Configuration from environment variables
    ckpt_path = os.environ.get("CKPT_EVAL", "outputs/manual_clip_diffusion_pusht/checkpoints/065000/pretrained_model")
    n_eval = int(os.environ.get("N_EVAL", "20"))
    debug = bool(int(os.environ.get("DEBUG", "0")))
    scale_actions = bool(int(os.environ.get("SCALE_ACTIONS", "1")))  # Default: scale actions
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🎯 Evaluating CLIP-Conditioned Diffusion Policy")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Episodes: {n_eval}")
    print(f"Debug: {debug}")
    print(f"Scale Actions: {scale_actions}\n")
    
    # The instruction used during training
    instruction = "Push the T-shaped block to the target."
    
    # Load base policy
    ckpt_path = Path(ckpt_path)
    base_policy_path = ckpt_path / "base_policy"
    clip_proj_path = ckpt_path / "clip_projector.pt"
    
    print(f"Loading policy components...")
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
        print(f"✓ Base policy loaded (from_pretrained)")
    except Exception as e:
        print(f"  from_pretrained failed: {e}")
        try:
            from safetensors.torch import load_file
            import json
            
            config_path = base_policy_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                print(f"  Found config.json")
            
            policy_config = make_policy_config("diffusion")
            env_config = make_env_config("pusht", task="PushT-v0")
            base_policy = make_policy(policy_config, env_cfg=env_config)
            
            weights_path = base_policy_path / "model.safetensors"
            if weights_path.exists():
                state_dict = load_file(str(weights_path))
                base_policy.load_state_dict(state_dict)
                print(f"✓ Base policy loaded (manual)")
            else:
                raise FileNotFoundError(f"No model.safetensors in {base_policy_path}")
        except Exception as e2:
            print(f"  Manual loading also failed: {e2}")
            raise RuntimeError(f"Could not load base policy from {base_policy_path}")
    
    # Wrap with CLIP conditioning
    policy = CLIPConditionedPolicy(base_policy)
    policy = policy.to(device)
    
    # Load CLIP projector weights
    projector_state = torch.load(clip_proj_path, map_location=device)
    policy.lang_proj.load_state_dict(projector_state['lang_proj'])
    
    print(f"✓ Policy loaded successfully")
    print(f"  CLIP model: {projector_state.get('clip_model_name', 'openai/clip-vit-base-patch32')}\n")
    
    # Create environment
    print("Creating PushT environment...")
    env_cfg = make_env_config("pusht", task="PushT-v0")
    env_bundle = make_env(env_cfg)
    
    if debug:
        print(f"  env_bundle type: {type(env_bundle)}")
        if isinstance(env_bundle, dict):
            print(f"  env_bundle keys: {env_bundle.keys()}")
    
    env = unwrap_env_bundle(env_bundle)
    
    if debug:
        print(f"  unwrapped env type: {type(env)}")
        if hasattr(env, 'num_envs'):
            print(f"  num_envs: {env.num_envs}")
    
    print(f"✓ Environment created\n")
    
    # Evaluate
    results = evaluate_policy(
        policy=policy,
        env=env,
        instruction=instruction,
        n_episodes=n_eval,
        max_steps=300,
        device=device,
        debug=debug,
        scale_actions=scale_actions,
    )
    
    # Save results
    scaling_suffix = "scaled" if scale_actions else "raw"
    results_path = ckpt_path.parent.parent.parent / f"eval_results_{ckpt_path.parent.name}_{scaling_suffix}.json"
    with open(results_path, "w") as f:
        json_results = {
            k: (v.tolist() if isinstance(v, np.ndarray) else 
                [x.tolist() if isinstance(x, np.ndarray) else x for x in v] if isinstance(v, list) else v)
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    
    env.close()
    return results


if __name__ == "__main__":
    main()