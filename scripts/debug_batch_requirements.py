#!/usr/bin/env python3
"""
Debug script to understand what LeRobot's diffusion policy expects in the batch.
Run this to see the exact key names and shapes needed.

Usage:
    python scripts/debug_batch_requirements.py
"""

import torch
from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("=" * 60)
print("Debugging LeRobot Batch Requirements")
print("=" * 60)

# Create policy
print("\n1. Creating policy...")
policy_config = make_policy_config("diffusion")
env_config = make_env_config("pusht", task="PushT-v0")
policy = make_policy(policy_config, env_cfg=env_config)
print("✓ Policy created")

# Load dataset
print("\n2. Loading dataset...")
dataset = LeRobotDataset("lerobot/pusht", video_backend=None)
sample = dataset[0]
print(f"✓ Dataset loaded")
print(f"  Dataset keys: {list(sample.keys())}")

# Check what the policy's source code expects
print("\n3. Checking policy source code for required keys...")
import inspect
source = inspect.getsource(policy.diffusion.compute_loss)
print("Policy compute_loss() source (first 30 lines):")
for i, line in enumerate(source.split('\n')[:30], 1):
    print(f"  {i:2d}: {line}")

# Try to find the constants
print("\n4. Looking for constant definitions...")
try:
    from lerobot.policies.diffusion.modeling_diffusion import OBS_STATE, ACTION
    print(f"  OBS_STATE = '{OBS_STATE}'")
    print(f"  ACTION = '{ACTION}'")
except ImportError as e:
    print(f"  Could not import constants: {e}")
    print("  Trying alternative imports...")
    
    # Try to find where these are defined
    import lerobot.policies.diffusion.modeling_diffusion as diff_module
    for name in dir(diff_module):
        if 'OBS' in name or 'ACTION' in name:
            print(f"    {name} = {getattr(diff_module, name)}")

# Check what batch keys the dataset provides
print("\n5. Dataset batch sample:")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"  '{key}': shape={tuple(value.shape)}, dtype={value.dtype}")
    else:
        print(f"  '{key}': {type(value).__name__}")

# Try a forward pass with the dataset batch
print("\n6. Testing forward pass with dataset batch...")
batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
         for k, v in sample.items()}

try:
    output = policy(batch)
    print("✓ Forward pass succeeded!")
    print(f"  Output keys: {output.keys() if isinstance(output, dict) else type(output)}")
except AssertionError as e:
    print(f"✗ Forward pass failed with AssertionError")
    print(f"  This tells us what keys are missing!")
    
    # Parse the assertion to see what's expected
    print("\n7. Analyzing the assertion error...")
    try:
        # Try to figure out what's missing by checking the assertion
        import traceback
        traceback.print_exc()
    except:
        pass
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("Run this script and share the output.")
print("It will show us exactly what keys the policy needs.")