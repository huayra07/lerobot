#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: scripts/check_lerobot_imports.py
# ==============================================================================
"""
Check your LeRobot installation and find correct import paths.
"""

import sys

print("=" * 70)
print("Checking LeRobot Installation")
print("=" * 70)

# Check LeRobot is installed
try:
    import lerobot
    print("✅ LeRobot installed")
    print(f"   Location: {lerobot.__file__}")
    if hasattr(lerobot, '__version__'):
        print(f"   Version: {lerobot.__version__}")
except ImportError:
    print("❌ LeRobot not installed")
    sys.exit(1)

print()

# Check dataset imports
print("Checking dataset imports...")
dataset_import = None
for attempt in [
    "from lerobot.common.datasets.factory import make_dataset",
    "from lerobot.datasets import make_dataset",
    "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset",
]:
    try:
        exec(attempt)
        print(f"✅ {attempt}")
        dataset_import = attempt
        break
    except ImportError as e:
        print(f"❌ {attempt}")
        print(f"   Error: {e}")

print()

# Check policy imports
print("Checking policy imports...")
try:
    from lerobot.policies.factory import make_policy_config, make_policy
    print("✅ from lerobot.policies.factory import make_policy_config, make_policy")
except ImportError as e:
    print(f"❌ Policy factory import failed: {e}")

try:
    from lerobot.envs.factory import make_env_config, make_env
    print("✅ from lerobot.envs.factory import make_env_config, make_env")
except ImportError as e:
    print(f"❌ Env factory import failed: {e}")

print()

# Check CLIP/transformers
print("Checking CLIP support...")
try:
    from transformers import CLIPTokenizer, CLIPTextModel
    print("✅ CLIP available (transformers installed)")
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print("✅ CLIP tokenizer loads successfully")
except ImportError as e:
    print(f"❌ CLIP not available: {e}")
    print("   Run: pip install transformers")
except Exception as e:
    print(f"⚠️  CLIP import works but loading failed: {e}")

print()

# Check if your eval script works
print("Checking your existing eval script...")
try:
    # Try to import it
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_simple", 
        "examples/training/eval_pusht_SIMPLE_LANG.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Your eval_pusht_SIMPLE_LANG.py imports work")
except Exception as e:
    print(f"⚠️  Could not import your eval script: {e}")

print()
print("=" * 70)
print("Summary")
print("=" * 70)

if dataset_import:
    print(f"✅ Use this for datasets: {dataset_import}")
else:
    print("❌ Dataset import not found - LeRobot may not be properly installed")

print("\n💡 Recommendation:")
print("   Share this output and I'll fix the training script imports.")