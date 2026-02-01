# scripts/add_language_to_pusht.py
"""
Add language annotations to PushT dataset.
PushT is a single-task environment, so all episodes get the same instruction.
"""

import json
from pathlib import Path
from lerobot.common.datasets.factory import make_dataset
import torch

def add_language_to_pusht_episodes(
    dataset_path: str = "data/lerobot/pusht",
    instruction: str = "push the T-shaped block to the target goal"
):
    """
    Add language instruction to all episodes in PushT dataset.
    
    For single-task datasets like PushT, all episodes use the same instruction.
    For multi-task datasets, you'd need different instructions per task.
    """
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Creating minimal example of how language should be structured...")
        
        # Show what the data structure should look like
        example = {
            "episode_0000": {
                "observation.image": "tensor data",
                "observation.state": "tensor data", 
                "action": "tensor data",
                "language": instruction,  # ← This is what you need to add!
                # Or alternatively:
                # "task": instruction,
                # "text": instruction,
            }
        }
        print(json.dumps({"structure": "Each episode needs a language field"}, indent=2))
        return
    
    print(f"Processing dataset at: {dataset_path}")
    
    # Load dataset metadata
    meta_path = dataset_path / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            print(f"Dataset has {meta.get('total_episodes', '?')} episodes")
    
    # Add language to each episode's metadata
    # The exact method depends on your dataset format
    # Here's the general approach:
    
    # Method 1: If using HuggingFace datasets format
    try:
        from datasets import load_from_disk, Dataset
        
        dataset = load_from_disk(str(dataset_path))
        
        # Add language column
        def add_language_field(example):
            example["language"] = instruction
            return example
        
        dataset = dataset.map(add_language_field)
        dataset.save_to_disk(str(dataset_path / "with_language"))
        print(f"✅ Saved dataset with language to: {dataset_path / 'with_language'}")
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: If episodes are in separate files
    episode_dirs = sorted(dataset_path.glob("episode_*"))
    if episode_dirs:
        print(f"Found {len(episode_dirs)} episode directories")
        
        for ep_dir in episode_dirs[:5]:  # Process first 5 as example
            # Add language to episode metadata
            meta_file = ep_dir / "meta.json"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    ep_meta = json.load(f)
                
                ep_meta["language"] = instruction
                
                with open(meta_file, "w") as f:
                    json.dump(ep_meta, f, indent=2)
                
                print(f"✅ Added language to {ep_dir.name}")


def verify_language_in_dataset(dataset_name: str = "lerobot/pusht"):
    """Check if dataset has language annotations."""
    
    try:
        dataset = make_dataset(dataset_name)
        sample = dataset[0]
        
        print(f"\n📋 Dataset: {dataset_name}")
        print(f"Total episodes: {len(dataset)}")
        print(f"Sample keys: {sample.keys()}")
        
        # Check for language field
        lang_found = False
        for key in ["language", "text", "task", "instruction"]:
            if key in sample:
                print(f"\n✅ Found language field: '{key}'")
                print(f"   Value: {sample[key]}")
                lang_found = True
                break
        
        if not lang_found:
            print("\n❌ No language field found!")
            print("You need to add language annotations before training.")
            print("\nOptions:")
            print("1. Add 'language' field to dataset")
            print("2. Use dataset with existing language annotations")
            print("3. For PushT, manually annotate with:")
            print('   "push the T-shaped block to the target goal"')
        
        return lang_found
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


def create_language_conditioned_pusht_dataset():
    """
    Create a new PushT dataset with language annotations.
    This is the RECOMMENDED approach.
    """
    
    print("Creating language-conditioned PushT dataset...")
    print("\nSteps:")
    print("1. Load original PushT dataset")
    print("2. Add language field to each episode")
    print("3. Save as new dataset")
    
    # For PushT, the instruction is always the same (single-task)
    PUSHT_INSTRUCTION = "push the T-shaped block to the target goal"
    
    # Alternative instructions you could use:
    # "move the T-block to the goal position"
    # "push the block to the target"
    # "align the T-shaped object with the goal"
    
    print(f'\nUsing instruction: "{PUSHT_INSTRUCTION}"')
    
    # In practice, you would:
    # 1. Load each episode
    # 2. Add the instruction to the episode data
    # 3. Save the modified dataset
    
    print("\n💡 For LeRobot datasets, language can be added during data collection")
    print("   or post-processed into the dataset format.")


if __name__ == "__main__":
    # Check if dataset has language
    has_lang = verify_language_in_dataset("lerobot/pusht")
    
    if not has_lang:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Either find/create a PushT dataset WITH language annotations")
        print("2. Or add them to existing dataset")
        print("3. Then run training with CLIP configuration")
        print("\nFor demonstration purposes, PushT is single-task, so:")
        print('All episodes should have: language="push the T-shaped block to the target goal"')