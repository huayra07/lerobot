# examples/training/train_lang_diffusion_pusht.py
"""
Train a language-conditioned diffusion policy on PushT with CLIP text encoder.
This is the CORRECT way to integrate language conditioning.
"""

import os
from pathlib import Path
from lerobot.common.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy_config, make_policy
from lerobot.envs.factory import make_env_config
import torch

def main():
    # ============================================================================
    # CRITICAL: Language Configuration
    # ============================================================================
    # This is what you were MISSING in your training!
    
    use_language_cond = True
    language_embedding_source = "clip"  # ← MUST be "clip", not "embedding"!
    text_encoder_name = "openai/clip-vit-base-patch32"  # ← Pretrained CLIP model
    language_cond_dim = 512  # ← CLIP outputs 512-dim embeddings
    freeze_text_encoder = True  # ← Don't fine-tune CLIP (saves memory/time)
    
    # ============================================================================
    # Policy Configuration
    # ============================================================================
    
    policy_cfg = make_policy_config(
        "diffusion",
        
        # Model architecture
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        crop_is_random=True,
        
        # Diffusion params
        n_action_steps=8,
        num_inference_steps=10,
        down_dims=(256, 512, 1024),
        
        # ⭐⭐⭐ LANGUAGE CONFIGURATION ⭐⭐⭐
        use_language_cond=use_language_cond,
        language_embedding_source=language_embedding_source,
        text_encoder_name=text_encoder_name,
        language_cond_dim=language_cond_dim,
        freeze_text_encoder=freeze_text_encoder,
        
        # Training
        batch_size=64,
        lr=1e-4,
        lr_scheduler="cosine",
        training_steps=100000,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print("✅ Policy config with CLIP:")
    print(f"  use_language_cond: {policy_cfg.use_language_cond}")
    print(f"  language_embedding_source: {policy_cfg.language_embedding_source}")
    print(f"  text_encoder_name: {policy_cfg.text_encoder_name}")
    print(f"  language_cond_dim: {policy_cfg.language_cond_dim}")
    print(f"  freeze_text_encoder: {policy_cfg.freeze_text_encoder}")
    
    # ============================================================================
    # Dataset with Language Annotations
    # ============================================================================
    
    # You MUST have language annotations in your dataset!
    # For PushT, you might need to add them manually
    dataset = make_dataset("lerobot/pusht")
    
    # Check if dataset has language
    sample = dataset[0]
    if "language" not in sample and "text" not in sample and "task" not in sample:
        print("\n⚠️ WARNING: Dataset has NO language annotations!")
        print("You need to add language instructions to your dataset.")
        print("For PushT, all episodes could use:")
        print('  "push the T-shaped block to the target goal"')
        return
    
    print(f"\n✅ Dataset has {len(dataset)} episodes")
    if "language" in sample:
        print(f"Sample instruction: {sample['language']}")
    
    # ============================================================================
    # Environment
    # ============================================================================
    
    env_cfg = make_env_config("pusht", task="PushT-v0")
    
    # ============================================================================
    # Create Policy
    # ============================================================================
    
    policy = make_policy(policy_cfg, env_cfg=env_cfg, dataset=dataset)
    
    # Check that CLIP was loaded
    if hasattr(policy, "clip_text_encoder"):
        print("\n✅ CLIP text encoder loaded!")
        print(f"  CLIP params: {sum(p.numel() for p in policy.clip_text_encoder.parameters()):,}")
        if freeze_text_encoder:
            print("  CLIP is frozen (not being trained)")
    else:
        print("\n❌ ERROR: No CLIP encoder found in policy!")
        print("Something is wrong with the configuration.")
        return
    
    # ============================================================================
    # Training Loop (simplified)
    # ============================================================================
    
    print(f"\n📊 Training for {policy_cfg.training_steps} steps...")
    print(f"Batch size: {policy_cfg.batch_size}")
    print(f"Learning rate: {policy_cfg.lr}")
    
    # In a real training script, you would:
    # 1. Create DataLoader from dataset
    # 2. Create optimizer
    # 3. Run training loop with policy.forward(batch)
    # 4. Save checkpoints
    
    # For now, just demonstrate that the model can process a batch
    # with language conditioning
    demo_batch = {
        "observation.image": torch.randn(2, 3, 84, 84),
        "observation.state": torch.randn(2, 2),
        "action": torch.randn(2, 8, 2),
        "language": ["push the T-shaped block to the target", 
                     "push the block to the goal position"],
    }
    
    print("\n🧪 Testing forward pass with language...")
    policy.eval()
    with torch.no_grad():
        try:
            output = policy(demo_batch)
            print("✅ Forward pass successful!")
            print(f"  Output keys: {output.keys() if isinstance(output, dict) else type(output)}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
    
    output_dir = Path("outputs") / "lang_diffusion_pusht"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Would save checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()