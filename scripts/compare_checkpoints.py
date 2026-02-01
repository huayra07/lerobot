#!/usr/bin/env python3
# ==============================================================================
# FILE PATH: scripts/compare_checkpoints.py
# ==============================================================================
"""
Compare your old checkpoint (simple embedding) vs new CLIP checkpoint.
This helps you understand what's different and which to use.

Usage:
    python scripts/compare_checkpoints.py
"""

import os
from pathlib import Path
from safetensors import safe_open

def analyze_checkpoint(ckpt_path: str):
    """Analyze a checkpoint and report what language conditioning it uses"""
    ckpt = Path(ckpt_path)
    
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {ckpt}")
    print(f"{'='*70}")
    
    # Find weight files
    safetensors_files = list(ckpt.glob("*.safetensors"))
    
    if not safetensors_files:
        print("❌ No safetensors files found")
        return None
    
    # Load main model file
    model_file = None
    for f in safetensors_files:
        if "model.safetensors" in f.name and "processor" not in f.name:
            model_file = f
            break
    
    if not model_file:
        model_file = safetensors_files[0]
    
    print(f"Loading: {model_file.name}\n")
    
    results = {
        "path": str(ckpt),
        "type": "unknown",
        "total_params": 0,
        "lang_params": 0,
        "clip_params": 0,
    }
    
    with safe_open(model_file, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        results["total_params"] = len(all_keys)
        
        # Find language-related keys
        lang_keys = [k for k in all_keys if any(x in k.lower() for x in 
                    ["language", "clip", "text_encoder", "token", "embed"])]
        results["lang_params"] = len(lang_keys)
        
        # Find specifically CLIP keys
        clip_keys = [k for k in all_keys if any(x in k.lower() for x in 
                    ["clip", "text_encoder", "text_model"])]
        results["clip_params"] = len(clip_keys)
        
        print(f"📊 Parameter Statistics:")
        print(f"  Total parameters: {results['total_params']}")
        print(f"  Language-related: {results['lang_params']}")
        print(f"  CLIP-specific: {results['clip_params']}")
        print()
        
        # Determine type
        if results["clip_params"] > 100:
            results["type"] = "CLIP"
            print("✅ Type: CLIP-based language conditioning")
            print("   This checkpoint uses a real CLIP text encoder")
            print(f"   Found {results['clip_params']} CLIP parameters")
            print()
            print("   Sample CLIP parameters:")
            for k in sorted(clip_keys)[:10]:
                tensor = f.get_tensor(k)
                print(f"     {k}: {tuple(tensor.shape)}")
        
        elif results["lang_params"] == 1:
            results["type"] = "Simple"
            print("⚠️  Type: Simple language embedding")
            print("   This checkpoint uses a single learned vector")
            print("   NOT a real text encoder - just 1 parameter:")
            for k in lang_keys:
                tensor = f.get_tensor(k)
                print(f"     {k}: {tuple(tensor.shape)}")
            print()
            print("   ⚠️  This won't understand different instructions!")
        
        elif results["lang_params"] == 0:
            results["type"] = "No Language"
            print("❌ Type: No language conditioning")
            print("   This is a vision-only policy")
        
        else:
            results["type"] = "Unknown"
            print(f"❓ Type: Unknown language conditioning")
            print(f"   Found {results['lang_params']} language parameters:")
            for k in lang_keys[:20]:
                tensor = f.get_tensor(k)
                print(f"     {k}: {tuple(tensor.shape)}")
    
    print(f"{'='*70}\n")
    return results


def main():
    print("\n🔍 Checkpoint Comparison Tool")
    print("This helps you understand what language conditioning your checkpoints have.\n")
    
    # Define checkpoints to compare
    checkpoints = [
        # Your old checkpoint (simple embedding)
        "outputs/lang_v1_ft_5k_from_100k/checkpoints/005000/pretrained_model",
        
        # New CLIP checkpoints (if they exist)
        "outputs/clip_lang_diffusion_pusht/checkpoints/050000/pretrained_model",
        "outputs/clip_lang_diffusion_pusht/checkpoints/100000/pretrained_model",
    ]
    
    # Allow custom checkpoints via env var
    custom = os.environ.get("CKPT_PATH")
    if custom:
        checkpoints = [custom] + checkpoints
    
    results = []
    for ckpt in checkpoints:
        result = analyze_checkpoint(ckpt)
        if result:
            results.append(result)
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("📊 SUMMARY COMPARISON")
        print("="*70)
        print(f"{'Checkpoint':<50} {'Type':<15} {'CLIP Params':<12}")
        print("-"*70)
        for r in results:
            ckpt_name = Path(r['path']).parent.parent.name + "/" + Path(r['path']).parent.name
            print(f"{ckpt_name:<50} {r['type']:<15} {r['clip_params']:<12}")
        print("="*70)
        
        print("\n💡 Recommendations:")
        
        clip_ckpts = [r for r in results if r['type'] == 'CLIP']
        simple_ckpts = [r for r in results if r['type'] == 'Simple']
        
        if clip_ckpts:
            print("\n✅ You have CLIP checkpoints! Use these for evaluation:")
            for r in clip_ckpts:
                print(f"   CKPT_EVAL={r['path']} python examples/training/eval_pusht_clip.py")
        
        if simple_ckpts:
            print("\n⚠️  Simple embedding checkpoints found:")
            print("   These won't understand different instructions")
            print("   Only useful if you want to test the base model without language")
            for r in simple_ckpts:
                print(f"   CKPT_EVAL={r['path']} python examples/training/eval_pusht_SIMPLE_LANG.py")
        
        if not clip_ckpts:
            print("\n❌ No CLIP checkpoints found yet!")
            print("   You need to train with the new CLIP training script:")
            print("   python examples/training/train_lang_diffusion_pusht_clip.py")
    
    elif len(results) == 1:
        r = results[0]
        print(f"\n💡 Next Steps:")
        if r['type'] == 'CLIP':
            print("✅ This is a CLIP checkpoint - evaluate with:")
            print(f"   CKPT_EVAL={r['path']} N_EVAL=20 python examples/training/eval_pusht_clip.py")
        elif r['type'] == 'Simple':
            print("⚠️  This is a simple embedding - it won't understand different instructions")
            print("   Recommend training a CLIP version instead:")
            print("   python examples/training/train_lang_diffusion_pusht_clip.py")
        else:
            print("❓ Checkpoint type unclear - may need manual inspection")


if __name__ == "__main__":
    main()