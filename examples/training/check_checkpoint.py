# Save this as check_checkpoint.py
import os
from pathlib import Path

ckpt = "outputs/lang_v1_ft_5k_from_100k/checkpoints/005000/pretrained_model"

# Check for safetensors (preferred) or .bin files
safetensors_files = list(Path(ckpt).glob("*.safetensors"))
bin_files = list(Path(ckpt).glob("*.bin"))

print(f"Found {len(safetensors_files)} safetensors files")
print(f"Found {len(bin_files)} .bin files\n")

if safetensors_files:
    try:
        from safetensors import safe_open
        
        # Load the main model file (not the processor files)
        model_file = None
        for f in safetensors_files:
            if "model.safetensors" in f.name and "processor" not in f.name:
                model_file = f
                break
        
        if not model_file and safetensors_files:
            model_file = safetensors_files[0]
        
        print(f"Loading: {model_file}\n")
        
        with safe_open(model_file, framework="pt", device="cpu") as f:
            all_keys = f.keys()
            
            # Find language-related keys
            lang_keys = [k for k in all_keys if any(x in k.lower() for x in 
                        ["language", "clip", "text_encoder", "token", "embed"])]
            
            print(f"Total keys: {len(list(all_keys))}")
            print(f"Language-related keys: {len(lang_keys)}\n")
            
            if lang_keys:
                print("✅ Found language components:")
                for k in sorted(lang_keys)[:40]:
                    tensor = f.get_tensor(k)
                    print(f"  {k}: {tuple(tensor.shape)}")
            else:
                print("⚠️ NO LANGUAGE KEYS FOUND - checkpoint is NOT language-conditioned")
                print("\nFirst 30 keys in checkpoint:")
                for k in list(all_keys)[:30]:
                    tensor = f.get_tensor(k)
                    print(f"  {k}: {tuple(tensor.shape)}")
    
    except ImportError:
        print("ERROR: safetensors library not installed")
        print("Run: pip install safetensors")

elif bin_files:
    import torch
    print(f"Loading: {bin_files[0]}\n")
    sd = torch.load(bin_files[0], map_location="cpu", weights_only=False)
    
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    
    lang_keys = [k for k in sd.keys() if any(x in k.lower() for x in 
                ["language", "clip", "text_encoder", "token", "embed"])]
    
    print(f"Total keys: {len(sd.keys())}")
    print(f"Language-related keys: {len(lang_keys)}\n")
    
    if lang_keys:
        print("✅ Found language components:")
        for k in sorted(lang_keys)[:40]:
            shape = tuple(sd[k].shape) if hasattr(sd[k], 'shape') else type(sd[k])
            print(f"  {k}: {shape}")
    else:
        print("⚠️ NO LANGUAGE KEYS FOUND")
        print("\nFirst 30 keys:")
        for k in list(sd.keys())[:30]:
            print(f"  {k}: {sd[k].shape}")
else:
    print("No checkpoint files found!")