
import torch
import torch.nn as nn
import sys
import os

# Ensure we can import app
sys.path.append(os.getcwd())

from app import video_detector, VideoDetector

def get_output_size(model, frames, height, width):
    try:
        # Input: (Batch, Channels, Depth/Time, Height, Width)
        # Note: app.py loads 16 frames. 
        # frames_array = np.transpose(frames_array, (3, 0, 1, 2)) -> (C, T, H, W)
        x = torch.randn(1, 3, frames, height, width)
        
        # Pass through conv layers
        out = model.model.conv3d_layers(x)
        features = out.view(out.size(0), -1)
        return features.shape[1]
    except Exception as e:
        return f"Error: {e}"

print("Debugging Video Model Shapes...")
model = video_detector

if model and model.model:
    # Try different configurations
    configs = [
        (16, 112, 112), # Current
        (16, 56, 56),
        (8, 112, 112),
        (8, 56, 56),
        (16, 64, 64),
        (6, 56, 56) # 6 frames?
    ]
    
    target = 9408
    
    for f, h, w in configs:
        size = get_output_size(model, f, h, w)
        print(f"Frames={f}, H={h}, W={w} -> Output Size={size} (Target={target})")
        if size == target:
            print(f"✅ FOUND MATCH! Frames={f}, H={h}, W={w}")
            
else:
    print("Video Model not loaded.")
