import numpy as np
from PIL import Image
from pathlib import Path

# Configurationw
INPUT_DIR = Path("problem2_outputs/generated")
OUTPUT_DIR = Path("problem2_outputs/generated_png")
OUTPUT_DIR.mkdir(exist_ok=True)

# GAN normalization constants (from case2_2.py)
GAN_MEAN = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
GAN_STD = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

def denormalize(tensor):
    """Convert from [-1, 1] back to [0, 1]"""
    return tensor * GAN_STD + GAN_MEAN

def npy_to_png(npy_path, png_path):
    """Convert single .npy file to .png"""
    # Load numpy array (shape: [3, 32, 32])
    img_tensor = np.load(npy_path)
    
    # Denormalize from [-1, 1] to [0, 1]
    img_tensor = denormalize(img_tensor)
    
    # Clip to valid range
    img_tensor = np.clip(img_tensor, 0, 1)
    
    # Convert to [32, 32, 3] and scale to [0, 255]
    img_array = (img_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(img_array).save(png_path)

# Convert all .npy files
npy_files = sorted(INPUT_DIR.glob("*.npy"))
print(f"Found {len(npy_files)} .npy files")

for npy_file in npy_files:
    png_file = OUTPUT_DIR / f"{npy_file.stem}.png"
    npy_to_png(npy_file, png_file)
    
print(f"Converted {len(npy_files)} images to {OUTPUT_DIR}")