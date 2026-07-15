from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# --- 1. Load an image from a URL ---
url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

print(f"Format: {img.format}")       # JPEG, PNG, etc.
print(f"Mode: {img.mode}")           # RGB, RGBA, L (grayscale)
print(f"Size: {img.size}")           # (width, height) in pixels

# --- 2. The key insight: images are numbers ---
img_array = np.array(img)
print(f"Array shape: {img_array.shape}")  # (height, width, 3)
print(f"Top-left pixel: {img_array[0, 0]}")  # e.g. [134, 189, 201]
print(f"Pixel value range: {img_array.min()} – {img_array.max()}")

# --- 3. Basic manipulations ---
img_resized = img.resize((224, 224))   # 224x224 is standard for CNNs — remember this
img_gray    = img.convert("L")         # grayscale (1 channel instead of 3)
img_cropped = img.crop((0, 0, 400, 400))  # (left, top, right, bottom)

# --- 4. Display them ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_resized); axes[0].set_title("Resized 224×224")
axes[1].imshow(img_gray, cmap="gray"); axes[1].set_title("Grayscale")
axes[2].imshow(img_cropped); axes[2].set_title("Cropped")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig("phase1_output.png")
print("Saved to phase1_output.png")

# --- 5. Build a simple tracking DataFrame ---
records = []
for label, sample_url in [
    ("moody",      "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400"),
    ("minimalist", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400"),
    ("vibrant",    "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=400"),
]:
    r = requests.get(sample_url)
    im = Image.open(BytesIO(r.content))
    arr = np.array(im)
    records.append({
        "url":        sample_url,
        "label":      label,
        "width":      im.size[0],
        "height":     im.size[1],
        "mean_r":     arr[:, :, 0].mean().round(1),
        "mean_g":     arr[:, :, 1].mean().round(1),
        "mean_b":     arr[:, :, 2].mean().round(1),
    })

df = pd.DataFrame(records)
print(df)
df.to_csv("dataset.csv", index=False)
print("Saved dataset.csv")