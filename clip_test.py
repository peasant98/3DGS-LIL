import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load CLIPSeg model and processor
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load and preprocess image
image_path = "img.jpg"
image = Image.open(image_path).convert("RGB")

print(image)

# Preprocess image and text
inputs = clip_processor(text=["chair"], images=image, return_tensors="pt", padding=True).to(device)

# Get segmentation
with torch.no_grad():
    outputs = clipseg_model(**inputs)

# Get the mask
print(outputs.logits.shape)
logits_np = outputs.logits.cpu().numpy()
# plot the logits and image in the same figure
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[1].imshow(logits_np)
plt.show()

# mask = outputs.logits.sigmoid().cpu().numpy()[0, 0]

# print(mask.shape)

# # Threshold the mask to create a binary mask
# threshold = 0.5
# binary_mask = (mask > threshold).astype(np.uint8) * 255

# # Save or display the mask
# cv2.imwrite("object_mask.png", binary_mask)
