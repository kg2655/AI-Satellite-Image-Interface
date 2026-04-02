from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from model import SiameseUNet

# ------------------
# APP SETUP
# ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# LOAD MODEL ONCE
# ------------------
model = SiameseUNet().to(DEVICE)
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "siamese_unet_full.pth"),
        map_location=DEVICE
    )
)
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# ------------------
# API ENDPOINT
# ------------------
@app.post("/detect-change")
async def detect_change(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):
    before_path = os.path.join(UPLOAD_DIR, "before.png")
    after_path = os.path.join(UPLOAD_DIR, "after.png")

    with open(before_path, "wb") as f:
        shutil.copyfileobj(before_image.file, f)

    with open(after_path, "wb") as f:
        shutil.copyfileobj(after_image.file, f)

    imgA = Image.open(before_path).convert("RGB")
    imgB = Image.open(after_path).convert("RGB")

    # Get original dimensions: (height, width)
    original_size = (imgA.size[1], imgA.size[0])

    imgA_t = transform(imgA).unsqueeze(0).to(DEVICE)
    imgB_t = transform(imgB).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(imgA_t, imgB_t)
        # Interpolate back to the original full-resolution rather than 256x256
        pred = torch.nn.functional.interpolate(
            pred,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

    # Convert to probability map
    pred_mask = pred.squeeze().cpu()

    # 🔥 Standard Threshold
    # The new U-Net model predicts the exact sharp location, so we just use 0.5
    threshold = 0.5
    binary_mask = (pred_mask > threshold).numpy().astype(np.uint8) * 255

    # 🔥 Light Post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Just a light close to ensure solid shape
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 2. Filter by Contour Area (Remove tiny false positives)
    final_mask = np.zeros_like(closed_mask)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100: # minimum pixel area to be considered a real building change
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # Save as clear black-white image
    output_path = os.path.join(OUTPUT_DIR, "predicted_mask.png")
    final_pil = Image.fromarray(final_mask)
    final_pil.save(output_path)

    # Calculate change statistics
    change_pixels = np.count_nonzero(final_mask)
    total_pixels = final_mask.size
    change_percentage = (change_pixels / total_pixels) * 100

    import base64
    with open(output_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return {
        "mask_base64": encoded_string,
        "change_pixels": int(change_pixels),
        "change_percentage": float(change_percentage),
        "total_pixels": int(total_pixels)
    }
