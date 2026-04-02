import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os

from model import SiameseUNet

# ------------------
# CONFIG
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "siamese_unet_small.pth"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "LEVIR_CD")

SAVE_PATH = "prediction_result.png"

# ------------------
# LOAD MODEL
# ------------------
model = SiameseUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------
# LOAD SAMPLE IMAGE
# ------------------
A_dir = os.path.join(DATASET_PATH, "test", "A")
B_dir = os.path.join(DATASET_PATH, "test", "B")
L_dir = os.path.join(DATASET_PATH, "test", "label")

image_name = sorted(os.listdir(A_dir))[0]

imgA = Image.open(os.path.join(A_dir, image_name)).convert("RGB")
imgB = Image.open(os.path.join(B_dir, image_name)).convert("RGB")
label = Image.open(os.path.join(L_dir, image_name)).convert("L")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

imgA_t = transform(imgA).unsqueeze(0).to(DEVICE)
imgB_t = transform(imgB).unsqueeze(0).to(DEVICE)

# ------------------
# INFERENCE
# ------------------
with torch.no_grad():
    pred = model(imgA_t, imgB_t)
    pred = torch.nn.functional.interpolate(
        pred,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

pred_mask = pred.squeeze().cpu().numpy()

# ------------------
# VISUALIZATION
# ------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title("Before Image")
plt.imshow(imgA)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("After Image")
plt.imshow(imgB)
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Ground Truth")
plt.imshow(label, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Predicted Change")
plt.imshow(pred_mask, cmap="hot")
plt.axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH)
plt.show()

print(f"Prediction saved as {SAVE_PATH}")
