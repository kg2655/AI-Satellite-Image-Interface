import torch
import torchvision.transforms as T
from PIL import Image
from model import SiameseUNet

# Load Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseUNet().to(DEVICE)
model.load_state_dict(torch.load("siamese_unet_full.pth", map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# Use the uploaded images
path_A = r"C:\Users\Kannu Goyal\.gemini\antigravity\brain\73849cec-4ba5-4928-8d1d-c91e8e57f209\media__1773482686422.jpg"
path_B = r"C:\Users\Kannu Goyal\.gemini\antigravity\brain\73849cec-4ba5-4928-8d1d-c91e8e57f209\media__1773482686439.jpg"
    
print(f"Testing on uploaded images")

imgA = Image.open(path_A).convert("RGB")
imgB = Image.open(path_B).convert("RGB")

imgA_t = transform(imgA).unsqueeze(0).to(DEVICE)
imgB_t = transform(imgB).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(imgA_t, imgB_t)
    
pred_mask = pred.squeeze().cpu().numpy()

print(f"Prediction Stats:")
print(f"Min probability: {pred_mask.min():.4f}")
print(f"Max probability: {pred_mask.max():.4f}")
print(f"Mean probability: {pred_mask.mean():.4f}")

# How many pixels are above various thresholds?
print(f"\nPixels > 0.5: {(pred_mask > 0.5).sum()}")
print(f"Pixels > 0.2: {(pred_mask > 0.2).sum()}")
print(f"Pixels > 0.1: {(pred_mask > 0.1).sum()}")
print(f"Pixels > 0.05: {(pred_mask > 0.05).sum()}")
print(f"Pixels > 0.01: {(pred_mask > 0.01).sum()}")
