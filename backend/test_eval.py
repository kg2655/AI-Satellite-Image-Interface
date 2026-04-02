import torch
from dataset_loader import LevirDataset
from model import SiameseUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ds = LevirDataset("dataset/LEVIR_CD", split="train")

model = SiameseUNet().to(DEVICE)
model.load_state_dict(torch.load("siamese_unet_full.pth", map_location=DEVICE))
model.eval()

# Find an image that actually has changes in it
good_idx = None
for i in range(50):
    _, _, mask = ds[i]
    if mask.sum() > 500:
        good_idx = i
        break

print(f"Testing on training image {good_idx} with actual changes:")
imgA, imgB, mask = ds[good_idx]

imgA = imgA.unsqueeze(0).to(DEVICE)
imgB = imgB.unsqueeze(0).to(DEVICE)
mask = mask.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    preds = model(imgA, imgB)
    if preds.shape[-2:] != mask.shape[-2:]:
        preds = torch.nn.functional.interpolate(
            preds,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

pred_mask = preds.squeeze().cpu()
true_mask = mask.squeeze().cpu()

print(f"True mask pixel sum: {true_mask.sum():.2f}")
print(f"Prediction Min: {pred_mask.min():.4f}, Max: {pred_mask.max():.4f}, Mean: {pred_mask.mean():.4f}")
print(f"Pixels > 0.5: {(pred_mask > 0.5).sum()}")
print(f"Pixels > 0.2: {(pred_mask > 0.2).sum()}")
print(f"Pixels > 0.05: {(pred_mask > 0.05).sum()}")
print(f"Pixels > 0.01: {(pred_mask > 0.01).sum()}")
