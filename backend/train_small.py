import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_loader import LevirDataset
from model import SiameseUNet

# ------------------
# CONFIG
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset/LEVIR_CD"
BATCH_SIZE = 4
EPOCHS = 3        # small test
LR = 1e-4
MAX_SAMPLES = 100  # small subset

print("Using device:", DEVICE)

# ------------------
# DATASET
# ------------------
full_dataset = LevirDataset(DATASET_PATH, split="train")

subset_indices = list(range(min(MAX_SAMPLES, len(full_dataset))))
small_dataset = Subset(full_dataset, subset_indices)

train_loader = DataLoader(
    small_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ------------------
# MODEL
# ------------------
model = SiameseUNet().to(DEVICE)

# ------------------
# LOSS FUNCTIONS
# ------------------
bce_loss = nn.BCELoss()

def dice_loss(pred, target):
    smooth = 1.0
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

# ------------------
# OPTIMIZER
# ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------
# TRAINING LOOP
# ------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for imgA, imgB, mask in loop:
        imgA = imgA.to(DEVICE)
        imgB = imgB.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        preds = model(imgA, imgB)
        preds = torch.nn.functional.interpolate(
    preds,
    size=mask.shape[-2:],
    mode="bilinear",
    align_corners=False
)
        loss = bce_loss(preds, mask) + dice_loss(preds, mask)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# ------------------
# SAVE MODEL
# ------------------
torch.save(model.state_dict(), "siamese_unet_small.pth")
print("Model saved as siamese_unet_small.pth")
