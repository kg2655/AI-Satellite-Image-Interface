import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import LevirDataset
from model import SiameseUNet

# ------------------
# CONFIG
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset/LEVIR_CD"
BATCH_SIZE = 8
EPOCHS = 50        # Train longer for actual results
LR = 1e-4

if __name__ == '__main__':
    print("Using device:", DEVICE)
    
    # ------------------
    # DATASET
    # ------------------
    # Load the full dataset (637 images)
    train_dataset = LevirDataset(DATASET_PATH, split="train")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, # Use multiple workers for speed
        pin_memory=True if DEVICE == "cuda" else False
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
    # OPTIMIZER & SCHEDULER
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ------------------
    # TRAINING LOOP
    # ------------------
    best_loss = float('inf')

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
            
            # Ensure preds match mask size
            if preds.shape[-2:] != mask.shape[-2:]:
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
        
        scheduler.step(avg_loss)
    
        # ------------------
        # SAVE BEST MODEL
        # ------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "siamese_unet_full.pth")
            print(f"--> Saved best model with loss: {best_loss:.4f}")
    
    print("Training Complete! The best weights are saved in 'siamese_unet_full.pth'")
