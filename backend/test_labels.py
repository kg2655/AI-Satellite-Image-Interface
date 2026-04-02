import torch
from dataset_loader import LevirDataset

try:
    ds = LevirDataset("dataset/LEVIR_CD", split="train")
    empty_masks = 0
    total_masks = min(20, len(ds))
    print(f"Checking first {total_masks} masks:")
    
    for i in range(total_masks):
        _, _, mask = ds[i]
        
        max_val = mask.max().item()
        min_val = mask.min().item()
        
        # In ToTensor(), 255 becomes 1.0
        if max_val < 0.1:
            empty_masks += 1
            print(f"Mask {i}: EMPTY (max: {max_val:.2f})")
        else:
            sum_val = mask.sum().item()
            print(f"Mask {i}: Contains changes (max: {max_val:.2f}, positive pixels roughly: {sum_val:.1f})")

    print(f"\nResult: {empty_masks} out of {total_masks} were completely blank.")

except Exception as e:
    print("Error:", e)
