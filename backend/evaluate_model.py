import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader import LevirDataset
from model import SiameseUNet

# ------------------
# CONFIGURATION
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset/LEVIR_CD"
BATCH_SIZE = 8
MODEL_WEIGHTS = "siamese_unet_full.pth"  # Update this if you're using a different weights file

def evaluate():
    print(f"Using device: {DEVICE}")

    # 1. Load the Evaluation Dataset
    # We will try to use the 'test' split. If it's missing, fallback to 'train'.
    try:
        eval_dataset = LevirDataset(DATASET_PATH, split="test")
        print("Loaded test split for evaluation.")
    except Exception as e:
        print("Could not load test split. Falling back to train split for evaluation.")
        eval_dataset = LevirDataset(DATASET_PATH, split="train")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # 2. Initialize the Model
    model = SiameseUNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
        print(f"Successfully loaded model weights from '{MODEL_WEIGHTS}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # 3. Initialize Metric Accumulators
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    total_tn = 0  # True Negatives
    
    total_intersection = 0
    total_union = 0

    print("\nStarting evaluation...")
    with torch.no_grad():
        for imgA, imgB, mask in tqdm(eval_loader, desc="Evaluating dataset"):
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)
            mask = mask.to(DEVICE)

            # Forward pass
            preds = model(imgA, imgB)
            
            # Ensure predictions match the mask size
            if preds.shape[-2:] != mask.shape[-2:]:
                preds = torch.nn.functional.interpolate(
                    preds,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Binarize outputs using a 0.5 threshold
            preds_bin = (preds > 0.5).float()
            mask_bin = (mask > 0.5).float()

            # Flatten for easier calculation
            preds_flat = preds_bin.view(-1)
            mask_flat = mask_bin.view(-1)

            # Calculate Confusion Matrix Elements
            # TP: Prediction is 1, Ground Truth is 1
            tp = (preds_flat * mask_flat).sum().item()
            # FP: Prediction is 1, Ground Truth is 0
            fp = (preds_flat * (1 - mask_flat)).sum().item()
            # FN: Prediction is 0, Ground Truth is 1
            fn = ((1 - preds_flat) * mask_flat).sum().item()
            # TN: Prediction is 0, Ground Truth is 0
            tn = ((1 - preds_flat) * (1 - mask_flat)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # IoU Elements
            intersection = tp
            union = tp + fp + fn
            
            total_intersection += intersection
            total_union += union

    # 4. Calculate Final Metrics
    epsilon = 1e-8 # Prevent division by zero
    
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = total_intersection / (total_union + epsilon)

    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)

    # 5. Display the Results in a Colorful Way
    print("\n" + Fore.CYAN + Style.BRIGHT + "="*60)
    print(Fore.YELLOW + Style.BRIGHT + "                     CONFUSION MATRIX")
    print(Fore.CYAN + "="*60)
    print(f"                      | {Fore.GREEN}Actual Unchanged{Style.RESET_ALL} | {Fore.RED}Actual Changed{Style.RESET_ALL}   |")
    print(Fore.CYAN + "-"*60)
    print(f" {Fore.GREEN}Predicted Unchanged{Style.RESET_ALL}  | {Back.GREEN}{Fore.BLACK} {int(total_tn):^14} {Style.RESET_ALL} | {Back.RED}{Fore.WHITE} {int(total_fn):^14} {Style.RESET_ALL} |")
    print(f" {Fore.RED}Predicted Changed{Style.RESET_ALL}    | {Back.RED}{Fore.WHITE} {int(total_fp):^14} {Style.RESET_ALL} | {Back.GREEN}{Fore.BLACK} {int(total_tp):^14} {Style.RESET_ALL} |")
    print(Fore.CYAN + "="*60 + "\n")

    print(Fore.MAGENTA + Style.BRIGHT + "="*60)
    print(Fore.YELLOW + Style.BRIGHT + "                 MODEL EVALUATION METRICS          ")
    print(Fore.MAGENTA + "="*60)
    print(f"{Fore.CYAN}Pixel Accuracy :{Style.RESET_ALL} {Fore.GREEN}{accuracy:.4f}{Style.RESET_ALL}  ({accuracy*100:5.2f}%)")
    print(f"{Fore.CYAN}Precision      :{Style.RESET_ALL} {Fore.GREEN}{precision:.4f}{Style.RESET_ALL}  ({precision*100:5.2f}%)")
    print(f"{Fore.CYAN}Recall         :{Style.RESET_ALL} {Fore.GREEN}{recall:.4f}{Style.RESET_ALL}  ({recall*100:5.2f}%)")
    print(f"{Fore.CYAN}F1-Score       :{Style.RESET_ALL} {Fore.GREEN}{f1_score:.4f}{Style.RESET_ALL}  ({f1_score*100:5.2f}%)")
    print(f"{Fore.CYAN}IoU            :{Style.RESET_ALL} {Fore.GREEN}{iou:.4f}{Style.RESET_ALL}  ({iou*100:5.2f}%)")
    print(Fore.MAGENTA + "="*60)
    
    # 6. Try to save a visual representation if libraries are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Format memory structure for confusion matrix plot: [TN, FP] \n [FN, TP]
        # X-axis will be Predicted, Y-axis will be Actual depending on how we plot it.
        cm = np.array([[total_tn, total_fp], [total_fn, total_tp]], dtype=int)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Unchanged (0)', 'Predicted Changed (1)'], 
                    yticklabels=['Actual Unchanged (0)', 'Actual Changed (1)'])
        plt.ylabel('Actual Label (Ground Truth)')
        plt.xlabel('Predicted Label')
        plt.title('Pixel-wise Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print(f"\n--> Saved visual confusion matrix image to 'confusion_matrix.png'")
    except ImportError:
        print("\nNote: You can install 'matplotlib' and 'seaborn' via pip to automatically save a visual confusion matrix image.")

if __name__ == "__main__":
    evaluate()
