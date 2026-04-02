import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class LevirDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(base_dir, "..", root_dir)

        self.A_dir = os.path.join(root_dir, split, "A")
        self.B_dir = os.path.join(root_dir, split, "B")
        self.L_dir = os.path.join(root_dir, split, "label")

        if not os.path.exists(self.A_dir):
            raise FileNotFoundError(f"Path not found: {self.A_dir}")

        valid_names = []
        for name in sorted(os.listdir(self.A_dir)):
            if os.path.exists(os.path.join(self.B_dir, name)) and os.path.exists(os.path.join(self.L_dir, name)):
                valid_names.append(name)
        self.image_names = valid_names
        print(f"Loaded {len(self.image_names)} valid complete image pairs for split '{split}'")

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        img_A = Image.open(os.path.join(self.A_dir, name)).convert("RGB")
        img_B = Image.open(os.path.join(self.B_dir, name)).convert("RGB")
        label = Image.open(os.path.join(self.L_dir, name)).convert("L")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        label = self.transform(label)

        return img_A, img_B, label
