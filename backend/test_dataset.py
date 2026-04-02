
from dataset_loader import LevirDataset

ds = LevirDataset("dataset/LEVIR_CD", split="train")


print("Dataset size:", len(ds))

imgA, imgB, mask = ds[0]
print(imgA.shape, imgB.shape, mask.shape)
