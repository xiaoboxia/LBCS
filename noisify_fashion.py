from torchvision.datasets import FashionMNIST
import torch
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
import argparse
import os

def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, default="dataset")
    arg.add_argument("--ratio", type=float, default=0.2)
    args = arg.parse_args()
    return args
args = parse_args()

# mislabel data ratio
mislabel_ratio = args.ratio

trainset = FashionMNIST(args.data, train=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=7, shuffle=False, pin_memory=True)

imgs, targets = [], []
for img, label in tqdm(trainloader):
    for i in range(label.shape[0]):
        p = np.random.rand()
        if p > 1 - mislabel_ratio:
            probs = np.ones((10,), dtype=float) / 9
            probs[label[i]] = 0
            label[i] = np.random.choice(10, p=probs)

    targets.append(label.numpy())
targets = np.concatenate(targets)

if not os.path.exists("tmp"):
    os.makedirs("tmp")

print("Data saving to tmp/fashion_noisy_target.npy")
np.save("tmp/fashion_noisy_target.npy", targets)
print((targets==trainset.targets.numpy()).sum())
