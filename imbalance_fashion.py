from datasets_utils.imbalanced_sampler import ImbalanceFashion
import numpy as np
import argparse
import os

def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, default="dataset")
    args = arg.parse_args()
    return args

args = parse_args()

save_dir = "tmp"
dataset = ImbalanceFashion(args.data, download=True)
print("Generated dataset size: ", len(dataset.targets))

print("Data saved to: ", save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(os.path.join(save_dir, "fashion_imb_targets.npy"), dataset.targets)
np.save(os.path.join(save_dir, "fashion_imb_train.npy"), dataset.data)