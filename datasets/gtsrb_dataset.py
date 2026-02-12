import os

import torch
from torch.utils import data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor, transforms, ToPILImage, Resize
import pandas as pd

class GTSRBDataset(data.Dataset):
    def __init__(self, root, transforms: Compose = None, train=True):
        super().__init__()
        root = os.path.abspath(root)
        self.transforms = transforms

        if train:
            df = pd.read_csv(os.path.join(root, "GT-training.csv"), sep=";")
            df["ClassId"] = df["ClassId"].apply(lambda x: f"{x:05d}")
            self.path_images = df.apply(lambda row: os.path.join(root, "Training", row["ClassId"], row["Filename"]), axis=1).values
        else:
            df = pd.read_csv(os.path.join(root, "GT-final_test.csv"), sep=";")
            self.path_images = df.apply(lambda row: os.path.join(root, "Final_Test", "Images", row["Filename"]), axis=1).values

        self.labels = df["ClassId"].astype(int).values
        self.stats = df["ClassId"].astype(int).value_counts().sort_index().to_dict()
        self.categories = list(self.stats.keys())

    def plot_statistics(self):
        keys = list(self.stats.keys())
        values = list(self.stats.values())

        plt.figure(figsize=(12, 6))
        plt.bar(keys, values)
        plt.grid(axis='y', alpha=0.2)
        plt.xlabel(f"Class Id ({len(keys)})")
        plt.ylabel(f"Number of images ({len(self.labels)})")

        plt.show()

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        img_path = self.path_images[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


if __name__ == '__main__':
    path = "../data/gtsrb/"


