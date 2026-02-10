import os
from torch.utils import data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from collections import defaultdict


class GTSRBDataset(data.Dataset):
    def __init__(self, path: str, transforms: Compose = None):
        super().__init__()
        self.path = os.path.abspath(path)
        self.transforms = transforms
        self.categories = sorted([
            label for label in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, label))
        ])
        self.path_images = []
        self.labels = []
        self.stats = defaultdict(int)

        for category in self.categories:
            path_category = os.path.join(self.path, category)
            for file in os.listdir(path_category):
                if not file.lower().endswith('.csv'):
                    self.path_images.append(os.path.join(path_category, file))
                    self.labels.append(int(category))
                    self.stats[int(category)] += 1

    def plot_statistics(self):
        keys = list(self.stats.keys())
        values = list(self.stats.values())

        plt.figure(figsize=(12, 6))
        plt.bar(keys, values)
        plt.grid(axis='y', alpha=0.2)
        plt.xlabel(f"Class Id ({len(self.categories)})")
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
    path = "../data/gtsrb/Training"
    transforms = Compose([
        ToTensor()
    ])
    dataset = GTSRBDataset(path=path, transforms=transforms)
    dataset.plot_statistics()
