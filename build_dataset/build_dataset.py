from torch.utils.data import Dataset
import os
import cv2

class GTSRTDataset(Dataset):
    def __init__(self, root):
        self.categories = []
        self.paths = []
        self.labels = []
        for dir in os.listdir(root):
            if os.path.isdir(os.path.join(root, dir)):
                self.categories.append(int(dir))
                for img in os.listdir(os.path.join(root, dir)):
                    self.paths.append(os.path.join(root, dir, img))
                    self.labels.append(int(dir))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, labels = self.paths[index], self.labels[index]
        return img, labels



if __name__ == '__main__':
    data = GTSRTDataset('dataset/gtsrb/Training')
    img, labels = data.__getitem__(100)
    print(img)
    print(labels)
    img = cv2.imread(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)