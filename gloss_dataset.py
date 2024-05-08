import torch
from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io, transform
import pandas as pd

class GlossDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.train_cutoff = 9000
        self.img_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.img_paths = [f for f in self.img_files]
        self.labels = self._load_labels()  # Load labels internally
        self.images = self._load_images()
        print(len(self.labels), len(self.images))

    def _load_images(self):
        images = []
        for img_path in self.img_paths:
            image = io.imread(os.path.join(self.data_dir, img_path))
            image = transform.resize(image[:,:,0:-1], (128,128,3))
            image = np.clip(255 * image, 0, 255).astype(np.uint8)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images

    def _load_labels(self):
        image_info = pd.read_csv('image_information.csv')
        gloss_labels = np.array(image_info['gloss_cat']).astype('int')
        return gloss_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_data(self):
        if self.train:
            return self.images[:self.train_cutoff], self.labels[:self.train_cutoff]
        else:
            return self.images[self.train_cutoff:], self.labels[self.train_cutoff:]
