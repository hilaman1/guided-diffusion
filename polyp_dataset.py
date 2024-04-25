import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import random
import cv2


TRAIN_FRACTION = 0.8

class polyp_dataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        super().__init__()
        assert mode in ["train", "test"], "Mode must be train/test"
        self.data_path = data_path
        self.mode = mode

        self.images_embeddings_path = r"D:\Hila\guided-diffusion\datasets\polyps\dataset_embeddings\train_embeddings\train_embeddings"
        self.gt_embeddings_path = r"D:\Hila\guided-diffusion\datasets\polyps\dataset_embeddings\train_gt_embeddings\train_gt_embeddings"
        self.transform = transform

        self.images_embeddings = os.listdir(os.path.join(self.images_embeddings_path))
        self.gt_embeddings = os.listdir(os.path.join(self.gt_embeddings_path))

        n_train_images = int(len(self.images_embeddings) * TRAIN_FRACTION)
        if self.mode == "train":
            self.images_embeddings = self.images_embeddings[:n_train_images]
            self.gt_embeddings = self.gt_embeddings[:n_train_images]
        elif self.mode == "test":
            self.images_embeddings = self.images_embeddings[n_train_images:]
            self.gt_embeddings = self.gt_embeddings[n_train_images:]

    def __len__(self):
        return len(self.images_embeddings)

    def __getitem__(self, idx):
        image_embeddings_path = self.images_embeddings[idx]
        gt_embeddings_path = self.gt_embeddings[idx]

        image_embeddings = torch.load(str(os.path.join(self.images_embeddings_path, image_embeddings_path)))
        gt_embeddings = torch.load(str(os.path.join(self.gt_embeddings_path, gt_embeddings_path)))

        image_embeddings = torch.squeeze(image_embeddings, dim=0)
        gt_embeddings = torch.squeeze(gt_embeddings, dim=0)

        image_embeddings = image_embeddings.mul_(0.18215)
        gt_embeddings = gt_embeddings.mul_(0.18215)

        if self.transform:
            if random.random() < 0.5:
                image_embeddings = self.transform(image_embeddings)
                gt_embeddings = self.transform(gt_embeddings)

        return gt_embeddings, image_embeddings


if __name__ == "__main__":
    data_path = "C:\\Users\\Admin\\Documents\\GitHub\\guided-diffusion\\scripts\\data\\polyps"
    mode = "train"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0)
    ])

    dataset = polyp_dataset(
        data_path=data_path,
        mode=mode,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_images = 6

    fig, axis = plt.subplots(2, n_images)
    i = 0
    for gt, image in dataloader:
        image = torch.squeeze(image, dim=0)
        gt = torch.squeeze(gt, dim=0)
        if i == n_images:
            break

        axis[0, i].imshow(torch.permute(image.cpu(), (1, 2, 0)))
        axis[1, i].imshow(torch.permute(gt.cpu(), (1, 2, 0)))
        i += 1

    plt.show()
