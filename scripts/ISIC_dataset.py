import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np


class ISIC_Dataset(Dataset):
    def __init__(self, path, height, width, cfg, cfg_prob=0.1, mode="train"):
        self.path = path
        self.height = height
        self.width = width
        self.mode = mode
        self.cfg = cfg
        self.cfg_prob = cfg_prob

        if self.mode == "train":
            self.data_path = os.path.join(self.path, "ISIC",  "ISBI2016_ISIC_Part3B_Training_Data")
        else:
            self.data_path = os.path.join(self.path, "ISIC", "ISBI2016_ISIC_Part3B_Test_Data")

        self.images = os.listdir(self.data_path)
        self.rgb_images = []
        self.seg_images = []
        for i in range(len(self.images)):
            if "Segmentation" in self.images[i]:
                self.seg_images.append(self.images[i])
            else:
                self.rgb_images.append(self.images[i])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        rgb_image_path = os.path.join(self.data_path, self.rgb_images[index])
        segmentation_path = os.path.join(self.data_path, self.seg_images[index])

        rgb_image = plt.imread(rgb_image_path)
        segmentation_image = plt.imread(segmentation_path)

        rgb_image = cv2.resize(rgb_image, (self.width, self.height))
        segmentation_image = cv2.resize(segmentation_image, (self.width, self.height))

        rgb_image = torch.from_numpy(rgb_image)
        segmentation_image = torch.from_numpy(segmentation_image)

        rgb_image = torch.permute(rgb_image, (2, 0, 1)).float()
        segmentation_image = torch.unsqueeze(segmentation_image, dim=0)

        rgb_image = (rgb_image - torch.mean(rgb_image)) / torch.std(rgb_image)
        if self.cfg:
            if random.random() < self.cfg_prob:
                rgb_image = torch.zeros(rgb_image.shape)

        return segmentation_image, rgb_image, ""


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data")
    new_height = 64
    new_width = 64
    mode = "train"
    batch_size = 4
    cfg = True
    cfg_prob=0.1

    dataset = ISIC_Dataset(path=path,
                           height=new_height,
                           width=new_width,
                           mode=mode,
                           cfg=cfg,
                           cfg_prob=cfg_prob)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    rgb_images, seg_images, name = next(iter(dataloader))

    fig, axis = plt.subplots(2, 4)
    for i in range(len(rgb_images)):
        rgb_image = torch.permute(rgb_images[i], (1, 2, 0))
        seg_image = torch.permute(seg_images[i], (1, 2, 0))

        axis[0, i].imshow(rgb_image)
        axis[1, i].imshow(seg_image)

    print(f"Data Length {len(dataset)}")
    print(f"RGB Images Shape {rgb_images.shape}")
    print(f"Seg Images Shape {seg_images.shape}")

    plt.show()

