import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import random
import cv2
from torchvision import transforms as transforms


TRAIN_FRACTION = 0.8

class polyp_dataset(Dataset):
    def __init__(self, data_path, mode):
        super().__init__()
        assert mode in ["train", "test"], "Mode must be train/test"
        self.data_path = data_path
        self.mode = mode
        if self.mode == "train":
            self.images_folder_path = os.path.join(self.data_path, "train_images")
            self.gts_folder_path = os.path.join(self.data_path, "train_gt_images")
        else:
            self.images_folder_path = os.path.join(self.data_path, "test_images")
            self.gts_folder_path = os.path.join(self.data_path, "test_gt_images")
        self.images_path = os.listdir(self.images_folder_path)
        self.gts_path = os.listdir(self.gts_folder_path)
    def create_train_images(self):
        if 'polyps' in self.data_path:
            self.images_folder_path = os.path.join(self.data_path, "train", "train")
            self.gts_folder_path = os.path.join(self.data_path, "train_gt", "train_gt")
        else:
            self.images_folder_path = os.path.join(self.data_path, "images")
            self.gts_folder_path = os.path.join(self.data_path, "masks")
        self.images_path = os.listdir(self.images_folder_path)
        self.gts_path = os.listdir(self.gts_folder_path)
        self.images_path = self.images_path[:int(TRAIN_FRACTION * len(self.images_path))]
        self.gts_path = self.gts_path[:int(TRAIN_FRACTION * len(self.gts_path))]
        if not os.path.exists(os.path.join(self.data_path, "train_images")):
            os.mkdir(os.path.join(self.data_path, "train_images"))
        if not os.path.exists(os.path.join(self.data_path, "train_gt_images")):
            os.mkdir(os.path.join(self.data_path, "train_gt_images"))
        for i in range(len(self.images_path)):
            image = cv2.imread(os.path.join(self.images_folder_path, self.images_path[i]))
            gt = cv2.imread(os.path.join(self.gts_folder_path, self.gts_path[i]))
            cv2.imwrite(os.path.join(self.data_path, "train_images", self.images_path[i]), image)
            cv2.imwrite(os.path.join(self.data_path, "train_gt_images", self.gts_path[i]), gt)

    def create_test_images(self):
        if 'polyps' in self.data_path:
            self.images_folder_path = os.path.join(self.data_path, "train", "train")
            self.gts_folder_path = os.path.join(self.data_path, "train_gt", "train_gt")
        else:
            self.images_folder_path = os.path.join(self.data_path, "images")
            self.gts_folder_path = os.path.join(self.data_path, "masks")
        self.images_path = os.listdir(self.images_folder_path)
        self.gts_path = os.listdir(self.gts_folder_path)
        self.images_path = self.images_path[int(TRAIN_FRACTION * len(self.images_path)):]
        self.gts_path = self.gts_path[int(TRAIN_FRACTION * len(self.gts_path)):]
        if not os.path.exists(os.path.join(self.data_path, "test_images")):
            os.mkdir(os.path.join(self.data_path, "test_images"))
        if not os.path.exists(os.path.join(self.data_path, "test_gt_images")):
            os.mkdir(os.path.join(self.data_path, "test_gt_images"))
        for i in range(len(self.images_path)):
            image = cv2.imread(os.path.join(self.images_folder_path, self.images_path[i]))
            gt = cv2.imread(os.path.join(self.gts_folder_path, self.gts_path[i]))
            cv2.imwrite(os.path.join(self.data_path, "test_images", self.images_path[i]), image)
            cv2.imwrite(os.path.join(self.data_path, "test_gt_images", self.gts_path[i]), gt)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):

        image_path = self.images_path[idx]
        gt_path = self.gts_path[idx]

        image = plt.imread(str(os.path.join(self.images_folder_path, image_path)))
        gt = plt.imread(str(os.path.join(self.gts_folder_path, gt_path)))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])

        image = transform(image)
        gt = transform(gt)


        if 'polyp' in self.data_path:
            # make the gt a single channel image (multipiled by 3 channels to be compatible with the vae input
            # dimensions)
            gt = torch.unsqueeze(gt, dim=0)
            sum_lst = []
            for channel in range(gt.shape[1]):
                channel_sum = torch.sum(gt[0, channel])
                sum_lst.append(channel_sum)
            max_sum = max(sum_lst)
            max_sum_idx = sum_lst.index(max_sum)

            max_sum_channel = torch.squeeze(gt, dim=0)[max_sum_idx, :, :]
            max_sum_channel = torch.unsqueeze(max_sum_channel, dim=0)
            gt = torch.cat((max_sum_channel, max_sum_channel, max_sum_channel), dim=0)

        return gt, image


if __name__ == "__main__":
    datasets_lst = ["polyps", "kvasir-seg"]
    for dataset in datasets_lst:
        data_path = os.path.join(os.getcwd(), "data", dataset)
        dataset = polyp_dataset(
            data_path=data_path,
            mode="train"
        )
        dataset.create_train_images()
        dataset.create_test_images()
        print(f"Created train and test images for {dataset}")