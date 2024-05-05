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
    def __init__(self, data_path, mode, device=None):
        super().__init__()
        assert mode in ["train", "test"], "Mode must be train/test"
        self.data_path = data_path
        self.mode = mode
        self.device = device
        if self.mode == "train":
            self.images_embeddings_path = os.path.join(self.data_path, "train_embeddings", "train_embeddings")
            self.gt_embeddings_path = os.path.join(self.data_path, "train_gt_embeddings", "train_gt_embeddings")
            self.images_embeddings = os.listdir(os.path.join(self.images_embeddings_path))
            self.gt_embeddings = os.listdir(os.path.join(self.gt_embeddings_path))

        # if not os.path.exists(self.images_folder_path):
        #     print("Creating test and train images...")
        #     self.create_train_images()
        #     self.create_test_images()
        #     print("Saved test and train images successfully")
        #
        # self.images_path = os.listdir(self.images_folder_path)
        # self.gts_path = os.listdir(self.gts_folder_path)
        if self.mode == "test":
            self.images_embeddings_path = os.path.join(self.data_path, "test_embeddings", "test_embeddings")
            self.gt_embeddings_path = os.path.join(self.data_path, "test_gt_embeddings", "test_gt_embeddings")
            self.images_embeddings = os.listdir(os.path.join(self.images_embeddings_path))
            self.gt_embeddings = os.listdir(os.path.join(self.gt_embeddings_path))

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(self.device)
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
        return gt_embeddings, image_embeddings


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    datasets_lst = ["polyps", "kvasir-seg"]
    for dataset in datasets_lst:
        data_path = os.path.join(os.getcwd(), "data", dataset)
        dataset = polyp_dataset(
            data_path=data_path,
            mode="train",
            device=device
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i in range(len(dataloader)):
            gt, image = next(iter(dataloader))
            print(f"GT shape: {gt.shape}, Image shape: {image.shape}")
        dataset.create_train_images()
        dataset.create_test_images()
        print(f"Created train and test images for {dataset}")