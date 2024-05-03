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
    def __init__(self, data_path, mode, device):
        super().__init__()
        assert mode in ["train", "test"], "Mode must be train/test"
        self.data_path = data_path
        self.mode = mode
        self.device = device
        if self.mode == "train":
            self.images_folder_path = os.path.join(self.data_path, "train_images")
            self.gts_folder_path = os.path.join(self.data_path, "train_gt_images")
        else:
            self.images_folder_path = os.path.join(self.data_path, "test_images")
            self.gts_folder_path = os.path.join(self.data_path, "test_gt_images")
        self.images_path = os.listdir(self.images_folder_path)
        self.gts_path = os.listdir(self.gts_folder_path)

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

        image = image.to(self.device)
        gt = gt.to(self.device)

        if 'polyp' in self.data_path:
            # make the gt a single channel image (multiplied by 3 channels to be compatible with the vae input
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

        # add augmentations
        if random.random() < 0.5:
            contrast = random.uniform(0.5, 1.5)
            image = TF.adjust_contrast(image, contrast)
        if random.random() < 0.5:
            brightness = random.uniform(0.7, 1.5)
            image = TF.adjust_brightness(image, brightness)
        if random.random() < 0.5:
            saturation = random.uniform(1.1, 1.5)
            image = TF.adjust_saturation(image, saturation)
        if random.random() < 0.5:
            # make the image more yellow
            hue = 0.07
            image = TF.adjust_hue(image, hue)
        if random.random() < 0.5:
            # make the image more red
            hue = -0.04
            image = TF.adjust_hue(image, hue)
        if random.random() < 0.5:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        if random.random() < 0.5:
            angle = random.randint(0, 360)
            image = TF.rotate(image, angle)
            gt = TF.rotate(gt, angle)

        with torch.no_grad():
            image = self.vae.encode(image).latent_dist.sample()
            gt = self.vae.encode(gt).latent_dist.sample()

        image = image.mul_(0.18215)
        gt = gt.mul_(0.18215)

        return gt, image


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
        dataset.create_train_images()
        dataset.create_test_images()
        print(f"Created train and test images for {dataset}")