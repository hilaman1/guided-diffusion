import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import cv2


TRAIN_IMAGES_FRACTION = 0.8

class polyp_dataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        super().__init__()
        assert mode in ["train", "test"], "Mode must be train or test"
        self.data_path = data_path
        self.transform = transform

        self.images_embeddings = os.listdir(os.path.join(data_path, "dataset_embeddings", "train_embeddings", "train_embeddings"))
        self.gt_embeddings = os.listdir(os.path.join(data_path, "dataset_embeddings", "train_gt_embeddings", "train_gt_embeddings"))

        n_images = int(TRAIN_IMAGES_FRACTION * len(self.images_embeddings))
        if mode == "train":
            self.images_embeddings = self.images_embeddings[:n_images]
            self.gt_embeddings = self.gt_embeddings[:n_images]
        if mode == "test":
            self.images_embeddings = self.images_embeddings[n_images:]
            self.gt_embeddings = self.gt_embeddings[n_images:]

    def __len__(self):
        return len(self.images_embeddings)

    def __getitem__(self, idx):
        image_embeddings_path = self.images_embeddings[idx]
        gt_embeddings_path = self.gt_embeddings[idx]

        image_embeddings = torch.load(str(os.path.join(self.data_path, "dataset_embeddings", "train_embeddings", "train_embeddings",
                                                       image_embeddings_path)))
        gt_embeddings = torch.load(str(os.path.join(self.data_path, "dataset_embeddings", "train_gt_embeddings", "train_gt_embeddings",
                                                    gt_embeddings_path)))

        image_embeddings = torch.squeeze(image_embeddings, dim=0)
        gt_embeddings = torch.squeeze(gt_embeddings, dim=0)

        image_embeddings = image_embeddings.mul_(0.18215)
        gt_embeddings = gt_embeddings.mul_(0.18215)
        return image_embeddings, gt_embeddings



if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device {device}")
    data_path = os.path.join(os.getcwd(), "data", "polyps")

    dataset = polyp_dataset(
        data_path=data_path,
        mode="train"
    )

    print(f"Dataset length {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)

    fig, axis = plt.subplots(2, 3)
    i = 0
    for batch_idx, (image_embeddings, gt_embeddings) in enumerate(dataloader):
        image_embeddings = torch.squeeze(image_embeddings, dim=0)
        gt_embeddings = torch.squeeze(gt_embeddings, dim=0)
        if i == 3:
            break

        axis[0, i].imshow(torch.permute(image_embeddings.cpu(), (1, 2, 0)))
        axis[1, i].imshow(torch.permute(gt_embeddings.cpu(), (1, 2, 0)))
        i += 1

    plt.show()
