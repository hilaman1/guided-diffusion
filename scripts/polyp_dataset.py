import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import cv2


class polyp_dataset(Dataset):
    def __init__(self, images_path, gt_path, images_embeddings_path, gt_embeddings_path, new_image_height,
                 new_image_width, transform=None):
        super().__init__()
        self.images_path = images_path
        self.gt_path = gt_path
        self.images_embeddings_path = images_embeddings_path
        self.gt_embeddings_path = gt_embeddings_path
        self.new_image_height = new_image_height
        self.new_image_width = new_image_width
        self.transform = transform

        self.images = os.listdir(os.path.join(self.images_path))
        self.images_embeddings = os.listdir(os.path.join(self.images_embeddings_path))
        self.gt = os.listdir(os.path.join(self.gt_path))
        self.gt_embeddings = os.listdir(os.path.join(self.gt_embeddings_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_embeddings_path = self.images_embeddings[idx]
        gt_path = self.gt[idx]
        gt_embeddings_path = self.gt_embeddings[idx]

        image = plt.imread(str(os.path.join(self.images_path, image_path)))
        image_embeddings = torch.load(str(os.path.join(self.images_embeddings_path, image_embeddings_path)))
        gt = plt.imread(str(os.path.join(self.gt_path, gt_path)))
        gt_embeddings = torch.load(str(os.path.join(self.gt_embeddings_path, gt_embeddings_path)))

        image = cv2.resize(image, (512, 512))
        gt = cv2.resize(gt, (512, 512))

        image = torch.permute(torch.from_numpy(np.copy(image)), (2, 0, 1))
        gt = torch.permute(torch.from_numpy(np.copy(gt)), (2, 0 ,1))

        image_embeddings = torch.squeeze(image_embeddings, dim=0)
        gt_embeddings = torch.squeeze(gt_embeddings, dim=0)

        image_embeddings = image_embeddings.mul_(0.18215)
        gt_embeddings = gt_embeddings.mul_(0.18215)
        return gt_embeddings, image_embeddings, ""


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device {device}")
    images_path = os.path.join(os.getcwd(), "data", "polyps", "train", "train")
    images_embeddings_path = os.path.join(os.getcwd(), "data", "polyps", "train_embeddings", "train_embeddings")
    gt_path = os.path.join(os.getcwd(), "data", "polyps", "train_gt", "train_gt")
    gt_path_embeddings = os.path.join(os.getcwd(), "data", "polyps", "train_gt_embeddings", "train_gt_embeddings")

    transform = transforms.Compose([
        transforms.Resize((512, 512), antialias=True)
    ])

    dataset = polyp_dataset(
        images_path=images_path,
        gt_path=gt_path,
        images_embeddings_path=images_embeddings_path,
        gt_embeddings_path=gt_path_embeddings,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    fig, axis = plt.subplots(4, 3)
    i = 0
    for image, image_embeddings, gt, gt_embeddings in dataloader:
        image = torch.squeeze(image, dim=0)
        gt = torch.squeeze(gt, dim=0)
        if i == 3:
            break

        with torch.no_grad():
            decoded_image = vae.decode(image_embeddings.to(device)).sample
            decoded_gt = vae.decode(gt_embeddings.to(device)).sample

        decoded_image = decoded_image.cpu().detach()
        decoded_gt = decoded_gt.cpu().detach()

        axis[0, i].imshow(image)
        axis[1, i].imshow(gt)
        axis[2, i].imshow(torch.permute(torch.squeeze(decoded_image, dim=0), (1, 2, 0)))
        axis[3, i].imshow(torch.permute(torch.squeeze(decoded_gt, dim=0), (1, 2, 0)))
        i += 1

    plt.show()
