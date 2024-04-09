import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import cv2
from unet import UNetModel



class polyp_dataset(Dataset):
    def __init__(self, images_path, gt_path, images_embeddings_path, gt_embeddings_path, new_image_height,
                 new_image_width, guided, transform=None):
        super().__init__()
        self.images_path = images_path
        self.gt_path = gt_path
        self.images_embeddings_path = images_embeddings_path
        self.gt_embeddings_path = gt_embeddings_path
        self.new_image_height = new_image_height
        self.new_image_width = new_image_width
        self.guided = guided
        self.transform = transform

        self.images = os.listdir(os.path.join(self.images_path))
        self.images_embeddings = os.listdir(os.path.join(self.images_embeddings_path))
        self.gt = os.listdir(os.path.join(self.gt_path))
        self.gt_embeddings = os.listdir(os.path.join(self.gt_embeddings_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.guided:  # Return embeddings
            image_embeddings_path = self.images_embeddings[idx]
            gt_embeddings_path = self.gt_embeddings[idx]

            image_embeddings = torch.load(str(os.path.join(self.images_embeddings_path, image_embeddings_path)))
            gt_embeddings = torch.load(str(os.path.join(self.gt_embeddings_path, gt_embeddings_path)))

            image_embeddings = torch.squeeze(image_embeddings, dim=0)
            gt_embeddings = torch.squeeze(gt_embeddings, dim=0)

            image_embeddings = image_embeddings.mul_(0.18215)
            gt_embeddings = gt_embeddings.mul_(0.18215)
            return image_embeddings, gt_embeddings, ""

        else:  # Return original images
            image_path = self.images[idx]
            gt_path = self.gt[idx]

            image_path = self.images[idx]
            gt_path = self.gt[idx]

            image = plt.imread(str(os.path.join(self.images_path, image_path)))
            gt = plt.imread(str(os.path.join(self.gt_path, gt_path)))

            image = cv2.resize(image, (self.new_image_width, self.new_image_height))
            gt = cv2.resize(gt, (self.new_image_width, self.new_image_height))

            image = torch.permute(torch.from_numpy(np.copy(image)), (2, 0, 1)).float()
            gt = torch.permute(torch.from_numpy(np.copy(gt)), (2, 0, 1)).float()

            # # devide by 255
            image = image.div_(255)
            gt = gt.div_(255)

            return gt, image, ""


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device {device}")
    images_path = r"D:\Hila\guided-diffusion\datasets\polyps\train\train"
    images_embeddings_path = r"D:\Hila\guided-diffusion\datasets\polyps\dataset_embeddings\train_embeddings\train_embeddings"
    gt_path = r"D:\Hila\guided-diffusion\datasets\polyps\train_gt\train_gt"
    gt_path_embeddings = r"D:\Hila\guided-diffusion\datasets\polyps\dataset_embeddings\train_gt_embeddings\train_gt_embeddings"
    new_image_height = 64
    new_image_width = 64
    guided = False

    dataset = polyp_dataset(
        images_path=images_path,
        gt_path=gt_path,
        images_embeddings_path=images_embeddings_path,
        gt_embeddings_path=gt_path_embeddings,
        new_image_height=new_image_height,
        new_image_width=new_image_width,
        guided=guided
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # plot the original image, the ground truth
    fig, axis = plt.subplots(1,2)
    for image, gt, _ in dataloader:
        image = torch.squeeze(image, dim=0)
        image = torch.permute(torch.from_numpy(np.copy(image)), (1, 2, 0))
        gt = torch.squeeze(gt, dim=0)
        gt = torch.permute(torch.from_numpy(np.copy(gt)), (1, 2, 0))
        # make image values between 0 and 255
        image = image.mul_(255).byte()
        gt = gt.mul_(255).byte()
        axis[0].imshow(image)
        axis[1].imshow(gt)
        break
    plt.show()

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
