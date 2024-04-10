import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import cv2
from unet import UNetModel
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt




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

            # # find edges in the image
            # edges = cv.Canny(image, 100, 200)
            # # find contours in the image
            # contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # plt.figure()
            # plt.imshow(edges, cmap='gray')
            # # plot the contours
            # plt.figure()
            # plt.imshow(cv.drawContours(image, contours, -1, (0, 255, 0), 3))
            # plt.show()


            image = torch.permute(torch.from_numpy(np.copy(image)), (2, 0, 1)).float()
            gt = torch.permute(torch.from_numpy(np.copy(gt)), (2, 0, 1)).float()

            # normalize the image and gt by dividing by 255
            # (smaller numerical values allow faster computation and prevent gradient explosion).
            # source:
            # https://kiansoon.medium.com/semantic-segmentation-is-the-task-of-partitioning-an-image-into-multiple-segments-based-on-the-356a5582370e
            image = image.div_(255)
            gt = gt.div_(255)
            # make the gt a binary image with one channel
            gt = torch.where(gt > 0.5, 1, 0)
            # calculate the sum of each channel in the gt
            sum_lst = []
            for channel in range(gt.shape[0]):
                channel_sum = torch.sum(gt[channel])
                sum_lst.append(channel_sum)
            # get the channel with the maximum sum
            max_sum = max(sum_lst)
            max_sum_idx = sum_lst.index(max_sum)
            gt = gt[max_sum_idx]
            # add a channel to the image
            gt = torch.unsqueeze(gt, dim=0)



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
    for gt, image, _ in dataloader:
        fig, axis = plt.subplots(1, 2)
        gt = gt[0]
        image = image[0]
        axis[0].imshow(gt.permute(1, 2, 0), cmap='Greys', interpolation='nearest')
        axis[1].imshow(image.permute(1, 2, 0))
        plt.show()
        plt.close()










