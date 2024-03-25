import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt


class polyp_dataset(Dataset):
    def __init__(self, images_path, gt_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.gt_path = gt_path
        self.transform = transform
        self.images = os.listdir(os.path.join(self.images_path))
        self.gt = os.listdir(os.path.join(self.gt_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        gt_path = self.gt[idx]

        image = plt.imread(str(os.path.join(self.images_path, image_path)))
        gt = plt.imread(str(os.path.join(self.gt_path, gt_path)))

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)
        return image, gt


if __name__ == "__main__":
    images_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data", "polyps", "train", "train")
    gt_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data", "polyps", "train_gt", "train_gt")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), antialias=True)
    ])

    dataset = polyp_dataset(
        images_path=images_path,
        gt_path=gt_path,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    fig, axis = plt.subplots(2, 3)
    i = 0
    for image, gt in dataloader:
        image = torch.squeeze(image, dim=0)
        gt = torch.squeeze(gt, dim=0)
        if i == 3:
            break
        axis[0, i].imshow(torch.permute(image, (1, 2, 0)))
        axis[1, i].imshow(torch.permute(gt, (1, 2, 0)))
        i += 1
    plt.show()
