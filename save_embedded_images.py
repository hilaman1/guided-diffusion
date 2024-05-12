import argparse
import matplotlib.pyplot as plt
import torch
import os.path
from torchvision.transforms import transforms
from diffusers.models import AutoencoderKL
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils import delete_dir
import numpy as np


random.seed(42)


def split_images(data_path, images_path, gt_path, train_fraction=0.8):
    images_list = os.listdir(images_path)
    if os.path.exists(os.path.join(data_path, "test_images")):
        delete_dir(os.path.join(data_path, "test_images"))
        os.mkdir(os.path.join(data_path, "test_images"))
    else:
        os.mkdir(os.path.join(data_path, "test_images"))

    if os.path.exists(os.path.join(data_path, "test_gt_images")):
        delete_dir(os.path.join(data_path, "test_gt_images"))
        os.mkdir(os.path.join(data_path, "test_gt_images"))
    else:
        os.mkdir(os.path.join(data_path, "test_gt_images"))

    if os.path.exists(os.path.join(data_path, "train_images")):
        delete_dir(os.path.join(data_path, "train_images"))
        os.mkdir(os.path.join(data_path, "train_images"))
    else:
        os.mkdir(os.path.join(data_path, "train_images"))

    if os.path.exists(os.path.join(data_path, "train_gt_images")):
        delete_dir(os.path.join(data_path, "train_gt_images"))
        os.mkdir(os.path.join(data_path, "train_gt_images"))
    else:
        os.mkdir(os.path.join(data_path, "train_gt_images"))

    train_data = random.sample(images_list, int(train_fraction * len(images_list)))
    test_data = [images_list[i] for i in range(len(images_list)) if images_list[i] not in train_data]

    print("Saving Training Images")
    for i in range(len(train_data)):
        image = plt.imread(os.path.join(images_path, train_data[i]))
        gt = plt.imread(os.path.join(gt_path, train_data[i]))

        plt.imsave(os.path.join(data_path, "train_images", train_data[i]), image)
        plt.imsave(os.path.join(data_path, "train_gt_images", train_data[i]), gt)

    print("Saving Testing Images")
    for i in range(len(test_data)):
        image = plt.imread(os.path.join(images_path, test_data[i]))
        gt = plt.imread(os.path.join(gt_path, test_data[i]))

        plt.imsave(os.path.join(data_path, "test_images", test_data[i]), image)
        plt.imsave(os.path.join(data_path, "test_gt_images", test_data[i]), gt)
    print("Saved Images Successfully")


def save_embedded_images(data_path, images_path, gt_path, mode, resize_height=512, resize_width=512):
    assert mode in ["train", "test"], "mode must be train/test."
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_image_path = os.path.join(data_path, f"{mode}_embeddings")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)
    output_image_path = os.path.join(output_image_path, f"{mode}_embeddings")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)

    output_gt_path = os.path.join(data_path, f"{mode}_gt_embeddings")
    if not os.path.exists(output_gt_path):
        os.mkdir(output_gt_path)
    output_gt_path = os.path.join(output_gt_path, f"{mode}_gt_embeddings")
    if not os.path.exists(output_gt_path):
        os.mkdir(output_gt_path)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    images_list = os.listdir(images_path)
    gt_list = os.listdir(gt_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_height, resize_width), antialias=True)
    ])

    for i in tqdm(range(len(images_list))):
        image = os.path.join(images_path, images_list[i])
        gt = os.path.join(gt_path, gt_list[i])

        image = plt.imread(image)
        gt = plt.imread(gt)

        image = transform(np.copy(image))
        gt = transform(np.copy(gt))

        image = torch.unsqueeze(image, dim=0).to(device)
        gt = torch.unsqueeze(gt, dim=0).to(device)

        if 'polyp' in data_path:
            # make the gt a single channel image (multipiled by 3 channels to be compatible with the vae input
            # dimensions)
            sum_lst = []
            for channel in range(gt.shape[1]):
                channel_sum = torch.sum(gt[0, channel])
                sum_lst.append(channel_sum)
            max_sum = max(sum_lst)
            max_sum_idx = sum_lst.index(max_sum)

            max_sum_channel = torch.squeeze(gt, dim=0)[max_sum_idx, :,:]
            max_sum_channel = torch.unsqueeze(max_sum_channel, dim=0)
            gt_single_channel = torch.cat((max_sum_channel, max_sum_channel, max_sum_channel), dim=0)
            gt_single_channel = torch.unsqueeze(gt_single_channel, dim=0)
        else:
            gt_single_channel = gt

        with torch.no_grad():
            image_embeddings = vae.encode(image).latent_dist.sample()
            gt_embeddings = vae.encode(gt_single_channel).latent_dist.sample()

        image_output_file = os.path.join(output_image_path, f"{images_list[i]}.pt")
        gt_output_file = os.path.join(output_gt_path, f"{gt_list[i]}.pt")

        torch.save(image_embeddings, image_output_file)
        torch.save(gt_embeddings, gt_output_file)

        if mode == "train":
            augmentations = ['contrast', 'brightness', 'saturation', 'hue_yellow', 'hue_red', 'flipping', 'rotation']

            for augmentation in augmentations:
                image_augmented = image.clone()
                gt_single_channel_augmented = gt_single_channel.clone()

                if augmentation == 'contrast':
                    contrast = random.uniform(0.5, 1.5)
                    image_augmented = TF.adjust_contrast(image_augmented, contrast)
                elif augmentation == 'brightness':
                    brightness = random.uniform(0.7, 1.5)
                    image_augmented = TF.adjust_brightness(image_augmented, brightness)
                elif augmentation == 'saturation':
                    saturation = random.uniform(1.1, 1.5)
                    image_augmented = TF.adjust_saturation(image_augmented, saturation)
                elif augmentation == 'hue_yellow':
                    hue = 0.07
                    image_augmented = TF.adjust_hue(image_augmented, hue)
                elif augmentation == 'hue_red':
                    hue = -0.04
                    image_augmented = TF.adjust_hue(image_augmented, hue)
                elif augmentation == 'flipping':
                    image_augmented = TF.hflip(image_augmented)
                    gt_single_channel_augmented = TF.hflip(gt_single_channel_augmented)
                elif augmentation == 'rotation':
                    angle = random.randint(0, 360)
                    image_augmented = TF.rotate(image_augmented, angle)
                    gt_single_channel_augmented = TF.rotate(gt_single_channel_augmented, angle)

                with torch.no_grad():
                    image_embeddings = vae.encode(image_augmented).latent_dist.sample()
                    gt_embeddings = vae.encode(gt_single_channel_augmented).latent_dist.sample()

                augmentation_string = '_'.join(augmentation)
                image_output_file = os.path.join(output_image_path, f"{images_list[i]}_{augmentation_string}.pt")
                gt_output_file = os.path.join(output_gt_path, f"{gt_list[i]}_{augmentation_string}.pt")

                torch.save(image_embeddings, image_output_file)
                torch.save(gt_embeddings, gt_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/Kvasir-SEG")
    parser.add_argument("--images_path", type=str, default="./data/Kvasir-SEG/images")
    parser.add_argument("--gt_path", type=str, default="./data/Kvasir-SEG/masks")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--resize", type=int, default=256)

    args = parser.parse_args()
    data_path = args.data_path
    images_path = args.images_path
    gt_path = args.gt_path
    train_fraction = args.train_fraction
    resize_height = args.resize
    resize_width = args.resize

    split_images(data_path, images_path, gt_path, train_fraction)

    train_images_path = os.path.join(data_path, "train_images")
    train_gt_path = os.path.join(data_path, "train_gt_images")
    save_embedded_images(data_path, train_images_path, train_gt_path, "train", resize_height, resize_width)

    test_images_path = os.path.join(data_path, "test_images")
    test_gt_path = os.path.join(data_path, "test_gt_images")
    save_embedded_images(data_path, test_images_path, test_gt_path, "test", resize_height, resize_width)
