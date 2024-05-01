import matplotlib.pyplot as plt
import torch
import os.path
from torchvision.transforms import transforms
from diffusers.models import AutoencoderKL
import random
import torchvision.transforms.functional as TF


def save_embedded_images(data_path, images_path, gt_path, resize_height=512, resize_width=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_image_path = os.path.join(data_path, "train_embeddings")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)
    output_image_path = os.path.join(output_image_path, "train_embeddings")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)

    output_gt_path = os.path.join(data_path, "train_gt_embeddings")
    if not os.path.exists(output_gt_path):
        os.mkdir(output_gt_path)
    output_gt_path = os.path.join(output_gt_path, "train_gt_embeddings")
    if not os.path.exists(output_gt_path):
        os.mkdir(output_gt_path)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    images_list = os.listdir(images_path)
    gt_list = os.listdir(gt_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_height, resize_width), antialias=True)
    ])

    for i in range(len(images_list)):
        if i % 100 == 0:
            print(f"Iteration {i}/{len(images_list)}")
        image = os.path.join(images_path, images_list[i])
        gt = os.path.join(gt_path, gt_list[i])

        image = plt.imread(image)
        gt = plt.imread(gt)

        image = transform(image)
        gt = transform(gt)

        image = torch.unsqueeze(image, dim=0).to(device)
        gt = torch.unsqueeze(gt, dim=0).to(device)

        # # check if all channels in gt are the same
        # if torch.all(gt[0, 0, :, :] == gt[0, 1, :, :]) and torch.all(gt[0, 1, :, :] == gt[0, 2, :, :]):
        #     continue

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

        # # plot the single channel gt
        # plt.figure()
        # plt.imshow(torch.permute(torch.squeeze(gt_single_channel.cpu(), dim=0), (1, 2, 0)))
        # plt.show()

        with torch.no_grad():
            image_embeddings = vae.encode(image).latent_dist.sample()
            gt_embeddings = vae.encode(gt_single_channel).latent_dist.sample()

        # # plot the gt
        # plt.figure()
        # plt.imshow(torch.permute(torch.squeeze(gt, dim=0), (1, 2, 0)).cpu().detach().numpy() * 255)
        # plt.show()
        #
        # gt=torch.squeeze(gt,dim=0)
        # # create a binary image
        # gt = torch.sqrt(gt[0, :, :] ** 2 + gt[1, :, :] ** 2 + gt[2, :, :] ** 2)
        # gt[gt < 0.5] = 0
        # gt[gt >= 0.5] = 1
        # plt.figure()
        # plt.imshow(gt.cpu().detach().numpy())
        # plt.show()

        image_output_file = os.path.join(output_image_path, f"{images_list[i]}.pt")
        gt_output_file = os.path.join(output_gt_path, f"{gt_list[i]}.pt")

        torch.save(image_embeddings, image_output_file)
        torch.save(gt_embeddings, gt_output_file)

        # image_embeddings = torch.load(image_output_file)
        # gt_embeddings = torch.load(gt_output_file)
        # with torch.no_grad():
        #     image = vae.decode(image_embeddings).sample
        #     gt = vae.decode(gt_embeddings).sample
        #
        # plt.imshow(torch.permute(torch.squeeze(image.cpu().detach(), dim=0), (1, 2, 0)))
        # plt.imshow(torch.permute(torch.squeeze(gt.cpu().detach(), dim=0), (1, 2, 0)))
        # plt.show()

#         create augmentations for the images and gt
        augmentations = ['image_contrast', 'image_brightness', 'image_saturation', 'image_hue_yellow',
                                      'image_hue_red', 'image_flipping', 'image_rotation']
        for augmentation in augmentations:
            if augmentation == 'image_contrast':
                contrast = random.uniform(0.5, 1.5)
                image = TF.adjust_contrast(image, contrast)
            elif augmentation == 'image_brightness':
                brightness = random.uniform(0.7, 1.5)
                image = TF.adjust_brightness(image, brightness)
            elif augmentation == 'image_saturation':
                saturation = random.uniform(1.1, 1.5)
                image = TF.adjust_saturation(image, saturation)
            # make the image more yellow
            elif augmentation == 'image_hue_yellow':
                hue = 0.07
                image = TF.adjust_hue(image, hue)
            # make the image more red
            elif augmentation == 'image_hue_red':
                hue = -0.04
                image = TF.adjust_hue(image, hue)
            elif augmentation == 'image_flipping':
                image = TF.hflip(image)
                gt_single_channel = TF.hflip(gt_single_channel)
            elif augmentation == 'image_rotation':
                angle = random.randint(0, 360)
                image = TF.rotate(image, angle)
                gt_single_channel = TF.rotate(gt_single_channel, angle)

            with torch.no_grad():
                image_embeddings = vae.encode(image).latent_dist.sample()
                gt_embeddings = vae.encode(gt_single_channel).latent_dist.sample()

            image_output_file = os.path.join(output_image_path, f"{images_list[i]}_{augmentation}.pt")
            gt_output_file = os.path.join(output_gt_path, f"{gt_list[i]}_{augmentation}.pt")

            torch.save(image_embeddings, image_output_file)
            torch.save(gt_embeddings, gt_output_file)


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "kvasir-seg")
    images_path = os.path.join(data_path, "images")
    gt_path = os.path.join(data_path, "masks")
    resize_height = 256
    resize_width = 256

    save_embedded_images(data_path, images_path, gt_path, resize_height, resize_width)
