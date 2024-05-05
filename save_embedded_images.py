import matplotlib.pyplot as plt
import torch
import os.path
from torchvision.transforms import transforms
from diffusers.models import AutoencoderKL
import random
import torchvision.transforms.functional as TF
from itertools import combinations



def save_embedded_images(data_path, images_path, gt_path, resize_height=512, resize_width=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_image_path = os.path.join(data_path, "test_embeddings")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)
    output_image_path = os.path.join(output_image_path, "test_embeddings")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)

    output_gt_path = os.path.join(data_path, "test_gt_embeddings")
    if not os.path.exists(output_gt_path):
        os.mkdir(output_gt_path)
    output_gt_path = os.path.join(output_gt_path, "test_gt_embeddings")
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
        augmentations = ['contrast', 'brightness', 'saturation', 'hue_yellow', 'hue_red', 'flipping', 'rotation']

        augmentation_combinations = []
        augmentation_combinations.extend(list(combinations(augmentations, 4)))
        # choose 8 from augmentation_combinations
        augmentation_combinations = random.sample(augmentation_combinations, 8)
        #
        #
        # for augmentation_combination in augmentation_combinations:
        #     image_augmented = image.clone()
        #     gt_single_channel_augmented = gt_single_channel.clone()
        #
        #     for augmentation in augmentation_combination:
        #         if augmentation == 'contrast':
        #             contrast_lst = [0.5, 1.5]
        #             rand_idx = random.randint(0, len(contrast_lst) - 1)
        #             contrast = contrast_lst[rand_idx]
        #             image_augmented = TF.adjust_contrast(image_augmented, contrast)
        #         elif augmentation == 'brightness':
        #             brightness_lst = [0.7, 0.8, 1.2, 1.5]
        #             # choose from the brightness list
        #             rand_idx = random.randint(0, len(brightness_lst) - 1)
        #             brightness = brightness_lst[rand_idx]
        #             image_augmented = TF.adjust_brightness(image_augmented, brightness)
        #         elif augmentation == 'saturation':
        #             saturation_lst = [0.5, 0.7, 1.5, 1.6]
        #             # choose from the saturation list
        #             rand_idx = random.randint(0, len(saturation_lst) - 1)
        #             saturation = saturation_lst[rand_idx]
        #             image_augmented = TF.adjust_saturation(image_augmented, saturation)
        #         elif augmentation == 'hue_yellow':
        #             hue = 0.07
        #             image_augmented = TF.adjust_hue(image_augmented, hue)
        #         elif augmentation == 'hue_red':
        #             hue = -0.04
        #             image_augmented = TF.adjust_hue(image_augmented, hue)
        #         elif augmentation == 'flipping':
        #             image_augmented = TF.hflip(image_augmented)
        #             gt_single_channel_augmented = TF.hflip(gt_single_channel_augmented)
        #         elif augmentation == 'rotation':
        #             angle_lst = [90, 180, 270]
        #             # choose from the angle list
        #             rand_idx = random.randint(0, len(angle_lst) - 1)
        #             angle = angle_lst[rand_idx]
        #             image_augmented = TF.rotate(image_augmented, angle)
        #             gt_single_channel_augmented = TF.rotate(gt_single_channel_augmented, angle)
        #     with torch.no_grad():
        #         image_embeddings = vae.encode(image_augmented).latent_dist.sample()
        #         gt_embeddings = vae.encode(gt_single_channel_augmented).latent_dist.sample()
        #
        #     augmentation_string = '_'.join(augmentation_combination)
        #     image_output_file = os.path.join(output_image_path, f"{images_list[i]}_{augmentation_string}.pt")
        #     gt_output_file = os.path.join(output_gt_path, f"{gt_list[i]}_{augmentation_string}.pt")
        #
        #     torch.save(image_embeddings, image_output_file)
        #     torch.save(gt_embeddings, gt_output_file)


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "kvasir-seg")
    images_path = os.path.join(data_path, "test_images")
    gt_path = os.path.join(data_path, "test_gt_images")
    resize_height = 256
    resize_width = 256

    save_embedded_images(data_path, images_path, gt_path, resize_height, resize_width)