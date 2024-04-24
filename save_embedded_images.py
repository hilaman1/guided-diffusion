import matplotlib.pyplot as plt
import torch
import os.path
from torchvision.transforms import transforms
from diffusers.models import AutoencoderKL


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

        with torch.no_grad():
            image_embeddings = vae.encode(image).latent_dist.sample()
            gt_embeddings = vae.encode(gt).latent_dist.sample()

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


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "polyps")
    images_path = os.path.join(data_path, "train", "train")
    gt_path = os.path.join(data_path, "train_gt", "train_gt")
    resize_height = 256
    resize_width = 256

    save_embedded_images(data_path, images_path, gt_path, resize_height, resize_width)
