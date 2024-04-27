import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from models.DiT import DiT_models
from diffusers.models import AutoencoderKL
from polyp_dataset import polyp_dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import random
import argparse
from utils import *


torch.manual_seed(42)


class Trainer:
    def __init__(self, model, model_name, load_pretrained_model, gpu_id, data_path, criterion, batch_size, lr, epochs,
                 num_training_steps, beta_start, beta_end, num_testing_steps, guided, apply_cfg, cfg_prob, cfg_scale):
        self.model = model.to(gpu_id)
        self.model_name = model_name
        self.load_pretrained_model = load_pretrained_model
        self.gpu_id = gpu_id
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_training_steps = num_training_steps
        self.num_testing_steps = num_testing_steps
        self.guided = guided
        self.apply_cfg = apply_cfg
        self.cfg_prob = cfg_prob
        self.cfg_scale = cfg_scale

        self.model_parameters = ModelParameters(model_name, batch_size, lr, epochs, num_training_steps, beta_start,
                                                beta_end, num_testing_steps, guided, apply_cfg, cfg_prob)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.load_pretrained_model:
            self.load_model()

        self.train_dataset = polyp_dataset(
            data_path=data_path,
            mode="train"
        )
        self.test_dataset = polyp_dataset(
            data_path=data_path,
            mode="test"
        )
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           sampler=DistributedSampler(self.train_dataset))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                          sampler=DistributedSampler(self.test_dataset))

        self.sampler = DDPMScheduler(
            num_train_timesteps=self.num_training_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=False
        )

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
        self.vae.to(self.gpu_id)

        self.writer = SummaryWriter(f"runs/{self.model_name}")

        self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

    def train_one_epoch(self, epoch, images_list):
        self.model.train()

        dataloader = self.train_dataloader
        dataloader.sampler.set_epoch(epoch)

        print_every = len(dataloader) // 10
        total_loss = 0

        for batch_idx, (gt, image) in enumerate(dataloader):
            if batch_idx % print_every == 0 and batch_idx != 0:
                average_loss = total_loss / batch_idx
                # print(f"| GPU[{self.gpu_id}] | Epoch {epoch} | Loss {average_loss:.5f} |")

            gt = gt.to(self.gpu_id)
            image = image.to(self.gpu_id)

            timestep = torch.randint(0, self.num_training_steps, (image.size(0), )).to(self.gpu_id)
            noise = torch.FloatTensor(torch.randn(gt.shape, dtype=torch.float32)).to(self.gpu_id)
            noisy_gt = self.sampler.add_noise(gt, noise, timestep)

            self.optimizer.zero_grad()

            if self.apply_cfg:
                image = image * create_cfg_mask(image.shape, self.cfg_prob, self.gpu_id)

            noise_prediction = self.model(noisy_gt.to(self.gpu_id), timestep, image.to(self.gpu_id))
            if batch_idx == 0:
                images_list.append(torch.unsqueeze(noise_prediction[0], dim=0))


            loss = self.criterion(noise_prediction.to(self.gpu_id), noise)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader), images_list

    def create_GIF(vae, images_list, path, device):
        print("Creating GIF")
        directory = os.path.join(os.getcwd(), "GIF images")
        if os.path.exists(directory):
            delete_dir(directory)
            os.mkdir(directory)
        else:
            os.mkdir(directory)

        images = []
        for i in tqdm(range(len(images_list))):
            image_path = os.path.join(os.getcwd(), "GIF images", f"{i}.png")

            noisy_image = images_list[i] / 0.18125
            prediction = vae.decode(noisy_image.to(device)).sample

            prediction = prediction[0]
            prediction = torch.sqrt(prediction[0, :, :] ** 2 + prediction[1, :, :] ** 2 + prediction[2, :, :] ** 2)
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1
            prediction = prediction.cpu().detach()

            plt.figure()
            plt.imshow(images_list[i])
            plt.savefig(image_path)
            plt.close()

            images.append(imageio.imread(image_path))

        imageio.mimsave(path, images, duration=1)
        print(f"Saved GIF to {path}")
        delete_dir(directory)

    def delete_dir(path):
        if os.path.exists(path):
            file_list = os.listdir(path)
            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        os.rmdir(path)

    def train_one_epoch_with_augmentations(self, epoch, augmentation, images_list):
        self.model.train()

        dataloader = self.train_dataloader
        dataloader.sampler.set_epoch(epoch)

        print_every = len(dataloader) // 10
        total_loss = 0

        for batch_idx, (gt, image) in enumerate(dataloader):
            if batch_idx % print_every == 0 and batch_idx != 0:
                average_loss = total_loss / batch_idx
                # print(f"| GPU[{self.gpu_id}] | Epoch {epoch} | Loss {average_loss:.5f} |")

            gt = gt.to(self.gpu_id)
            image = image.to(self.gpu_id)

            # train the model on a given augmentation of the images in the batch
            if augmentation == 'image_shifting':
                shift = random.randint(-10, 10)
                image = TF.affine(image, angle=0, translate=[shift, 0], scale=1, shear=0)
                gt = TF.affine(gt, angle=0, translate=[shift, 0], scale=1, shear=0)
            elif augmentation == 'image_flipping':
                image = TF.hflip(image)
                gt = TF.hflip(gt)
            elif augmentation == 'image_rotation':
                angle = random.randint(0, 360)
                image = TF.rotate(image, angle)
                gt = TF.rotate(gt, angle)
            elif augmentation == 'image_blurring':
                image = TF.gaussian_blur(image, 5)
                gt = TF.gaussian_blur(gt, 5)


            timestep = torch.randint(0, self.num_training_steps, (image.size(0), )).to(self.gpu_id)
            noise = torch.FloatTensor(torch.randn(gt.shape, dtype=torch.float32)).to(self.gpu_id)
            noisy_gt = self.sampler.add_noise(gt, noise, timestep)

            self.optimizer.zero_grad()

            if self.apply_cfg:
                image = image * create_cfg_mask(image.shape, self.cfg_prob, self.gpu_id)

            noise_prediction = self.model(noisy_gt.to(self.gpu_id), timestep, image.to(self.gpu_id))
            if batch_idx == 0:
                images_list.append(torch.unsqueeze(noise_prediction[0], dim=0))

            loss = self.criterion(noise_prediction.to(self.gpu_id), noise)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader), images_list

    def train(self):
        print(f"Start Training {self.model_name}...")

        if not os.path.exists(os.path.join(os.getcwd(), "saved_models")):
            os.mkdir(os.path.join(os.getcwd(), "saved_models"))
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "saved_models", self.model_name))

        training_avg_loss = 0

        images_list = []

        for epoch in range(self.epochs):
            print("-" * 40)

            training_avg_loss, images_list = self.train_one_epoch(epoch, images_list)

            if epoch == 0:
                print(f"| GPU[{self.gpu_id}] | initiative Loss {training_avg_loss:.5f} |")

            # augmentations = ['image_shifting', 'image_flipping', 'image_rotation', 'image_blurring']
            # for augmentation in augmentations:
            #     training_avg_loss, images_list = self.train_one_epoch_with_augmentations(epoch, augmentation, images_list)

            print("-" * 40)
            print(f"| End of epoch {epoch} | Loss {training_avg_loss:.5f} |")

            self.writer.add_scalars(f"Loss/{self.model_name}", {"Train Loss": training_avg_loss}, epoch)
            if self.gpu_id == 0:
                self.save_model()

        create_GIF(self.vae, images_list, os.path.join(os.getcwd(), "saved_models", self.model_name, "GIF.gif"), self.gpu_id)

        self.model_parameters.write_parameters(training_avg_loss)

    def save_model(self):
        checkpoint_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        state = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)

    def load_model(self):
        print(f"Loading model {self.model_name}...")
        model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        assert os.path.exists(model_path), f"Model {self.model_name}.pt does not exist."

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded Model Successfully.")


def main(rank: int, world_size: int, args):
    print(f"Detected {world_size} {'GPU' if world_size == 1 else 'GPUs'}")
    setup(rank, world_size)

    data_path = args.data_path
    model_name = args.model_name  # AdaptiveLayerNormalization_B4
    batch_size = args.batch_size
    lr = 1e-4
    beta_start = 10 ** -4
    beta_end = 2 * 10 ** -2
    epochs = args.epochs
    num_training_steps = 1000
    num_testing_steps = 100
    criterion = nn.MSELoss()
    apply_cfg = True
    cfg_prob = 0.1
    cfg_scale = 3
    guided = True
    load_pretrained_model = args.load_pretrained

    if "PolypDiT_B2" in model_name:
        model = DiT_models['DiT-B/2'](in_channels=4, condition_channels=4, learn_sigma=False)
    if "PolypDiT_B4" in model_name:
        model = DiT_models['DiT-B/4'](in_channels=4, condition_channels=4, learn_sigma=False)
    handler = Trainer(model=model,
                      model_name=model_name,
                      load_pretrained_model=load_pretrained_model,
                      gpu_id=rank,
                      data_path=data_path,
                      criterion=criterion,
                      batch_size=batch_size,
                      lr=lr,
                      epochs=epochs,
                      num_training_steps=num_training_steps,
                      beta_start=beta_start,
                      beta_end=beta_end,
                      num_testing_steps=num_testing_steps,
                      guided=guided,
                      apply_cfg=apply_cfg,
                      cfg_prob=cfg_prob,
                      cfg_scale=cfg_scale)

    print(f"Training Model: {model_name}\nData Path: {data_path}\nBatch Size: {batch_size}\nEpochs {epochs}\n"
          f"Pretrained: {load_pretrained_model}")
    handler.train()
    cleanup()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Did not find a GPU"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="PolypDiT_B2_one_channel_no_augmentations")
    parser.add_argument("--data-path", type=str, default="./data/polyps")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-pretrained", type=bool, default=False)

    args = parser.parse_args()
    args.model_name = args.model_name + "_" + str(args.epochs) + "epochs"

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args,), nprocs=world_size)

    # model = Denoiser(image_size=32,
    #                  noise_embed_dims=4,
    #                  patch_size=4,
    #                  embed_dim=768,
    #                  dropout=0.2,
    #                  n_layers=12,
    #                  text_emb_size=4096)
    # model = DiT_POLYP(batch_size=batch_size)
    # model = UNetModel(in_channels=4, out_channels=4, channels=32, n_res_blocks=3, attention_levels=[0, 1, 2],
    #                   channel_multipliers=[2, 4, 6], n_heads=1, d_cond=4)
