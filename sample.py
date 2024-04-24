import torch
from torch.utils.data import DataLoader
from models.DiT import DiT_models
import os
from utils import sample, create_GIF
from diffusers.models import AutoencoderKL
from diffusers import DDPMScheduler
from polyp_dataset import polyp_dataset
import matplotlib.pyplot as plt


torch.manual_seed(42)


class Sampler:
    def __init__(self, model, model_name, data_path, beta_start, beta_end, num_training_steps, num_testing_steps,
                 cfg_scale, guided):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        self.cfg_scale = cfg_scale
        self.guided = guided

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(self.device)

        test_dataset = polyp_dataset(
            data_path=data_path,
            mode="train"
        )
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_training_steps = num_training_steps
        self.num_testing_steps = num_testing_steps
        self.sampler = DDPMScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=False
        )

        model.to(self.device)
        self.load_model()

    def load_model(self):
        print(f"Loading model {self.model_name}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    def sample(self, index):
        print(f"Sampling image {index}")
        model.eval()
        self.sampler.set_timesteps(len(range(num_training_steps - 1, 0, -int(num_training_steps / num_testing_steps))),
                                   self.device)
        data_iter = iter(self.test_dataloader)
        gt = None
        image = None
        for i in range(index):
            gt, image = next(data_iter)
        decoded_gt, noise_images = sample(model, self.vae, self.sampler, image, self.num_training_steps,
                                          self.num_testing_steps, self.device, self.cfg_scale, self.guided)

        gif_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.gif")
        create_GIF(self.vae, noise_images, gif_path, self.device)

        fig, axis = plt.subplots(1, 3)
        axis[0].imshow(torch.permute(torch.squeeze(decoded_gt, dim=0), (1, 2, 0)).cpu().detach())
        axis[1].imshow(torch.permute(torch.squeeze(gt, dim=0), (1, 2, 0)).cpu().detach())
        axis[2].imshow(torch.permute(torch.squeeze(image, dim=0), (1, 2, 0)).cpu().detach())
        plt.show()



if __name__ == "__main__":
    model = DiT_models['DiT-B/4'](condition_channels=4, learn_sigma=False)
    model_name = f"AdaptiveLayerNormalization_B4"
    data_path = os.path.join(os.getcwd(), "data", "polyps")
    beta_start = 10 ** -4
    beta_end = 2 * 10 ** -2
    num_training_steps = 1000
    num_testing_steps = 100
    cfg_scale = 3.0
    guided = True

    sampler = Sampler(
        model=model,
        model_name=model_name,
        data_path=data_path,
        beta_start=beta_start,
        beta_end=beta_end,
        num_training_steps=num_training_steps,
        num_testing_steps=num_testing_steps,
        cfg_scale=cfg_scale,
        guided=guided
    )

    index = 3
    sampler.sample(index=index)