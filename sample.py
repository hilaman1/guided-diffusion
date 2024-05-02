import numpy as np
import torch
from torch.utils.data import DataLoader
from models.DiT import DiT_models
import os
from utils import sample, create_GIF
from diffusers.models import AutoencoderKL
from diffusers import DDPMScheduler
from polyp_dataset import polyp_dataset
import matplotlib.pyplot as plt
import cv2


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
            mode="test"
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

    def sample(self, predictions_path):
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models", self.model_name, "samples")):
            os.mkdir(os.path.join(os.getcwd(), "saved_models", self.model_name, "samples"))

        model.eval()
        self.sampler.set_timesteps(len(range(num_training_steps - 1, 0, -int(num_training_steps / num_testing_steps))),
                                   self.device)
        data_iter = iter(self.test_dataloader)

        for i in range(len(self.test_dataloader)):
            print(f"Sampling image {i+1}")

            gt, image = next(data_iter)

            prediction, noise_images = sample(model, self.vae, self.sampler, image, self.num_training_steps,
                                              self.num_testing_steps, self.device, self.cfg_scale, self.guided)
            model_pred_path = os.path.join(predictions_path, model_name)
            if not os.path.exists(model_pred_path):
                os.mkdir(model_pred_path)
            curr_prediction_path = os.path.join(model_pred_path, f"pred_{i + 1}.png")

            prediction = torch.squeeze(prediction, dim=0)
            # create a binary image
            prediction = torch.sqrt(prediction[0, :, :] ** 2 + prediction[1, :, :] ** 2 + prediction[2, :, :] ** 2)
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1
            # plt.figure()
            # plt.imshow(gt.cpu().detach().numpy())
            # plt.show()

            cv2.imwrite(curr_prediction_path, prediction.cpu().detach().numpy()*255)

            # if i+1 in [1, 33, 36, 71, 159]:
            #     gif_path = os.path.join(os.getcwd(), "saved_models", self.model_name, "samples", f"{i+1}.gif")
            #     create_GIF(self.vae, noise_images, gif_path, self.device)

            gt = gt / 0.18125
            gt = self.vae.decode(gt).sample
            gt = torch.squeeze(gt, dim=0)
            gt = torch.sqrt(gt[0, :, :] ** 2 + gt[1, :, :] ** 2 + gt[2, :, :] ** 2)
            gt[gt < 0.5] = 0
            gt[gt >= 0.5] = 1

            image = image / 0.18125
            image = self.vae.decode(image).sample

            fig, axis = plt.subplots(1, 3)
            axis[2].imshow(prediction.cpu().detach())
            axis[2].set_title("Prediction")
            axis[1].imshow(gt.cpu().detach())
            axis[1].set_title("GT")
            axis[0].imshow(torch.permute(torch.squeeze(image, dim=0), (1, 2, 0)).cpu().detach())
            axis[0].set_title("Image")
            plt.savefig(os.path.join(os.getcwd(), "saved_models", self.model_name, "samples", f"{i+1}.png"))




if __name__ == "__main__":
    model_name = "KvasirDiT_B2_with_augmentations_150epochs"

    if "DiT_B2" in model_name:
        model = DiT_models['DiT-B/2'](in_channels=4, condition_channels=4, learn_sigma=False)
    if "DiT_B4" in model_name:
        model = DiT_models['DiT-B/4'](in_channels=4, condition_channels=4, learn_sigma=False)

    data_path = os.path.join(os.getcwd(), "data", "kvasir-seg")
    predictions_path = os.path.join(os.getcwd(), "data", "kvasir-seg", "pred")
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

    sampler.sample(predictions_path)