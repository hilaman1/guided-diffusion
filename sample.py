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
from models.DiT_cross import DiT_cross_models
import argparse
from tqdm import tqdm


torch.manual_seed(42)


class Sampler:
    def __init__(self, model, model_name, data_path, beta_start, beta_end, num_training_steps, num_testing_steps,
                 cfg_scale, guided, ema):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        self.cfg_scale = cfg_scale
        self.guided = guided
        self.ema = ema

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(self.device)

        test_dataset = polyp_dataset(
            data_path=data_path,
            mode="test",
            device=self.device
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
        print(f"Loading model {self.model_name} {'(EMA version)' if self.ema else ''}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if self.ema:
            self.model.load_state_dict(checkpoint["ema"])
        else:
            self.model.load_state_dict(checkpoint["model"])

    def sample(self, predictions_path):
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models", self.model_name, "samples")):
            os.mkdir(os.path.join(os.getcwd(), "saved_models", self.model_name, "samples"))
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)

        model.eval()
        self.sampler.set_timesteps(len(range(num_training_steps - 1, 0, -int(num_training_steps / num_testing_steps))),
                                   self.device)
        data_iter = iter(self.test_dataloader)

        for i in tqdm(range(len(self.test_dataloader))):
            gt, image = next(data_iter)

            prediction, noise_images = sample(model, self.vae, self.sampler, image, self.num_training_steps,
                                              self.num_testing_steps, self.device, self.cfg_scale, self.guided)
            model_pred_path = os.path.join(predictions_path, self.model_name)
            if not os.path.exists(model_pred_path):
                os.mkdir(model_pred_path)
            curr_prediction_path = os.path.join(model_pred_path, f"pred_{i + 1}.png")

            prediction = torch.squeeze(prediction, dim=0)
            # create a binary image
            prediction = torch.sqrt(prediction[0, :, :] ** 2 + prediction[1, :, :] ** 2 + prediction[2, :, :] ** 2)
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1

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
            image = torch.permute(torch.squeeze(image, dim=0), (1, 2, 0)).cpu().detach()
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            image = np.asarray(255 * image.numpy(), dtype=np.uint8)

            fig, axis = plt.subplots(1, 3)
            axis[2].imshow(prediction.cpu().detach())
            axis[2].set_title("Prediction")
            axis[1].imshow(gt.cpu().detach())
            axis[1].set_title("GT")
            axis[0].imshow(image)
            axis[0].set_title("Image")
            plt.savefig(os.path.join(os.getcwd(), "saved_models", self.model_name, "samples", f"{i+1}.png"))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="DiT_S8_CROSS_Kvasir")
    parser.add_argument("--model", type=str, default="DiT_S8", choices=["DiT_XL2", "DiT_XL4", "DiT_XL8",
                                                                        "DiT_L2", "DiT_L4", "DiT_L8",
                                                                        "DiT_B2", "DiT_B4", "DiT_B8",
                                                                        "DiT_S2", "DiT_S4", "DiT_S8"])
    parser.add_argument("--data-path", type=str, default="./data/Kvasir-SEG")
    parser.add_argument("--cross-model", type=bool, default=True)
    parser.add_argument("--ema", type=bool, default=False)

    args = parser.parse_args()
    model_type = args.model
    model_names = {
        "DiT_XL2": "DiT-XL/2", "DiT_XL4": "DiT-XL/4", "DiT_XL8": "DiT-XL/8",
        "DiT_L2": "DiT-L/2", "DiT_L4": "DiT-L/4", "DiT_L8": "DiT-L/8",
        "DiT_B2": "DiT-B/2", "DiT_B4": "DiT-B/4", "DiT_B8": "DiT-B/8",
        "DiT_S2": "DiT-S/2", "DiT_S4": "DiT-S/4", "DiT_S8": "DiT-S/8",
    }
    if args.cross_model:
        model = DiT_cross_models[model_names[model_type]](in_channels=4, condition_channels=4, learn_sigma=False)
    else:
        model = DiT_models[model_names[model_type]](in_channels=4, condition_channels=4, learn_sigma=False)

    predictions_path = os.path.join(os.getcwd(), "data", "Kvasir-SEG", "pred")
    beta_start = 10 ** -4
    beta_end = 2 * 10 ** -2
    num_training_steps = 1000
    num_testing_steps = 100
    cfg_scale = 3.0
    guided = True

    sampler = Sampler(
        model=model,
        model_name=args.model_name,
        data_path=args.data_path,
        beta_start=beta_start,
        beta_end=beta_end,
        num_training_steps=num_training_steps,
        num_testing_steps=num_testing_steps,
        cfg_scale=cfg_scale,
        guided=guided,
        ema=args.ema
    )

    print(f"Model Name: {args.model_name}\nModel: {args.model}\nData Path: {args.data_path}\n"
          f"Cross Model: {args.cross_model}\nEMA: {args.ema}")
    sampler.sample(predictions_path)
