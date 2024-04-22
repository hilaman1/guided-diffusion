import argparse
import os
import sys
import random
sys.path.append(".")
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from models.unet import UNetModel
from models.DiT import DiT_models
from polyp_dataset import polyp_dataset
from ISIC_dataset import ISIC_Dataset
from diffusers import DDPMScheduler

seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.data_name == "POLYP":
        images_path = os.path.join(os.getcwd(), "data", "polyps", "train", "train")
        gt_path = os.path.join(os.getcwd(), "data", "polyps", "train_gt", "train_gt")
        images_embeddings_path = os.path.join(os.getcwd(), "data", "polyps", "train_embeddings", "train_embeddings")
        gt_embeddings_path = os.path.join(os.getcwd(), "data", "polyps", "train_gt_embeddings", "train_gt_embeddings")
        new_image_height = 64
        new_image_width = 64
        guided = False
        normalize = True
        binary_seg = True
        ds = polyp_dataset(
            images_path=images_path,
            gt_path=gt_path,
            images_embeddings_path=images_embeddings_path,
            gt_embeddings_path=gt_embeddings_path,
            new_image_height=new_image_height,
            new_image_width=new_image_width,
            guided=guided,
            normalize=normalize,
            binary_seg=binary_seg
        )

    elif args.data_name == "ISIC":
        path = os.path.join(os.getcwd(), "data")
        new_image_height = 64
        new_image_width = 64
        mode = "train"
        cfg = False

        ds = ISIC_Dataset(path=path,
                          height=new_image_height,
                          width=new_image_width,
                          mode=mode,
                          cfg=cfg)

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    logger.log("creating model and diffusion...")

    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )

    diffusion = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="linear",
        clip_sample=True
    )

    condition_channels = 3
    seg_channels = 1
    unet_channels = 32
    res_blocks = 3
    attention_levels = [0, 1, 2]
    channel_multipliers = [2, 4, 6]
    n_heads = 1
    d_cond = 3
    # model = UNetModel(in_channels=seg_channels,
    #                   out_channels=seg_channels,
    #                   channels=unet_channels,
    #                   n_res_blocks=res_blocks,
    #                   attention_levels=attention_levels,
    #                   channel_multipliers=channel_multipliers,
    #                   condition_channels=condition_channels,
    #                   n_heads=n_heads,
    #                   d_cond=d_cond)
    model = DiT_models["DiT-B/4"](input_size=new_image_height,
                                  in_channels=1,
                                  condition_channels=3,
                                  learn_sigma=False)
    state_dict = dist_util.load_state_dict(args.model_path, map_location=device)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()
    sampling_steps = 50
    diffusion.set_timesteps(sampling_steps, device)
    for batch_idx, (seg, condition, path) in enumerate(dataloader):
        logger.log("sampling...")

        seg = seg.to(device)
        condition = condition.to(device)
        noise = torch.FloatTensor(torch.randn(seg.shape, dtype=torch.float32)).to(device)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
        #     model_kwargs = {}
            # start.record()
            # sample_fn = (
            #     diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            # )
            # sample, x_noisy, org, cal, cal_out = sample_fn(
            #     model,
            #     seg.shape,
            #     condition,
            #     step=1000,
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs=model_kwargs,
            # )
            #
            # end.record()


        for timestep in range(1000 - 1, 0, -int(1000 / sampling_steps)):
            model_input = torch.cat((condition, noise), dim=1)
            with torch.no_grad():
                t = torch.tensor([timestep], device=device)
                noise_prediction = model(model_input, t)
                print(timestep)

                noise = diffusion.step(noise_prediction, int(timestep), noise, return_dict=False)[0]

        with torch.no_grad():
            model_input = torch.cat((condition, noise), dim=1)
            timestep = 0
            t = torch.tensor([timestep], device=device)
            noise_prediction = model(model_input, t)

            sample = diffusion.step(noise_prediction, int(timestep), noise, return_dict=False)[0]
            torch.cuda.synchronize()

            numpy_seg = torch.permute(seg[0].cpu().detach(), (1, 2, 0)).numpy()
            numpy_sample = torch.permute(sample[0].cpu().detach(), (1, 2, 0)).numpy()

            fig, axis = plt.subplots(1, 2)
            axis[0].imshow(numpy_seg)
            axis[1].imshow(numpy_sample)
            plt.show()

def create_argparser():
    defaults = dict(
        data_name='ISIC',
        data_dir="../dataset",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./results/savedmodel006000.pt",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None, #"0,1,2"
        debug=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()