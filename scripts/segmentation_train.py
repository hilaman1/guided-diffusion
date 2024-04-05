import os
import os.path
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from guided_diffusion.train_util import TrainLoop

from models.unet import UNetModel
from models.DiT import DiT_models
from polyp_dataset import polyp_dataset


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

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
        
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    data = iter(dataloader)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
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
                                  in_channels=seg_channels,
                                  condition_channels=condition_channels,
                                  learn_sigma=False)

    if args.multi_gpu:
        model = nn.DataParallel(model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name='POLYP',
        data_dir="C:\\Users\\Admin\\Documents\\GitHub\\guided-diffusion\\datasets",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=1000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None, #"0,1,2"
        out_dir='./results/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
