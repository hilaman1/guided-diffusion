import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
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
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms

from models.unet import UNetModel
from models.DiT import DiT_models
from polyp_dataset import polyp_dataset


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif args.data_name == "POLYP":
        images_path = "C:\\Users\\Admin\\Documents\\GitHub\\diffusion\\data\\polyps\\train\\train"
        gt_path = "C:\\Users\\Admin\\Documents\\GitHub\\diffusion\\data\\polyps\\train_gt\\train_gt"
        images_embeddings_path = "C:\\Users\\Admin\\Documents\\GitHub\\diffusion\\data\\polyps\\train_embeddings\\train_embeddings"
        gt_embeddings_path = "C:\\Users\\Admin\\Documents\\GitHub\\diffusion\\data\\polyps\\train_gt_embeddings\\train_gt_embeddings"
        new_image_height = 64
        new_image_width = 64
        guided = False
        normalize = True
        ds = polyp_dataset(
            images_path=images_path,
            gt_path=gt_path,
            images_embeddings_path=images_embeddings_path,
            gt_embeddings_path=gt_embeddings_path,
            new_image_height=new_image_height,
            new_image_width=new_image_width,
            guided=guided,
            normalize=normalize,
        )
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ", args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    data = iter(dataloader)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model = UNetModel(in_channels=3, out_channels=3, channels=32, n_res_blocks=3, attention_levels=[0, 1, 2],
                      channel_multipliers=[2, 4, 6], condition_channels=3, n_heads=1, d_cond=3)
    # model = DiT_models["DiT-B/4"]

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
