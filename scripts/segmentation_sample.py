

import argparse
import os
from ssl import OP_NO_TLSv1

import matplotlib.pyplot as plt
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random

import torch
import random
from torch.utils.data import DataLoader, random_split

sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from unet import UNetModel
from torchsummary import summary
from polyp_dataset import polyp_dataset
seed=42
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
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

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test)#, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5

    elif args.data_name == "POLYP":
        images_path = r"D:\Hila\guided-diffusion\datasets\polyps\train\train"
        gt_path = r"D:\Hila\guided-diffusion\datasets\polyps\train_gt\train_gt"
        images_embeddings_path = r"D:\Hila\guided-diffusion\datasets\polyps\dataset_embeddings\train_embeddings\train_embeddings"
        gt_embeddings_path = r"D:\Hila\guided-diffusion\datasets\polyps\dataset_embeddings\train_gt_embeddings\train_gt_embeddings"
        new_image_height = 64
        new_image_width = 64
        guided = False
        ds = polyp_dataset(
            images_path=images_path,
            gt_path=gt_path,
            images_embeddings_path=images_embeddings_path,
            gt_embeddings_path=gt_embeddings_path,
            new_image_height=new_image_height,
            new_image_width=new_image_width,
            guided=guided
        )
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)


    train_ratio = 0.8
    # Calculate the number of samples for each set
    num_train = int(len(datal.dataset) * train_ratio)
    num_valid = len(datal.dataset) - num_train

    # Split the dataset into train and validation sets
    train_set, valid_set = random_split(datal.dataset, [num_train, num_valid])
    datal = th.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    d_cond = 3
    model = UNetModel(in_channels=1, out_channels=1, channels=32, n_res_blocks=3, attention_levels=[0, 1, 2],
                      channel_multipliers=[2, 4, 6], condition_channels=3, n_heads=1, d_cond=d_cond)
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
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

    for batch_idx, (gt, image, path) in enumerate(data):
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                image.shape,
                gt.shape,
                image,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            # make sample a binary image
            sample = torch.where(sample > 0.5, 1, 0)

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            co = th.tensor(cal_out)
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
            else:
                enslist.append(co)

        if args.debug:
            if args.data_name == "POLYP":
                #   plot the original image, the ground truth and the generated masks from samples_lst
                fig, axis = plt.subplots(1, 2 + args.num_ensemble, figsize=(15, 15))
                axis[0].imshow(torch.permute(image[0].cpu().detach().mul_(255).byte(), (1, 2, 0)))
                axis[0].set_title('Original Image')
                axis[1].imshow(gt[0].permute(1, 2, 0).cpu().detach(), cmap='Greys', interpolation='nearest')
                axis[1].set_title('Ground Truth')
                for i in range(args.num_ensemble):
                    axis[2 + i].imshow(enslist[i][0].cpu().detach(), cmap='Greys',
                                       interpolation='nearest')
                    axis[2 + i].set_title(f'Mask {i + 1}')
                # make the distance between the subplots larger
                plt.subplots_adjust(wspace=0.8)
                # make each subplot larger
                plt.tight_layout()
                plt.suptitle(f'Results using {args.model_path}')
                plt.show()
                plt.close()








def create_argparser():
    defaults = dict(
        data_name='POLYP',
        data_dir="../dataset",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./results/savedmodel002000.pt",         #path to pretrain model
        num_ensemble=1,      #number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None, #"0,1,2"
        debug=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()