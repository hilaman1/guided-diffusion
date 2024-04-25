
# credits: MedSegDiff

import cv2
import sys

sys.path.append(".")

import numpy as np
import torch

from torch.autograd import Function
import torchvision

from PIL import Image
import argparse
from polyp_dataset import polyp_dataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt


import os


def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.reshape(-1), target.reshape(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device=input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred, true_mask_p, threshold=(0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            cup_pred = vpred_cpu[:, 1, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            iou_d += iou(disc_pred, disc_mask)
            iou_c += iou(cup_pred, cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()

        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()

            # choose green or red channel based on the maximum sum of the channel
            sum_lst = []
            for channel in range(vpred_cpu.shape[1]):
                channel_sum = torch.sum(vpred_cpu[:, channel, :, :])
                sum_lst.append(channel_sum)
            max_sum = max(sum_lst)
            max_sum_idx = sum_lst.index(max_sum)
            vpred_cpu = vpred_cpu[:, max_sum_idx, :, :]

            sum_lst = []
            for channel in range(gt_vmask_p.shape[1]):
                channel_sum = torch.sum(gt_vmask_p[:, channel, :, :])
                sum_lst.append(channel_sum)
            max_sum = max(sum_lst)
            max_sum_idx = sum_lst.index(max_sum)
            gt_vmask_p = gt_vmask_p[:, max_sum_idx, :, :]

            disc_pred = vpred_cpu.numpy().astype('int32')

            disc_mask = gt_vmask_p.squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            eiou += iou(disc_pred, disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred_cpu, gt_vmask_p).item()

        return eiou / len(threshold), edice / len(threshold)


def main():
    data_path = r"D:\Hila\guided-diffusion\datasets\polyps"
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model_name", type=str, default="PolypDiT_B2")
    argParser.add_argument("--pred_path", type=str, default=r"D:\Hila\guided-diffusion\datasets\polyps\pred")
    argParser.add_argument("--data_path", type=str, default=r"D:\Hila\guided-diffusion\datasets\polyps")
    args = argParser.parse_args()
    mix_res = (0, 0)
    num = 0
    model_pred_path = os.path.join(f"{args.pred_path}", f"{args.model_name}")
    test_dataset = polyp_dataset(
        data_path=data_path,
        mode="test"
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    data_iter = iter(test_dataloader)

    for i in range(len(test_dataloader)):
        print(f"Sampling image {i + 1}")
        gt, image = next(data_iter)
        num += 1
        curr_prediction_path = os.path.join(model_pred_path, f"pred_{i + 1}.png")
        pred = plt.imread(curr_prediction_path)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        gt = gt / 0.18125
        gt = vae.decode(gt).sample

        gt = torch.permute(torch.squeeze(gt, dim=0), (0,2,1)).cpu().detach()
        pred = torch.permute(torch.from_numpy(np.copy(pred)), (2,1,0)).float()

        gt = torch.unsqueeze(gt, dim=0)
        pred = torch.unsqueeze(pred, dim=0)


        # # make the gt a binary image with one channel
        # gt = torch.where(gt > 0.5, 1, 0)
        # # calculate the sum of each channel in the gt
        # sum_lst = []
        # for channel in range(gt.shape[0]):
        #     channel_sum = torch.sum(gt[channel])
        #     sum_lst.append(channel_sum)
        # # get the channel with the maximum sum
        # max_sum = max(sum_lst)
        # max_sum_idx = sum_lst.index(max_sum)
        # gt = gt[max_sum_idx]
        #
        # # make the pred a binary image with one channel
        # pred = torch.where(pred > 0.5, 1, 0)
        # # calculate the sum of each channel in the pred
        # sum_lst = []
        # for channel in range(pred.shape[0]):
        #     channel_sum = torch.sum(pred[channel])
        #     sum_lst.append(channel_sum)
        # # get the channel with the maximum sum
        # max_sum = max(sum_lst)
        # max_sum_idx = sum_lst.index(max_sum)
        # pred = pred[max_sum_idx]

        # if args.debug:
        #     vutils.save_image(gt, fp = os.path.join('./results/' + str(ind)+'gt.jpg'), nrow = 1, padding = 10)
        temp = eval_seg(pred, gt)
        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
    iou, dice = tuple([a / num for a in mix_res])
    print('iou is', iou)
    print('dice is', dice)


if __name__ == "__main__":
    main()