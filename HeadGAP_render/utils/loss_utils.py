#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips

lpips_fn = lpips.LPIPS(net='vgg').cuda()  # 如果没有 GPU，请移除 `.cuda()`

def compute_lpips_loss(pred_image, gt_image,cuda_index):
    """
    计算 LPIPS 损失。

    参数：
        pred_image (torch.Tensor): 网络输出图像，形状为 [3, H, W]。
        gt_image (torch.Tensor): Ground Truth 图像，形状为 [3, H, W]。

    返回：
        loss (torch.Tensor): LPIPS 损失值。
    """
    # 检查图像形状，并添加 batch 维度
    if pred_image.dim() == 3:
        pred_image = pred_image.unsqueeze(0)  # [1, 3, H, W]
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)      # [1, 3, H, W]

    # 确保图像在 [-1, 1] 范围内（LPIPS 要求输入为标准化图像）
    pred_image = 2. * pred_image - 1.
    gt_image = 2. * gt_image - 1.

    lpips_fn.cuda(cuda_index)
    # 计算 LPIPS 损失
    loss = lpips_fn(pred_image, gt_image)

    return loss

def rec_loss(network_output,gt,lambda_l1,lambda_ssim,lambda_lpips,cuda_index=0):
    l1=l1_loss(network_output,gt)*lambda_l1
    lssim=(1-ssim(network_output,gt))*lambda_ssim
    lpips=compute_lpips_loss(network_output,gt,cuda_index)*lambda_lpips
    return l1+lssim+lpips

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

