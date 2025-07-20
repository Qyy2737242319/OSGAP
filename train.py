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
import gc
import os

os.environ["MASTER_ADDR"] = "127.0.0.1"  # 可以改为主节点 IP 地址
os.environ["MASTER_PORT"] = "29500"  # 选择一个未占用的端口
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from random import randint
from gaussian_renderer import render, network_gui
# from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from scene.wcheadgap_latest import WCHeadGAPModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import random
from utils.config_util import EasyConfig
from utils.camera_utils import cameraList_from_camInfos
from scene.dataset_readers import sceneLoadTypeCallbacks
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from scene import CameraDataset
import torch.optim as optim
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
import copy

from torch.utils.data import Dataset


def save_concatenated_image(tensor_list, save_path="concatenated_image.png"):
    # 创建一个列表用于存储处理后的图像
    processed_images = []

    for img_tensor in tensor_list:
        # 检查通道数，如果是 1 通道（灰度图）
        if img_tensor.shape[0] == 1:
            # 将灰度图扩展为 3 通道
            img_tensor = img_tensor.repeat(3, 1, 1)

        # 将张量值从 (0, 1) 范围转换到 (0, 255)
        img_tensor = (img_tensor * 255).clamp(0, 255).byte()

        # 使用 `torchvision` 的 `ToPILImage` 转换为 PIL 图像
        pil_image = T.ToPILImage()(img_tensor)
        processed_images.append(pil_image)

    # 拼接所有图像（横向拼接）
    total_width = sum(img.width for img in processed_images)
    max_height = max(img.height for img in processed_images)

    # 创建一个空白的图像用于拼接
    concatenated_image = Image.new("RGB", (total_width, max_height))

    # 开始拼接
    x_offset = 0
    for img in processed_images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存拼接后的图像
    concatenated_image.save(save_path)


def custom_collate_fn(batch):
    return batch  # 不做任何拼接，直接返回列表


def load_dataset(args, dataset_list, model, shuffle=False):
    train_cameras = []
    val_cameras = []
    test_cameras = []
    for source_path in dataset_list:
        scene_info = sceneLoadTypeCallbacks["DynamicNerf"](source_path, args.white_background, args.eval,
                                                           target_path=args.target_path)
        resolution_scale = 1.0
        model.load_meshes(scene_info.train_meshes, scene_info.test_meshes, scene_info.train_cameras[0].avatar_id)
        train_cameras.extend(cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args))
        val_cameras.extend(cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args))
        test_cameras.extend(cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args))

    dataset_obj = CameraDataset(train_cameras)
    sampler = DistributedSampler(dataset_obj, shuffle=True)
    train_data_loader = DataLoader(dataset_obj, batch_size=2, num_workers=0, pin_memory=False, sampler=sampler,
                                   collate_fn=custom_collate_fn)
    # train_data_loader=DataLoader(CameraDataset(train_cameras), batch_size=4, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True,collate_fn=custom_collate_fn)
    # val_data_loader=DataLoader(CameraDataset(val_cameras), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    # test_data_loader=DataLoader(CameraDataset(test_cameras), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_data_loader, None, None


def training(rank, dataset, opt, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device_index = torch.cuda.current_device()
    root_path = dataset.data_path
    save_path = './ckpt/'
    if rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    with open("./arguments/list.txt", 'r') as file:
        id_list = [line.strip() for line in file]

    #id_list = id_list[:10]

    #target_ids = ["074", "175", "210"]

    #id_list = [id for id in id_list if id not in target_ids]
    # file_path = os.path.join(save_path, "idlist.txt")
    #
    # if rank == 0:
    #     with open(file_path, "w") as file:
    #         for line in id_list:
    #             file.write(line + "\n")

    dataset_list = [os.path.join(root_path, id) for id in id_list]
    ###step1.###   headgap dataloader
    headgap_config = EasyConfig("./arguments/headgap_config.yaml")
    part_list = ["forehead", "nose", "eye", "teeth", "lip", "ear", "hair", "boundary", "neck", "face", "other"]
    model = WCHeadGAPModel(headgap_config, id_list, part_list, device_index).cuda(device_index)

    model = DDP(model, device_ids=[rank])
    model_para_dict_temp = torch.load('./ckpt/model.pth')
    model_para_dict = {}
    for key_i in model_para_dict_temp.keys():
        if key_i!="module.w":
            model_para_dict[key_i] = model_para_dict_temp[key_i]  # 删除掉前7个字符'module.'
    del model_para_dict_temp
    model.load_state_dict(model_para_dict)

    train_data_loader, _, _ = load_dataset(dataset, dataset_list, model.module, shuffle=False)
    iter_camera_train = iter(train_data_loader)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=model.device)

    for p in model.module.parameters():
        p.requires_grad = True

    model.module.train_network_setup()

    if rank == 0:
        progress_bar = tqdm(range(0, 500000 + 1), desc="Training Generator Progress")

    for iteration in range(0, 500000 + 1):

        # for iteration in progress_bar:
        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(train_data_loader)
            viewpoint_cam = next(iter_camera_train)

        model.module.optimizer.zero_grad()
        loss, images, attr0 = model.module.forward(viewpoint_cam, background, False)
        loss.backward()
        model.module.optimizer.step()
        #scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()
        if rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
        if rank == 0 and (iteration % 2000 == 0 or iteration == opt.iterations):
            save_concatenated_image(images, os.path.join(save_path, str(iteration) + ".png"))
            # torch.save(model.state_dict(), os.path.join(save_path,"model.pth"))
            # with open(os.path.join(save_path,str(iteration)+".xyz"), 'w') as f:
            #     for point in points.detach().numpy():
            # #         f.write(f"{point[0]} {point[1]} {point[2]}\n")
            with torch.no_grad():
                gs_model = model.module.create_gsavatars_model(viewpoint_cam[0].avatar_id, *attr0)
                gs_model.save_ply(os.path.join(save_path, "point_cloud.ply"))
                del gs_model

        if rank == 0 and (iteration % 2000 == 0 or iteration == opt.iterations):
            # save_concatenated_image(images, os.path.join(save_path,str(iteration)+".png"))
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{iteration}.pth"))

    ########################################################
    dist.destroy_process_group()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=30_000,
                        help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))

    # print("Optimizing " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # # Start GUI server, configure and run training
    # #network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)

    world_size = torch.cuda.device_count()
    mp.spawn(training, args=(lp.extract(args), op.extract(args), world_size), nprocs=world_size)

    # All done
    print("\nTraining complete.")
