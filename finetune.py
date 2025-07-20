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
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
# from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from scene.wcheadgap_test import WCHeadGAPModel
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
from scene import CameraDataset
import torch.optim as optim
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T



def save_concatenated_image(tensor_list, save_path="concatenated_image.png", save_image=False):

    processed_images = []

    for img_tensor in tensor_list:

        if img_tensor.shape[0] == 1:

            img_tensor = img_tensor.repeat(3, 1, 1)

        img_tensor = (img_tensor * 255).clamp(0, 255).byte()

        pil_image = T.ToPILImage()(img_tensor)
        processed_images.append(pil_image)

    total_width = sum(img.width for img in processed_images)
    max_height = max(img.height for img in processed_images)

    concatenated_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in processed_images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    if save_image:
        concatenated_image.save(save_path)

    return concatenated_image


def custom_collate_fn(batch):
    return batch


def load_dataset(args, dataset_list, model, shuffle=False):
    train_cameras = []
    val_cameras = []
    test_cameras = []
    for source_path in dataset_list:
        scene_info = sceneLoadTypeCallbacks["DynamicNerf"](source_path, args.white_background, args.eval,
                                                           target_path=args.target_path)
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        if scene_info.val_cameras:
            camlist.extend(scene_info.val_cameras)
        resolution_scale = 1.0
        model.load_meshes(scene_info.train_meshes, scene_info.test_meshes, scene_info.train_cameras[0].avatar_id)
        train_cameras.extend(cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args))
        val_cameras.extend(cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args))
        test_cameras.extend(cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args))

    train_dataset = CameraDataset(train_cameras)
    test_dataset = CameraDataset(test_cameras)
    val_dataset = CameraDataset(val_cameras)

    return train_dataset, val_dataset, test_dataset


def training(dataset, opt):

    root_path = dataset.data_path
    save_path = './ckpt/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open("./arguments/list.txt", 'r') as file:
        id_list = [line.strip() for line in file]

    ###step1.###   load model
    dataset_list = [root_path]

    headgap_config = EasyConfig("./arguments/headgap_config_new.yaml")

    part_list = ["forehead", "nose", "eye", "teeth", "lip", "ear", "hair", "boundary", "neck", "face", "other"]

    model = WCHeadGAPModel(headgap_config, id_list, part_list,
                           uv_gt_path=dataset.uv_gt_path).cuda()

    train_data, val_dataset, test_data = load_dataset(dataset, dataset_list, model, shuffle=False)

    state_dict = torch.load("./WCHeadGAP_pth/model.pth")
    filtered_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "cnn" not in k}
    model.load_state_dict(filtered_state_dict,strict=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ###step2.###   fintune_w----inversion
    finetune_iter = 100000
    progress_bar = tqdm(range(0, finetune_iter + 1), desc="finetune Progress")


    few_shot_cameras = train_data


    for p in model.parameters():
        p.requires_grad = False

    for p in model.Encoder.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.get_opt_param())

    camera_index = [i for i in range(0, 15)]
    for iteration in progress_bar:
        timestep = random.randint(0, 1249)
        index = random.choices(camera_index, k=2)
        index = [i + 15 * timestep for i in index]

        input_camera = []
        for i in index:
            input_camera.append(few_shot_cameras[i])

        front_camera = []
        for _ in index:
            front_camera.append(random.choice(val_dataset))

        optimizer.zero_grad()
        loss, images, gs_model_attr = model.finetune(input_camera, front_camera, background)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        gc.collect()
        progress_bar.set_postfix({"loss": loss.item()})

        if iteration % 1000 == 0 and iteration != 0:
            save_concatenated_image([images[0]], os.path.join(save_path, input_camera[
                0].image_name + f"fintune_{iteration}.png"),
                                    True)
        if iteration % 2500 == 0 and iteration != 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{iteration}.pth"))


    with torch.no_grad():
        gs_model = model.create_gsavatars_model(front_camera.avatar_id, *gs_model_attr)
        gs_model.flame_param = model.flame_param[val_dataset[0].avatar_id]
        gs_model.save_ply(os.path.join(save_path, 'point_cloud.ply'))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000,
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

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args))

    # All done
    print("\nTraining complete.")
