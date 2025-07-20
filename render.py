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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        
        view.image_height=2138
        view.image_width=1466
        
        
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + rendering.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            # gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset)
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
            
    
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        #id="277"
        id=os.path.basename(dataset.model_path)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # render oneframe_gaussianavatars
        #render_set(dataset, "oneframe", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh)
        
        
        
        # create headgap_model
        gaussians_headgap= FlameGaussianModel(dataset.sh_degree)
        gaussians_headgap.load_ply("./exp_result/"+id+"/point_cloud.ply",has_target=False)
        
        # render headgap
        render_set(dataset, "headgap", scene.loaded_iter, scene.getValCameras(), gaussians_headgap, pipeline, background, False)
        
        
        timestep=scene.getTestCameras()[0].timestep
        #==========================knn modify binding========================================
        gaussians.select_mesh_by_timestep(timestep)
        gaussians_headgap.select_mesh_by_timestep(timestep)
        points_gsa = gaussians.get_xyz
        points_headgap=gaussians_headgap.get_xyz
        headgap_binding=gaussians_headgap.binding            
        _xyz=gaussians.get_xyz
        _scaling=gaussians.get_scaling
        _rotation=gaussians.get_rotation
        
        
        distances = torch.cdist(points_gsa.cpu(), points_headgap.cpu())

        indices = distances.argmin(dim=1)

        A = torch.tensor(headgap_binding[indices.numpy()])
        diff_count = torch.sum(gaussians.binding != A.cuda())
        gaussians.binding=A.cuda()
        
        gaussians._xyz=gaussians.inverse_xyz(_xyz)
        gaussians._scaling=gaussians.inverse_scaling(_scaling)
        gaussians._rotation=gaussians.inverse_rotation(_rotation)
        #render changebinding
        #render_set(dataset, "changebinding", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh) 
        #====================================================================================
        
        part_list=["forehead","nose","eye","teeth","lip","ear","hair","boundary","neck","face","other"]
        
        gsavatar_part_index=[0,1,5,6,7,8,9,10]
        headgap_part__index=[2,3,4]
        
        mask_faces_gsavatar=gaussians.flame_model.mask.get_headgap_part(part_list).cuda()
        gsavatar_part_infor=mask_faces_gsavatar[gaussians.binding]
        headgap_part_infor=mask_faces_gsavatar[gaussians_headgap.binding]
        
        
        gsavatar_part_index = torch.tensor(gsavatar_part_index).cuda( )
        mask_gsavatar_part_infor = torch.isin(gsavatar_part_infor, gsavatar_part_index)
        indices_gsavatar_part_index = torch.nonzero(mask_gsavatar_part_infor).squeeze()
        
        
        headgap_part__index = torch.tensor(headgap_part__index).cuda( )
        mask_headgap_part__index = torch.isin(headgap_part_infor, headgap_part__index)
        indices_headgap_part_index = torch.nonzero(mask_headgap_part__index).squeeze()
        
        gaussians._xyz=torch.cat([gaussians._xyz[indices_gsavatar_part_index],gaussians_headgap._xyz[indices_headgap_part_index]],dim=0)
        gaussians._features_dc=torch.cat([gaussians._features_dc[indices_gsavatar_part_index],gaussians_headgap._features_dc[indices_headgap_part_index]],dim=0)
        #gaussians._features_rest=torch.zeros((gaussians._features_dc.shape[0],0, gaussians._features_dc.shape[2]))
        gaussians._opacity=torch.cat([gaussians._opacity[indices_gsavatar_part_index],gaussians_headgap._opacity[indices_headgap_part_index]],dim=0)
        gaussians._scaling=torch.cat([gaussians._scaling[indices_gsavatar_part_index],gaussians_headgap._scaling[indices_headgap_part_index]],dim=0)
        gaussians._rotation=torch.cat([gaussians._rotation[indices_gsavatar_part_index],gaussians_headgap._rotation[indices_headgap_part_index]],dim=0)
        gaussians.binding=torch.cat([gaussians.binding[indices_gsavatar_part_index],gaussians_headgap.binding[indices_headgap_part_index]],dim=0)
        
        render_set(dataset, "concat2gsmodel", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh)
        
        
       
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh)