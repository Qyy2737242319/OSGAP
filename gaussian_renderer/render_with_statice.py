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
import math
from typing import Union
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: Union[GaussianModel, FlameGaussianModel], pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, static_gs=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    num_active_gs = pc.get_xyz.shape[0]
    if static_gs is not None:
        num_gs = num_active_gs + static_gs.get_xyz.shape[0]
    else:
        num_gs = num_active_gs

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((num_gs, 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros((num_gs, 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D_densify = screenspace_points_densify
    means2D = screenspace_points
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors_precomp = pc.get_rgb.squeeze()
    feature = pc.get_seg.squeeze()
    if feature.shape[1] == 0:
        feature = torch.zeros_like(colors_precomp)

    if static_gs is not None:
        means3D = torch.cat((means3D, static_gs.get_xyz), dim=0)
        opacity = torch.cat((opacity, static_gs.get_opacity), dim=0)
        scales = torch.cat((scales, static_gs.get_scaling), dim=0)
        rotations = torch.cat((rotations, static_gs.get_rotation), dim=0)
        colors_precomp = torch.cat((colors_precomp, static_gs.get_rgb.squeeze()), dim=0)

        feature_static_gs = static_gs.get_seg.squeeze()
        if feature_static_gs.shape[1] == 0:
            feature_static_gs = torch.zeros_like(static_gs.get_rgb.squeeze())
        feature = torch.cat((feature, feature_static_gs), dim=0)
    else:
        if feature.shape[1] == 0:
            feature = torch.zeros_like(colors_precomp)

    cov3D_precomp = None
    shs = None
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D=means3D,
        means2D=means2D,
        means2D_densify=means2D_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
        extra_attrs=feature)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "seg_fea": extra,
            "depth": rendered_depth,
            "opacity": rendered_alpha,
            "normal": rendered_norm,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii[:num_active_gs] > 0,
            "radii": radii[:num_active_gs],
            'num_active_gs': num_active_gs}
