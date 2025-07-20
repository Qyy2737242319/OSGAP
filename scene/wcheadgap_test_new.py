
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flame_model.flame import FlameHead
from utils.graphics_utils import compute_face_orientation
from utils.general_utils import Pytorch3dRasterizer, face_vertices_gen
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from scene.cameras import Camera
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from gaussian_renderer import gs_render, gs_render_raw
from utils.loss_utils import rec_loss, l1_loss
from torch.cuda.amp import autocast
from scene import FlameGaussianModel
from core.utils.network_util import set_requires_grad, trunc_normal_
import torch.nn.init as init
from torchvision.transforms import Resize
import torchvision.transforms.functional as F1
from scene.encoder import GradualStyleEncoder as psp_Encoder
from scene.model import EG3DInvEncoder
from dreifus.matrix import Intrinsics, Pose
from scene.generator import GGHGenerator as GGHStyleGAN2Backbone
from dataclasses import dataclass, field, asdict
from arguments.gaussian_attribute import GaussianAttribute, GaussianAttributeConfig
from typing import Literal, List, Union, Dict, Optional, Tuple
from elias.config import Config, implicit
from dreifus.camera import PoseType, CameraCoordinateConvention
from utils.constants import DEFAULT_INTRINSICS
from elias.util.batch import batchify_sliced

def decode_camera_params(camera_params: np.ndarray, disable_rotation_check: bool = False) -> Tuple[Pose, Intrinsics]:
    pose = Pose(camera_params[:16].reshape((4, 4)), pose_type=PoseType.CAM_2_WORLD, disable_rotation_check=disable_rotation_check)
    intrinsics = Intrinsics(camera_params[16:].reshape((3, 3)))
    return pose, intrinsics


def encode_camera_params(pose: Pose, intrinsics: Intrinsics) -> np.ndarray:
    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    return np.concatenate([pose.flatten(), intrinsics.flatten()])


@dataclass
class MappingNetworkConfig(Config):
    num_layers: int = 2  # Number of mapping layers.
    embed_features: Optional[int] = None  # Label embedding dimensionality, None = same as w_dim.
    layer_features: Optional[int] = None  # Number of intermediate features in the mapping layers, None = same as w_dim.
    activation: Literal['lrelu', 'linear', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: float = 0.01  # Learning rate multiplier for the mapping layers.
    # w_avg_beta: float = 0.998  # Decay for tracking the moving average of W during training, None = do not track.


@dataclass
class SynthesisNetworkConfig(Config):
    channel_base: int = 32768  # Overall multiplier for the number of channels.
    channel_max: int = 512  # Maximum number of channels in any layer.
    num_fp16_res: int = 0  # Use FP16 for the N highest resolutions.

    # Block Config
    architecture: Literal['orig', 'skip', 'resnet'] = 'skip'  # Architecture: 'orig', 'skip', 'resnet'.
    resample_filter: List[int] = field(
        default_factory=lambda: [1, 3, 3, 1])  # Low-pass filter to apply when resampling activations.
    conv_clamp: Optional[int] = 256  # Clamp the output of convolution layers to +-X, None = disable clamping.
    fp16_channels_last: bool = False  # Use channels-last memory format with FP16?
    fused_modconv_default: Union[
        bool, str] = True  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.

    # Layer config
    kernel_size: int = 3  # Convolution kernel size.
    use_noise: bool = True  # Enable noise input?
    activation: Literal['lrelu', 'linear', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.

    def get_block_kwargs(self) -> dict:
        block_kwargs = {k: v for k, v in asdict(self).items() if
                        k not in ['channel_base', 'channel_max', 'num_fp16_res']}
        return block_kwargs

@dataclass
class RenderingConfig(Config):
    c_gen_conditioning_zero: bool = False  # if true, fill generator pose conditioning label with dummy zero vector
    c_scale: float = 1  # Scale factor for generator pose conditioning
    box_warp: float = 1  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].


@dataclass
class SuperResolutionConfig(Config):
    use_superresolution: bool = False
    superresolution_version: int = 1
    n_channels: int = 3
    n_downsampling_layers: int = 1
    use_skip: bool = True
    cbase: int = 32768  # Capacity multiplier
    cmax: int = 512  # Max. feature maps
    fused_modconv_default: str = 'inference_only'
    sr_num_fp16_res: int = 4  # Number of fp16 layers in superresolution
    sr_antialias: bool = True
    noise_mode: Literal['random', 'none'] = 'none'


@dataclass
class GGHeadConfig(Config):
    z_dim: int = 512
    w_dim: int = 512
    c_dim: int = implicit(default=25)
    # img_resolution: int
    mapping_network_config: MappingNetworkConfig = MappingNetworkConfig()
    synthesis_network_config: SynthesisNetworkConfig = SynthesisNetworkConfig()
    rendering_config: RenderingConfig = RenderingConfig()
    super_resolution_config: SuperResolutionConfig = SuperResolutionConfig()

    uv_attributes: List[GaussianAttribute] = field(default_factory=lambda: [
        GaussianAttribute.POSITION,GaussianAttribute.SCALE,GaussianAttribute.ROTATION,
    GaussianAttribute.COLOR,GaussianAttribute.OPACITY])  # Which attributes should be predicted in UV space

    n_triplane_channels: int = 16  # number of channels for each TriPlane
    disable_position_offsets: bool = False  # If set, no position offsets will be predicted and Gaussians will always be fixed to template vertices
    use_align_corners: bool = False  # For grid_sample()
    interpolation_mode: str = 'bilinear'

    # FLAME template
    n_flame_subdivisions: int = 0  # How often the FLAME template mesh should be subdivided (increases number of predicted Gaussians)
    use_uniform_flame_vertices: bool = True  # If true, will not use predefined FLAME vertices, but instead uniformly distribute points on mesh surface using UV
    n_uniform_flame_vertices: int = 512  # How many points (squared) should be sampled in FLAME's UV space. Final number of Gaussians will be slightly smaller due to holes in UV map
    n_shells: int = 1
    shell_distance: float = 0.05
    use_learnable_template_offsets: bool = False  # If true, position of flame vertices can be adapted during training
    use_learnable_template_offset_plane: bool = False
    learnable_template_offset_plane_size: int = 64
    use_gsm_flame_template: bool = True  # Use template with back removed and more efficient UV layout
    use_flame_template_v2: bool = False
    use_sphere_template: bool = False
    use_plane_template: bool = False
    use_auxiliary_sphere: bool = False  # Predict additional set of Gaussians in front of face to models microphones, hands, other stuff that occludes the face
    auxiliary_sphere_radius: float = 0.1
    auxiliary_sphere_position: Tuple[float, float, float] = (0, -0.1, 0.4)
    uv_grid_threshold: Optional[
        float] = None  # If set, template positions with uv coordinates closer to the boundary than threshold will be dropped

    plane_resolution: int = 512  #256
    effective_plane_resolution: Optional[int] = None
    pretrained_plane_resolution: Optional[int] = implicit()
    pretrained_resolution: Optional[int] = implicit()
    # Gaussian Attribute decoding
    use_position_activation: bool = True
    use_color_activation: bool = True
    use_scale_activation: bool = True
    center_scale_activation: bool = True  # If true, the max_scale option will be properly applied inside the softplus
    use_initial_scales: bool = False
    use_rotation_activation: bool = True
    use_periodic_rotation_activation: bool = False  # If true, will use sine() activation instead of tanh()
    normalize_quaternions: bool = True
    position_attenuation: float = 1
    position_range: float = 0.25  # Maximum range that predicted positions can have. 1 means [-1, 1]
    color_attenuation: float = 1
    scale_attenuation: float = 1
    rotation_attenuation: float = 1
    scale_offset: float = -5
    additional_scale_offset: float = 0
    max_scale: float = -3
    use_softplus_scale_activation: bool = False
    no_exp_scale_activation: bool = False  # Disable 3DGS default exp() scale activation
    scale_overshoot: float = 0.001
    color_overshoot: float = 0.001  # Allows prediction of colors slightly outside of the range to prevent tanh saturation. EG3D uses 0.001
    opacity_overshoot: float = 0.001  # Avoid having to predict ridiculously large opacities to saturate sigmoid
    clamp_opacity: bool = False
    use_optimizable_gaussian_attributes: bool = False  # For debugging: Gaussians are directly learnable instead of building them from predicted UV / TriPlanes
    gaussian_attribute_config: GaussianAttributeConfig = GaussianAttributeConfig()
    use_zero_conv_position: bool = True
    use_zero_conv_scale: bool = False
    use_density_map: bool = False

    # Gaussian Attribute MLP
    mlp_layers: int = 1
    mlp_hidden_dim: int = 256

    # Gaussian Hierarchy MLP
    use_gaussian_hierarchy: bool = False
    exclude_position_from_hierarchy: bool = False  # If true, positions will be directly sampled in uv map while all other attributes will be decoded with MLP
    use_uv_position_and_hierarchy: bool = False  # If true, positions will be directly sampled in uv map in addition to decoded offset
    n_gaussians_per_texel: int = 4
    gaussian_hierarchy_feature_dim: int = 16  # number of features in uv map that will be decoded into actual Gaussian UV attributes
    use_separate_hierarchy_mlps: bool = False  # If true, use one MLP per attribute

    # Gradient Multipliers
    grad_multiplier_position: Optional[float] = None
    grad_multiplier_scale: Optional[float] = None
    grad_multiplier_rotation: Optional[float] = None
    grad_multiplier_color: Optional[float] = None
    grad_multiplier_opacity: Optional[float] = None

    # Background modeling
    use_background_plane: bool = False  # If True, will additionally generate Gaussians behind the FLAME template
    curve_background_plane: bool = True
    background_cylinder_angle: float = torch.pi  # Angle of the cylinder patch if curve_background_plane=True. Larger angle = larger background plane
    background_plane_distance: float = 0.5  # Distance of background plane to FLAME template
    background_plane_width: float = 0.5
    background_plane_height: float = 1
    n_background_gaussians: int = 64  # Number of background gaussians PER DIMENSION that will be distributed on background plane. E.g., 128 -> 128x128
    use_background_cnn: bool = False  # If True, will use 3 additional RGB channels from StyleGAN2 to models background
    use_background_upsampler: bool = False  # If use_background_cnn=True and the rendering resolution is larger than the backbone synthesis resolution
    use_separate_background_cnn: bool = False  # If True, will use an additional StyleGAN network to models background
    n_background_channels: int = 3  # Relevant if bg upsampler is used. Will be number of channels for intermediate upsampling layers
    use_masks: bool = False
    fix_alpha_blending: bool = False
    use_cnn_adaptor: bool = False

    # Maintenance
    maintenance_interval: Optional[int] = None  # How often Gaussians should be densified / pruned
    maintenance_grad_threshold: float = 0.01
    use_pruning: bool = False
    use_densification: bool = True
    use_template_update: bool = False
    template_update_attributes: List[GaussianAttribute] = field(default_factory=list)
    position_map_update_factor: float = 1  # How much of the average position map should be baked into the template at each maintenance step
    prune_opacity_threshold: float = 0.005

    use_autodecoder: bool = False  # Whether to assign one learnable latent code to each person
    use_flame_to_bfm_registration: bool = True
    load_average_offset_map: bool = False
    img_resolution: int = 512
    neural_rendering_resolution: int = 512

    n_persons: Optional[int] = implicit()
    random_background: Optional[bool] = implicit(default=False)
    return_background: Optional[bool] = implicit(default=False)
    background_color: Tuple[int, int, int] = implicit(
        default=(255, 255,
                 255))  # Background color to use during training. Should match the background color used in the dataset

    @staticmethod
    def from_eg3d_config(z_dim,  # Input latent (Z) dimensionality.
                         c_dim,  # Conditioning label (C) dimensionality.
                         w_dim,  # Intermediate latent (W) dimensionality.
                         img_resolution,  # Output resolution.
                         img_channels,  # Number of output color channels.
                         sr_num_fp16_res=0,
                         mapping_kwargs={},  # Arguments for MappingNetwork.
                         rendering_kwargs={},
                         sr_kwargs={},
                         **synthesis_kwargs,  # Arguments for SynthesisNetwork
                         ) -> 'GGHeadConfig':
        config = GGHeadConfig(z_dim, w_dim,
                                                 mapping_network_config=MappingNetworkConfig(**mapping_kwargs),
                                                 synthesis_network_config=SynthesisNetworkConfig(**synthesis_kwargs),
                                                 rendering_config=RenderingConfig(**rendering_kwargs),
                                                 use_flame_to_bfm_registration=True,
                                                 img_resolution=img_resolution)
        config.c_dim = c_dim
        return config

    @staticmethod
    def default() -> 'GGHeadConfig':
        config = GGHeadConfig(512, 512,
                                                 mapping_network_config=MappingNetworkConfig(),
                                                 synthesis_network_config=SynthesisNetworkConfig(),
                                                 rendering_config=RenderingConfig(),
                                                 use_flame_to_bfm_registration=True)
        config.c_dim = 25
        return config

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_c, hidden_layers):
        super(CNN, self).__init__()

        # 初始化通道数设置，逐步递增
        # channels = input_channels
        layers = []

        self.input_layer = nn.Conv2d(input_channels, hidden_c, kernel_size=3, stride=1, padding=1)

        # 添加隐藏层
        for i in range(hidden_layers - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.InstanceNorm2d(hidden_c))

        # 创建隐藏层的 Sequential 模块
        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Conv2d(hidden_c, output_channels, kernel_size=3, stride=1, padding=1)

        # 激活函数，保证输出在 [0, 1] 范围内
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=4, dropout_rate=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers - 1)]
        )

        self.output_linear = nn.Linear(hidden_dim, output_dim)

        # self._initialize_weights()

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size * N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.leaky_relu(h)

        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)


class MLP1(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class WCHeadGAPModel(nn.Module):
    def __init__(self, config, avatar_id_list, part_id_list, cuda_index=0, n_shape=300, n_expr=100, uv_gt_path=None):
        super(WCHeadGAPModel, self).__init__()

        self.cuda_index = cuda_index
        self.flame_model = FlameHead(
            n_shape,
            n_expr,
            add_teeth=True,
        ).cuda(self.cuda_index)

        # some info
        self.config = config

        self.uv_size = config.uv_size
        self.avatars_list = avatar_id_list
        self.avatars_num = len(avatar_id_list)

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # initial sth.
        self.initial_binding_infor(config.uv_size, 1)
        self.initial_part_info(part_id_list)
        self.cano_vertices_xyz = {}

        self.uv_gt = None
        if uv_gt_path is not None:
            self.uv_gt = np.load(uv_gt_path)



        ##################################
        # self.backbone_config = GGHeadConfig.default()
        # _uv_attribute_names = [attribute_name for attribute_name in self.backbone_config.uv_attributes
        #                             if (
        #                                     attribute_name != GaussianAttribute.POSITION or not self.backbone_config.disable_position_offsets)]
        #
        # _n_uv_channels = sum(
        #     [gaussian_attribute.get_n_channels(self.backbone_config.gaussian_attribute_config) for gaussian_attribute in
        #      _uv_attribute_names])
        #
        # n_backbone_channels = _n_uv_channels
        # if self.backbone_config.use_background_cnn:
        #     n_backbone_channels += self.backbone_config.n_background_channels
        #
        # self.backbone = GGHStyleGAN2Backbone(self.backbone_config.z_dim, self.backbone_config.c_dim, self.backbone_config.w_dim,
        #                                      img_resolution=self.backbone_config.plane_resolution,
        #                                      pretrained_plane_resolution=self.backbone_config.pretrained_plane_resolution,
        #                                      img_channels=n_backbone_channels,
        #                                      mapping_kwargs=asdict(self.backbone_config.mapping_network_config),
        #                                      **asdict(self.backbone_config.synthesis_network_config))
        ################################

        # initial latent
        # self.latent_z = nn.Parameter(torch.randn(self.gs_num, config.latent_z_dim))
        #
        #self.latent_f = nn.Parameter(torch.randn(self.avatars_num, len(part_id_list), config.latent_f_dim))

        #self.w = torch.nn.Parameter(torch.zeros(self.latent_f.shape[0], 11, 1).cuda(self.cuda_index))

        #self.Encoder = psp_Encoder(50, 'ir_se', config)
        self.Encoder = EG3DInvEncoder(in_channels=5, encoder_name="resnet34", encoder_depth=3, mode="Normal",
                                      use_bn=False)
        # trunc_normal_(self.latent_z, std=.02)
        # trunc_normal_(self.latent_f, std=.02)

        self.flame_param = nn.ParameterDict()

        #self.mlp1s = nn.ModuleList()
        #self.mlp2s = nn.ModuleList()

        # for part_id in part_id_list:
        #     self.mlp1s.append(
        #         MLP(config.latent_f_dim + config.latent_z_dim, 11, config.mlp1_hidden_dim, config.mlp1_hidden_layers))
        #     self.mlp2s.append(
        #         MLP(config.latent_f_dim + config.latent_z_dim + 3 + 11, config.latent_h_dim, config.mlp2_hidden_dim,
        #             config.mlp2_hidden_layers))

        #self.cnn = CNN(config.latent_h_dim, 3, config.cnn_hidden_channels, config.cnn_hidden_layers)



    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.backbone_config.rendering_config.c_gen_conditioning_zero:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.backbone_config.rendering_config.c_scale, truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff,
                                     update_emas=update_emas)

    def predict_planes(self, ws: torch.Tensor, update_emas=False, cache_backbone=False, use_cached_backbone=False,
                       alpha_plane_resolution: Optional[float] = None, **synthesis_kwargs):
        # Predict 2D planes
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        return planes

    def generate_planes(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False,
                  return_raw_attributes: bool = False, return_uv_map: bool = False,
                  alpha_plane_resolution: Optional[float] = None,
                  return_masks: bool = False,
                  sh_ref_cam: Optional[Pose] = None,
                  triplane_offsets: Optional[torch.Tensor] = None,
                  **synthesis_kwargs) :

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        if triplane_offsets is not None:
            planes = triplane_offsets
        else:
            planes = self.predict_planes(ws, update_emas=update_emas, cache_backbone=cache_backbone,
                                         use_cached_backbone=use_cached_backbone,
                                         alpha_plane_resolution=alpha_plane_resolution, **synthesis_kwargs)
        return planes

    def get_opt_param(self, lr=0.001):
        # indices = torch.where(self.gs_part_infor == 9)[0].tolist()

        param_groups = [
            {'params': self.Encoder.parameters(), 'lr': 0.00001},  # 为 mlp1s 设置学习率 0.00001
        ]

        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):

                    # 原文是添加arap loss同时对static_offset优化，暂时没有这个loss，为保持稳定采取gsavatars的方法，以下三个参数不优化
                    if key in {"shape", "static_offset", "dynamic_offset"}:
                        # if key in {"dynamic_offset"}:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, dict):
                    collect_params(value, lr)

        collect_params(self.flame_param, lr=1e-5)

        return param_groups

    def get_opt_param_without_flame(self, lr=0.001):
        # indices = torch.where(self.gs_part_infor == 9)[0].tolist()

        param_groups = [
            {'params': self.Encoder.parameters(), 'lr': 0.0001},  # 为 mlp1s 设置学习率 0.00001
        ]

        return param_groups

    def initial_part_info(self, part_id_list):
        mask_faces = self.flame_model.mask.get_headgap_part(part_id_list).to(self.binding.device)
        self.gs_part_infor = mask_faces[self.binding]

    def set_cano_vertices_xyz(self, flame_param, name):
        with torch.no_grad():
            verts, verts_cano = self.flame_model(
                flame_param['shape'].clone()[None, ...],
                torch.zeros_like(flame_param['expr'][[0]]),
                torch.zeros_like(flame_param['rotation'][[0]]),
                torch.zeros_like(flame_param['neck_pose'][[0]]),
                torch.zeros_like(flame_param['jaw_pose'][[0]]),
                torch.zeros_like(flame_param['eyes_pose'][[0]]),
                torch.zeros_like(flame_param['translation'][[0]]),
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'].clone(),
            )
            batch_size = 1
            uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
            face_vertices_shape = face_vertices_gen(verts, self.flame_model.faces.expand(batch_size, -1, -1))
            verts_uvs = self.flame_model.verts_uvs
            verts_uvs = verts_uvs * 2 - 1
            verts_uvs[..., 1] = - verts_uvs[..., 1]
            verts_uvs = verts_uvs[None]
            verts_uvs = torch.cat([verts_uvs, verts_uvs[:, :, 0:1] * 0. + 1.], -1)
            rast_out, pix_to_face, bary_coords = uv_rasterizer(verts_uvs.expand(batch_size, -1, -1),
                                                               self.flame_model.textures_idx.expand(batch_size, -1, -1),
                                                               face_vertices_shape)
            uvmask = rast_out[:, -1].unsqueeze(1)
            uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1)  # batch=1
            uvmask_flaten_idx = (uvmask_flaten[:] > 0)

            uv_vertices_shape = rast_out[:, :3]
            uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1,
                                                                                                         0)  # batch=1
            uv_vertices_shape = uv_vertices_shape_flaten[uvmask_flaten_idx].unsqueeze(0)

            assert uv_vertices_shape[0].size(0) == self.gs_num, f"There is a problem with sampling"
            self.cano_vertices_xyz[name] = uv_vertices_shape[0]

    def initial_binding_infor(self, uv_size, batch_size):
        uv_rasterizer = Pytorch3dRasterizer(uv_size)
        face_vertices_shape = face_vertices_gen(self.flame_model.v_template.expand(batch_size, -1, -1),
                                                self.flame_model.faces.expand(batch_size, -1, -1))
        verts_uvs = self.flame_model.verts_uvs
        verts_uvs = verts_uvs * 2 - 1
        verts_uvs[..., 1] = - verts_uvs[..., 1]
        verts_uvs = verts_uvs[None]
        verts_uvs = torch.cat([verts_uvs, verts_uvs[:, :, 0:1] * 0. + 1.], -1)
        rast_out, pix_to_face, bary_coords = uv_rasterizer(verts_uvs.expand(batch_size, -1, -1),
                                                           self.flame_model.textures_idx.expand(batch_size, -1, -1),
                                                           face_vertices_shape)
        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1)  # batch=1
        uvmask_flaten_idx = (uvmask_flaten[:] > 0)

        pix_to_face_flaten = pix_to_face[0].clone().view(-1)  # batch=1
        self.binding = pix_to_face_flaten[uvmask_flaten_idx]  # pix to face idx

        # self.pix_to_v_idx = self.flame_model.faces.expand(batch_size, -1, -1)[0, self.binding, :] # pix to vert idx

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1,
                                                                                                     0)  # batch=1
        uv_vertices_shape = uv_vertices_shape_flaten[uvmask_flaten_idx].unsqueeze(0)

        self.gs_num = uv_vertices_shape.shape[1]

    def update_mesh_by_param_dict(self, flame_param):
        verts, verts_cano = self.flame_model(
            torch.from_numpy(flame_param['shape']).cuda(self.cuda_index),
            torch.from_numpy(flame_param['expr']).cuda(self.cuda_index),
            torch.from_numpy(flame_param['rotation']).cuda(self.cuda_index),
            torch.from_numpy(flame_param['neck_pose']).cuda(self.cuda_index),
            torch.from_numpy(flame_param['jaw_pose']).cuda(self.cuda_index),
            torch.from_numpy(flame_param['eyes_pose']).cuda(self.cuda_index),
            torch.from_numpy(flame_param['translation']).cuda(self.cuda_index),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=torch.from_numpy(flame_param['static_offset']).cuda(self.cuda_index),
        )
        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_by_param_dict_torch(self, flame_param):
        verts, verts_cano = self.flame_model(
            flame_param['shape'],
            flame_param['expr'],
            flame_param['rotation'],
            flame_param['neck_pose'],
            flame_param['jaw_pose'],
            flame_param['eyes_pose'],
            flame_param['translation'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
        )
        self.update_mesh_properties(verts, verts_cano)

    def local2global_xyz(self, local_xyz, indices):
        xyz = torch.matmul(self.face_orien_mat[:, self.binding[indices], :, :],  local_xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[:, self.binding[indices], :] + self.face_center[:, self.binding[indices], :]

    def local2global_rotation(self, local_rotation, indices):
        rot = self.rotation_activation(local_rotation, dim=-1)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[:, self.binding[indices], :], dim=-1)
        # rot=local_rotation
        # face_orien_quat=self.face_orien_quat[:, self.binding[indices], :]
        return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))

    def local2global_scale(self, local_scale, indices):
        local_scale = self.scaling_activation(local_scale)
        return local_scale * self.face_scaling[:, self.binding[indices], :]  # +0.001

    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts, faces, return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano

    def merge_flame_params(self, camera_list):
        # 获取第一个 Camera 对象的 flame_param 结构，用于初始化 merged_dict
        sample_flame_param = self.flame_param[camera_list[0].avatar_id]
        merged_dict = {key: [] for key in sample_flame_param.keys()}
        list_avatar_ids = []
        cano_vertices_xyz = []
        # 遍历所有 Camera 对象，收集每个 flame_param 的值
        for camera in camera_list:
            cano_vertices_xyz.append(self.cano_vertices_xyz[camera.avatar_id])
            list_avatar_ids.append(
                self.avatars_list.index(camera.avatar_id) if camera.avatar_id in self.avatars_list else -1)
            flame_param = self.flame_param[camera.avatar_id]
            for key, value in flame_param.items():
                if torch.is_tensor(value):
                    if key == "dynamic_offset":
                        continue
                    if value.dim() == 1:  # shape
                        merged_dict[key].append(value)
                    else:
                        if value.shape[0] == 1:  # stasistic_offset
                            merged_dict[key].append(value.squeeze(0))
                        else:
                            merged_dict[key].append(value[camera.timestep])

        merged_dict = {
            key: torch.stack(value_list, axis=0) if len(value_list) > 0 else None
            for key, value_list in merged_dict.items()
        }

        return merged_dict, list_avatar_ids, torch.stack(cano_vertices_xyz, dim=0)

    def fintune_rgb_setup(self):
        temp = self.gs_features[0, :, 0:3].clone()
        self.gsrgb = torch.nn.Parameter(temp)
        self.optimizer = torch.optim.Adam([self.gsrgb], lr=0.001) #0.001

    def fintune_w_setup(self):

        for p in self.Encoder.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.Adam([
            {'params': self.Encoder.parameters(), 'lr': 0.0002}
        ])

    def fintune_w_setup_new(self):

        for p in self.Encoder.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.Adam([
            {'params': self.Encoder.parameters(), 'lr': 0.0002}
        ])

    def fintune_network_setup(self):

        # for p in self.Encoder.parameters():
        #     p.requires_grad = False

        self.optimizer = torch.optim.Adam([
            {'params': self.Encoder.parameters(), 'lr': 0.0005},
            {'params': self.mlp1s.parameters(), 'lr': 0.0005},  # 为 mlp1s 设置学习率 0.00001
            {'params': self.mlp2s.parameters(), 'lr': 0.0005},  # 为 mlp2s 设置学习率 0.00001
            {'params': self.cnn.parameters(), 'lr': 0.0005},  # 为 cnn 设置学习率 0.00001
            {'params':self.backbone.parameters(),'lr':0.0005},
        ])
        # 加入flame参数

    def train_setup(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.latent_z, 'lr': 0.001},  # 为 latent_z 设置学习率 0.001
            {'params': self.latent_f, 'lr': 0.00001},  # 为 latent_f 设置学习率 0.00001
            {'params': self.mlp1s.parameters(), 'lr': 0.00001},  # 为 mlp1s 设置学习率 0.00001
            {'params': self.mlp2s.parameters(), 'lr': 0.00001},  # 为 mlp2s 设置学习率 0.00001
            {'params': self.cnn.parameters(), 'lr': 0.00001},  # 为 cnn 设置学习率 0.00001
            {'params':self.backbone.parameters(),'lr':0.001}
        ])
        # 加入flame参数

    def fintune_network(self, cameras, background: torch.tensor, usecnn=False):

        gt_rgb = []
        gt_opecity = []
        for index, camera in enumerate(cameras):
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)

        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params([cameras[0]])
        # update flame


        #convet to 512,512
        ###-------------------------------------
        patch_image = torch.ones((gt_rgb.shape[0], gt_rgb.shape[1],
                                  gt_rgb.shape[2], (gt_rgb.shape[2] - gt_rgb.shape[3]) // 2)).cuda()

        input_image = torch.cat([patch_image, gt_rgb, patch_image], dim=3)
        ###-------------------------------------

        input_image = F1.resize(input_image, [512, 512])
        latent = self.Encoder(input_image)

        latent = latent.contiguous()

        local_position_uv = latent[:, :9, ...]
        rot_delta_uv = latent[:, 9:17, ...]
        local_scaling_uv = latent[:, 17:26, ...]
        gs_opacity_uv = latent[:, 26:28, ...]
        gs_features_uv = latent[:, 28:, ...]

        local_position_uv = torch.mean(local_position_uv.reshape(-1, 3, 3, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        rot_delta_uv = torch.mean(rot_delta_uv.reshape(-1, 4, 2, 256, 256), dim=2).reshape(-1, 4, 256, 256)
        local_scaling_uv = torch.mean(local_scaling_uv.reshape(-1, 3, 3, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        gs_opacity_uv = torch.mean(gs_opacity_uv.reshape(-1, 1, 2, 256, 256), dim=2).reshape(-1, 1, 256, 256)
        gs_features_uv = torch.mean(gs_features_uv.reshape(-1, 34, 2, 256, 256), dim=2).reshape(-1, 1, 256, 256)
        uv = torch.cat([local_position_uv, rot_delta_uv, local_scaling_uv, gs_opacity_uv, gs_features_uv], dim=1)

        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []
        gs_attr = []

        xs = torch.linspace(-1, 1, steps=self.config.n_uniform_flame_vertices, device="cuda")
        ys = torch.linspace(-1, 1, steps=self.config.n_uniform_flame_vertices, device="cuda")

        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        sampled_uv_coords = torch.stack([ys, xs], dim=-1)
        sampled_uv_coords = torch.flatten(sampled_uv_coords, start_dim=0, end_dim=1)

        for part_index in range(11):
            gs_indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(gs_indices)

            # uv_i = gs_indices // 300
            # uv_j = gs_indices % 300
            # sampled_indices = uv_i * 512 + uv_j
            sampled_indices = gs_indices
            valid_uv_coords = sampled_uv_coords[sampled_indices]
            valid_uv_coords = valid_uv_coords.unsqueeze(0).unsqueeze(2).contiguous()

            #gs_attr = torch.zeros((input_image.shape[0],gs_indices.shape[0],45))

            attr = torch.nn.functional.grid_sample(uv, valid_uv_coords,
                                                       align_corners=False,
                                                       mode="bilinear")  # [B*S, C_uv, G, 1]


            gs_attr.append(attr.squeeze(3).permute(0, 2, 1))

            #gs_attr = mlp1(input)

            local_position = gs_attr[part_index][..., :3]
            rot_delta_0 = gs_attr[part_index][..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[part_index][..., 7:10]
            gs_opacity = gs_attr[part_index][..., 10].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, gs_indices)
            global_xyzs.append(global_position)
            h = gs_attr[part_index][..., 11:]
            h[..., :3] = torch.sigmoid(h[..., :3])
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)

        all_indices = torch.cat(all_indices)
        local_scalings = torch.cat(local_scalings, dim=1)
        local_xyz = torch.cat(local_xyz, dim=1)
        local_rots = torch.cat(local_rots, dim=1)
        gs_op = torch.cat(gs_alpha, dim=1)
        global_alpha = self.opacity_activation(gs_op)
        global_features = torch.cat(gs_features, dim=1)
        global_xyz = torch.cat(global_xyzs, dim=1)
        global_rotation = self.local2global_rotation(local_rots, all_indices)
        global_scale = self.local2global_scale(local_scalings, all_indices)

        losses = []
        I_fea = []
        I_rgb = []
        I_opecity = []
        v_filter = []
        gt_ref = []
        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[0], global_rotation[0], global_scale[0], global_alpha[0],
                                         global_features[0], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_ref.append(camera.ref_image.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        v_filter = torch.stack(v_filter, dim=0)
        gt_ref = torch.stack(gt_ref, dim=0)
        if usecnn:
            I = self.cnn(I_fea)
            losses.append(rec_loss(I, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                                   self.cuda_index).mean())
            losses.append(rec_loss(I, gt_ref, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                                   self.cuda_index).sum() * self.config.lambda_ref)
        else:
            losses.append(
                rec_loss(I_rgb, gt_ref, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                         self.cuda_index).sum() * self.config.lambda_ref)

        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.cuda_index).mean())
        losses.append(l1_loss(I_opecity, gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings) - self.config.threshold_scale).norm(
            dim=-1).mean() * self.config.lambda_s)

        if usecnn:
            images = I
        else:
            images = I_rgb
        return sum(losses), images, [local_xyz[0].clone(), global_features[0, :, :3].clone(), gs_op[0].clone(),
                                     local_scalings[0].clone(), local_rots[0].clone(),
                                     self.binding[all_indices].clone()]

    def fintune_w(self, cameras,front_camera, background: torch.tensor, usecnn=False):

        gt_rgb = []
        gt_opecity = []
        for index, camera in enumerate(front_camera):
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)

        pre_gt_rgb = []
        pre_gt_opecity = []
        for index, camera in enumerate(cameras):
            pre_gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            pre_gt_opecity.append(camera.mask.cuda(self.cuda_index))
        pre_gt_rgb = torch.stack(pre_gt_rgb, dim=0)
        pre_gt_opecity = torch.stack(pre_gt_opecity, dim=0)

        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        # update flame

        patch_image = torch.ones((gt_rgb.shape[0],gt_rgb.shape[1],
                                  gt_rgb.shape[2],(gt_rgb.shape[2]-gt_rgb.shape[3])//2)).cuda()

        input_image = torch.cat([patch_image,gt_rgb,patch_image],dim=3)
        input_image = F1.resize(input_image, [512, 512])

        latent = self.Encoder(input_image)

        latent = latent.contiguous()

        local_position_uv = latent[:, :9, ...]
        rot_delta_uv = latent[:, 9:17, ...]
        local_scaling_uv = latent[:, 17:26, ...]
        gs_opacity_uv = latent[:, 26:28, ...]
        gs_features_uv = latent[:, 28:, ...]

        local_position_uv = torch.mean(local_position_uv.reshape(-1, 3, 3, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        rot_delta_uv = torch.mean(rot_delta_uv.reshape(-1, 4, 2, 256, 256), dim=2).reshape(-1, 4, 256, 256)
        local_scaling_uv = torch.mean(local_scaling_uv.reshape(-1, 3, 3, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        gs_opacity_uv = torch.mean(gs_opacity_uv.reshape(-1, 1, 2, 256, 256), dim=2).reshape(-1, 1, 256, 256)
        gs_features_uv = torch.mean(gs_features_uv.reshape(-1, 34, 2, 256, 256), dim=2).reshape(-1, 34, 256, 256)
        uv = torch.cat([local_position_uv, rot_delta_uv, local_scaling_uv, gs_opacity_uv, gs_features_uv], dim=1)

        uv_gt = None
        if self.uv_gt is not None:
            gs_features_uv_gt = torch.tensor(self.uv_gt['gs_features_uv'],dtype=torch.float32).cuda()
            gs_opacity_uv_gt = torch.tensor(self.uv_gt['gs_opacity_uv'],dtype=torch.float32).cuda()
            rot_delta_uv_gt = torch.tensor(self.uv_gt['rot_delta_uv'],dtype=torch.float32).cuda()
            local_scaling_uv_gt = torch.tensor(self.uv_gt['local_scaling_uv'],dtype=torch.float32).cuda()
            local_position_uv_gt = torch.tensor(self.uv_gt['local_position_uv'],dtype=torch.float32).cuda()

            uv_gt = torch.cat([local_position_uv_gt,rot_delta_uv_gt,local_scaling_uv_gt,gs_opacity_uv_gt,gs_features_uv_gt], dim=2).unsqueeze(0).permute(0, 3, 1, 2)
            uv_gt = uv_gt.repeat(latent.shape[0],1,1,1)

        #uv_gt = F1.resize(uv_gt, [256, 256])

        # local_position_uv = latent[:, :27, ...]
        # rot_delta_uv = latent[:, 27:63, ...]
        # local_scaling_uv = latent[:, 63:90, ...]
        # gs_opacity_uv = latent[:, 90:, ...]
        # gs_features_uv = latent[:, :, ...]
        # pad = torch.zeros([latent.shape[0],6,latent.shape[2],latent.shape[3]],device="cuda",dtype=torch.float32)
        # gs_features_uv = torch.cat([gs_features_uv,pad],dim=1)
        #
        # local_position_uv = torch.mean(local_position_uv.reshape(-1, 3, 9, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        # rot_delta_uv = torch.mean(rot_delta_uv.reshape(-1, 4, 9, 256, 256), dim=2).reshape(-1, 4, 256, 256)
        # local_scaling_uv = torch.mean(local_scaling_uv.reshape(-1, 3, 9, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        # gs_opacity_uv = torch.mean(gs_opacity_uv.reshape(-1, 1, 6, 256, 256), dim=2).reshape(-1, 1, 256, 256)
        #
        # gs_features_uv = torch.mean(gs_features_uv.reshape(-1, 34, 3, 256, 256), dim=2).reshape(-1, 34, 256, 256)

        #uv = torch.cat([local_position_uv, rot_delta_uv, local_scaling_uv, gs_opacity_uv, gs_features_uv], dim=1)

        self.update_mesh_by_param_dict_torch(flame_parea)

        all_indices = []

        xs = torch.linspace(-1, 1, steps=self.config.n_uniform_flame_vertices, device="cuda")
        ys = torch.linspace(-1, 1, steps=self.config.n_uniform_flame_vertices, device="cuda")

        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        sampled_uv_coords = torch.stack([ys, xs], dim=-1)
        sampled_uv_coords = torch.flatten(sampled_uv_coords, start_dim=0, end_dim=1)

        for part_index in range(11):
            gs_indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(gs_indices)
        all_indices = torch.cat(all_indices)

        sampled_indices = all_indices
        valid_uv_coords = sampled_uv_coords[sampled_indices]
        valid_uv_coords = valid_uv_coords.unsqueeze(0).unsqueeze(2).contiguous()
        valid_uv_coords = valid_uv_coords.repeat(len(cameras), 1, 1, 1)

        attr = torch.nn.functional.grid_sample(uv, valid_uv_coords,
                                                   align_corners=False,
                                                   mode="bilinear")  # [B*S, C_uv, G, 1]

        attr = attr.squeeze(3).permute(0,2,1)


        local_xyz = attr[..., :3]
        rot_delta_0 = attr[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        local_rots = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        local_scalings = attr[..., 7:10]
        gs_op = attr[..., 10].unsqueeze(-1)

        h = attr[..., 11:]
        h[..., :3] = torch.sigmoid(h[..., :3])
        global_features = h
        global_alpha = self.opacity_activation(gs_op)

        global_xyz = self.local2global_xyz(local_xyz, all_indices)
        global_rotation = self.local2global_rotation(local_rots, all_indices)
        global_scale = self.local2global_scale(local_scalings, all_indices)

        losses = []
        I_fea = []
        I_rgb = []
        I_opecity = []
        v_filter = []

        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[0], global_rotation[0], global_scale[0], global_alpha[0],
                                         global_features[0], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            v_filter.append(rendering["visibility_filter"])

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)


        if usecnn:
            I = self.cnn(I_fea)
            losses.append(rec_loss(I, pre_gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                                   self.cuda_index).mean())
        losses.append(rec_loss(I_rgb, pre_gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.cuda_index).mean())
        if uv_gt is not None:
            losses.append(l1_loss(uv[:,:14,...], uv_gt).mean() * self.config.lambda_alpha)
        losses.append(l1_loss(I_opecity, pre_gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings) - self.config.threshold_scale).norm(
            dim=-1).mean() * self.config.lambda_s)

        if usecnn:
            images = I
        else:
            images = I_rgb


        return sum(losses), images,[local_xyz[0].clone(), global_features[0, :, :3].clone(), gs_op[0].clone(),
                                     local_scalings[0].clone(), local_rots[0].clone(),
                                     self.binding[all_indices].clone()]

    def create_gsavatars_model(self, avatars_id, xyz, rgb, opacity, scaling, rotation, binding):
        gaussians = FlameGaussianModel(0, False)
        gaussians.flame_param = self.flame_param[avatars_id]
        gaussians._xyz = xyz
        gaussians._features_dc = rgb.unsqueeze(1)
        gaussians._features_rest = torch.zeros((rgb.shape[0], 0, rgb.shape[1]))
        gaussians._opacity = opacity
        gaussians._scaling = scaling
        gaussians._rotation = rotation
        gaussians.binding = binding

        return gaussians

    def forward(self, cameras: Camera, background: torch.tensor, usecnn=True):
        gt_rgb = []
        gt_opecity = []
        for index, camera in enumerate(cameras):
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)

        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        # update flame

        patch_image = torch.ones((gt_rgb.shape[0], gt_rgb.shape[1],
                                  gt_rgb.shape[2], (gt_rgb.shape[2] - gt_rgb.shape[3]) // 2)).cuda()

        input_image = torch.cat([patch_image, gt_rgb, patch_image], dim=3)
        input_image = F1.resize(input_image, [512, 512])

        latent = self.Encoder(input_image)

        latent = latent.contiguous()

        local_position_uv = latent[:, :9, ...]
        rot_delta_uv = latent[:, 9:17, ...]
        local_scaling_uv = latent[:, 17:26, ...]
        gs_opacity_uv = latent[:, 26:28, ...]
        gs_features_uv = latent[:, 28:, ...]

        local_position_uv = torch.mean(local_position_uv.reshape(-1, 3, 3, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        rot_delta_uv = torch.mean(rot_delta_uv.reshape(-1, 4, 2, 256, 256), dim=2).reshape(-1, 4, 256, 256)
        local_scaling_uv = torch.mean(local_scaling_uv.reshape(-1, 3, 3, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        gs_opacity_uv = torch.mean(gs_opacity_uv.reshape(-1, 1, 2, 256, 256), dim=2).reshape(-1, 1, 256, 256)
        gs_features_uv = torch.mean(gs_features_uv.reshape(-1, 34, 2, 256, 256), dim=2).reshape(-1, 34, 256, 256)
        uv = torch.cat([local_position_uv, rot_delta_uv, local_scaling_uv, gs_opacity_uv, gs_features_uv], dim=1)

        # local_position_uv = latent[:, :27, ...]
        # rot_delta_uv = latent[:, 27:63, ...]
        # local_scaling_uv = latent[:, 63:90, ...]
        # gs_opacity_uv = latent[:, 90:, ...]
        # gs_features_uv = latent[:, :, ...]
        # pad = torch.zeros([latent.shape[0],6,latent.shape[2],latent.shape[3]],device="cuda",dtype=torch.float32)
        # gs_features_uv = torch.cat([gs_features_uv,pad],dim=1)
        #
        # local_position_uv = torch.mean(local_position_uv.reshape(-1, 3, 9, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        # rot_delta_uv = torch.mean(rot_delta_uv.reshape(-1, 4, 9, 256, 256), dim=2).reshape(-1, 4, 256, 256)
        # local_scaling_uv = torch.mean(local_scaling_uv.reshape(-1, 3, 9, 256, 256), dim=2).reshape(-1, 3, 256, 256)
        # gs_opacity_uv = torch.mean(gs_opacity_uv.reshape(-1, 1, 6, 256, 256), dim=2).reshape(-1, 1, 256, 256)
        #
        # gs_features_uv = torch.mean(gs_features_uv.reshape(-1, 34, 3, 256, 256), dim=2).reshape(-1, 34, 256, 256)

        # uv = torch.cat([local_position_uv, rot_delta_uv, local_scaling_uv, gs_opacity_uv, gs_features_uv], dim=1)

        self.update_mesh_by_param_dict_torch(flame_parea)

        all_indices = []

        xs = torch.linspace(-1, 1, steps=self.config.n_uniform_flame_vertices, device="cuda")
        ys = torch.linspace(-1, 1, steps=self.config.n_uniform_flame_vertices, device="cuda")

        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        sampled_uv_coords = torch.stack([ys, xs], dim=-1)
        sampled_uv_coords = torch.flatten(sampled_uv_coords, start_dim=0, end_dim=1)

        for part_index in range(11):
            gs_indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(gs_indices)
        all_indices = torch.cat(all_indices)

        sampled_indices = all_indices
        valid_uv_coords = sampled_uv_coords[sampled_indices]
        valid_uv_coords = valid_uv_coords.unsqueeze(0).unsqueeze(2).contiguous()

        attr = torch.nn.functional.grid_sample(uv, valid_uv_coords,
                                               align_corners=False,
                                               mode="bilinear")  # [B*S, C_uv, G, 1]

        attr = attr.squeeze(3).permute(0, 2, 1)

        local_xyz = attr[..., :3]
        rot_delta_0 = attr[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        local_rots = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        local_scalings = attr[..., 7:10]
        gs_op = attr[..., 10].unsqueeze(-1)

        h = attr[..., 11:]
        h[..., :3] = torch.sigmoid(h[..., :3])
        global_features = h
        global_alpha = self.opacity_activation(gs_op)

        global_xyz = self.local2global_xyz(local_xyz, all_indices)
        global_rotation = self.local2global_rotation(local_rots, all_indices)
        global_scale = self.local2global_scale(local_scalings, all_indices)

        losses = []
        I_fea = []
        I_rgb = []
        I_opecity = []
        v_filter = []

        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[0], global_rotation[0], global_scale[0], global_alpha[0],
                                         global_features[0], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            v_filter.append(rendering["visibility_filter"])

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)


        if usecnn:
            I = self.cnn(I_fea)
            losses.append(rec_loss(I, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                                   self.cuda_index).mean())
        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.cuda_index).mean())
        losses.append(l1_loss(I_opecity, gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings) - self.config.threshold_scale).norm(
            dim=-1).mean() * self.config.lambda_s)

        if usecnn:
            images = [I_rgb[0], I[0], gt_rgb[0], I_opecity[0], gt_opecity[0]]
        else:
            images = [I_rgb[0], gt_rgb[0], I_opecity[0], gt_opecity[0]]
        return sum(losses), images, [local_xyz[0].clone(), global_features[0, :, :3].clone(), gs_op[0].clone(),
                                     local_scalings[0].clone(), local_rots[0].clone(),
                                     self.binding[all_indices].clone()]

    def forward_all_view(self, cameras: Camera, background: torch.tensor):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params([cameras[0]])
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
        # gs_xyz=[]
        # gs_rotation=[]
        # gs_scale=[]
        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []

        # latent_f_B=[self.latent_f[key].unsqueeze(0) for key in list_avatar_ids if key in self.latent_f]
        # latent_f_B=torch.cat(latent_f_B, dim=0)

        latent_f_B = self.latent_f[list_avatar_ids]

        for part_index in range(11):
            # prepare latent by part id
            indices = torch.where(self.gs_part_infor == part_index)[0]

            all_indices.append(indices)
            # latent_z =torch.nn.functional.normalize(self.latent_z[indices],dim=-1)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]
            B, d = latent_f.shape
            N, D = latent_z.shape

            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)

            # Step 3: 在最后一个维度上拼接
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)
            # gs_attr = torch.tanh(gs_attr)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = self.scaling_activation(gs_attr[..., 7:10])
            gs_opacity = self.opacity_activation(gs_attr[..., -1].unsqueeze(-1))

            global_xyz = self.local2global_xyz(local_position, indices)
            e = global_xyz - cano_vertices_xyz[:, indices, :]

            input2 = torch.cat((input, e, local_position, local_rot, local_scaling, gs_opacity), dim=2)
            h = mlp2(input2)
            # h = F.relu(h)
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)

        all_indices = torch.cat(all_indices)
        local_scalings = torch.cat(local_scalings, dim=1)
        local_xyz = torch.cat(local_xyz, dim=1)
        local_rots = torch.cat(local_rots, dim=1)
        gs_alpha = torch.cat(gs_alpha, dim=1)
        gs_features = torch.cat(gs_features, dim=1)

        gs_xyz = self.local2global_xyz(local_xyz, all_indices)
        gs_rotation = self.local2global_rotation(local_rots, all_indices)
        gs_scale = self.local2global_scale(local_scalings, all_indices)

        losses = []
        I_fea = []
        I_rgb = []
        gt_rgb = []
        I_opecity = []
        gt_opecity = []
        v_filter = []
        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, gs_xyz[0], gs_rotation[0], gs_scale[0], gs_alpha[0], gs_features[0],
                                         background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        # I_rgb=I_fea[:,:3,:,:]
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter = torch.stack(v_filter, dim=0)
        I = self.cnn(I_fea)

        losses.append(rec_loss(I, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.cuda_index).mean())
        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.cuda_index).mean())
        losses.append(l1_loss(I_opecity, gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(local_scalings - self.config.threshold_scale).norm(dim=-1).mean() * self.config.lambda_s)

        images = [I_rgb[0], I[0], gt_rgb[0], I_opecity[0], gt_opecity[0]]
        # images=[I_rgb[0], torch.zeros_like(I_rgb[0]) , gt_rgb[0] ,I_opecity[0],gt_opecity[0]]
        return sum(losses), images, gs_xyz[0].clone().squeeze(0).cpu()

    def inference_get_gaussian_attr(self, cameras: Camera):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
        gs_xyz = []
        gs_rotation = []
        gs_scale = []
        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_scale = []
        local_rotation = []
        gs_indices = []

        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0).repeat(len(cameras), 1, 1)

        for part_index in range(11):
            # prepare latent by part id
            indices = torch.where(self.gs_part_infor == part_index)[0]
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]
            B, d = latent_f.shape
            N, D = latent_z.shape

            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)

            # Step 3: 在最后一个维度上拼接
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            global_xyz = self.local2global_xyz(gs_attr[..., :3], indices)
            e = global_xyz - cano_vertices_xyz[:, indices, :]

            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)

            local_xyz.append(gs_attr[..., :3])
            local_scale.append(gs_attr[..., 7:10])
            gs_indices.append(indices)
            gs_xyz.append(global_xyz)
            local_rotation.append(gs_attr[..., 3:7])
            gs_rotation.append(self.local2global_rotation(gs_attr[..., 3:7], indices))
            gs_scale.append(self.local2global_scale(gs_attr[..., 7:10], indices))
            gs_alpha.append(self.opacity_activation(gs_attr[..., -1].unsqueeze(-1)))
            gs_features.append(h)

        self.gs_alpha = torch.cat(gs_alpha, dim=1)
        self.gs_features = torch.cat(gs_features, dim=1)
        self.local_xyz = torch.cat(local_xyz, dim=1)
        self.local_scale = torch.cat(local_scale, dim=1)
        self.local_rotation = torch.cat(gs_rotation, dim=1)
        self.gs_indices = torch.cat(gs_indices, dim=0)

        return self.local2global_xyz(self.local_xyz, self.gs_indices), self.binding[self.gs_indices], self.gs_alpha

    def load_meshes(self, train_meshes, test_meshes, avatarid):
        meshes = {**train_meshes, **test_meshes}
        pose_meshes = meshes

        self.num_timesteps = max(pose_meshes) + 1  # required by viewers
        num_verts = self.flame_model.v_template.shape[0]

        # if not self.disable_flame_static_offset:
        static_offset = torch.from_numpy(meshes[0]['static_offset'])
        if static_offset.shape[0] != num_verts:
            static_offset = torch.nn.functional.pad(static_offset,
                                                    (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
        # else:
        #     static_offset = torch.zeros([num_verts, 3])

        T = self.num_timesteps

        flame_param = {
            'shape': torch.from_numpy(meshes[0]['shape']),
            'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
            'rotation': torch.zeros([T, 3]),
            'neck_pose': torch.zeros([T, 3]),
            'jaw_pose': torch.zeros([T, 3]),
            'eyes_pose': torch.zeros([T, 6]),
            'translation': torch.zeros([T, 3]),
            'static_offset': static_offset,
            'dynamic_offset': torch.zeros([T, num_verts, 3]),
        }

        for i, mesh in pose_meshes.items():
            flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
            flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
            flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
            flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
            flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
            flame_param['translation'][i] = torch.from_numpy(mesh['translation'])

        for k, v in flame_param.items():
            flame_param[k] = nn.Parameter(v.float().cuda(self.cuda_index))

        self.flame_param[avatarid] = flame_param
        self.set_cano_vertices_xyz(flame_param, avatarid)

    def inference(self, cameras: Camera, background: torch.tensor):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
        gs_xyz = []
        gs_rotation = []
        gs_scale = []
        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_scale = []

        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0).repeat(len(cameras), 1, 1)

        # for part_index in range(11):
        #     #prepare latent by part id
        #     indices = torch.where(self.gs_part_infor == part_index)[0]
        #     latent_z =self.latent_z[indices]
        #     latent_f = latent_f_B[:,part_index,:]
        #     B,d=latent_f.shape
        #     N,D=latent_z.shape

        #     latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
        #     latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)

        #     # Step 3: 在最后一个维度上拼接
        #     input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

        #     mlp1=self.mlp1s[part_index]
        #     mlp2=self.mlp2s[part_index]

        #     gs_attr=mlp1(input)

        #     global_xyz=self.local2global_xyz(gs_attr[..., :3],indices)
        #     e=global_xyz-cano_vertices_xyz[:,indices,:]

        #     input2=torch.cat((input,e,gs_attr), dim=2)
        #     h=mlp2(input2)

        #     local_xyz.append(gs_attr[..., :3])
        #     local_scale.append(gs_attr[..., 7:10])
        #     gs_xyz.append(global_xyz)
        #     gs_rotation.append(self.local2global_rotation(gs_attr[..., 3:7],indices))
        #     gs_scale.append(self.local2global_scale(gs_attr[..., 7:10],indices))
        #     gs_alpha.append(self.opacity_activation(gs_attr[..., -1].unsqueeze(-1)))
        #     gs_features.append(h)

        gs_xyz = self.local2global_xyz(self.local_xyz, self.gs_indices)
        gs_rotation = self.local2global_rotation(self.local_rotation, self.gs_indices)
        gs_scale = self.local2global_scale(self.local_scale, self.gs_indices)
        gs_alpha = self.gs_alpha
        gs_features = self.gs_features

        I_fea = []
        I_rgb = []
        gt_rgb = []
        I_opecity = []
        gt_opecity = []

        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, gs_xyz[index], gs_rotation[index], gs_scale[index], gs_alpha[index],
                                         gs_features[index], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)

        I = self.cnn(I_fea)

        images = [I_rgb[0], I[0], gt_rgb[0], I_opecity[0], gt_opecity[0]]
        return images

    def finetune_rgb(self, cameras: Camera, background: torch.tensor, gtimage):

        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_xyz = self.local2global_xyz(self.local_xyz, self.gs_indices)
        gs_rotation = self.local2global_rotation(self.local_rotation, self.gs_indices)
        gs_scale = self.local2global_scale(self.local_scale, self.gs_indices)
        gs_alpha = self.gs_alpha
        gs_features = self.gs_features
        gs_rgb = self.gsrgb

        I_fea = []
        I_rgb = []
        gt_rgb = []
        I_opecity = []
        gt_opecity = []

        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, gs_xyz[index], gs_rotation[index], gs_scale[index], gs_alpha[index],
                                         gs_features[index], background, gs_rgb)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(gtimage.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)

        # I=self.cnn(I_fea)

        losses = []
        # losses.append(rec_loss(I,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean())
        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.cuda_index).mean())
        # losses.append(l1_loss(I_opecity,gt_opecity).mean()*self.config.lambda_alpha)
        # losses.append(F.relu(local_xyz.norm(dim=2) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        # losses.append(F.relu(torch.exp(local_scale) - self.config.threshold_scale).norm(dim=2).mean() * self.config.lambda_s)

        images = [I_rgb[0], None, gt_rgb[0], I_opecity[0], gt_opecity[0]]
        return sum(losses), images, gs_xyz[0].clone().squeeze(0).cpu()

