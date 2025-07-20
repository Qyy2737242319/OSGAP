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
from gaussian_renderer import gs_render
from utils.loss_utils import rec_loss, l1_loss, chamfer_distance
from torch.cuda.amp import autocast
from scene import FlameGaussianModel
from core.utils.network_util import set_requires_grad, trunc_normal_
import torch.nn.init as init
import lpips
from utils.sh_utils import RGB2SH
from scene.mini_gaussian_model import MiniGaussianModel
from scene.encoder import GradualStyleEncoder as psp_Encoder
import torchvision.transforms.functional as F1

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_c, hidden_layers):
        super(CNN, self).__init__()

        # 初始化通道数设置，逐步递增
        # channels = input_channels
        layers = []

        self.input_layer = nn.Conv2d(input_channels, hidden_c, kernel_size=3, stride=1, padding=1)

        # 添加隐藏层
        for i in range(hidden_layers - 1):
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            layers.append(nn.BatchNorm2d(hidden_c))
            layers.append(nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=1, padding=1))

        # 创建隐藏层的 Sequential 模块
        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Conv2d(hidden_c, output_channels, kernel_size=3, stride=1, padding=1)

        # 激活函数，保证输出在 [0, 1] 范围内
        self.activation = nn.Sigmoid()

        # 初始化权重

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     # 对所有子模块进行初始化
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

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


class HeadGAPModel(nn.Module):
    def __init__(self, config, avatar_id_list, part_id_list, cuda_index=0, n_shape=300, n_expr=100):
        super(HeadGAPModel, self).__init__()

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
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda(self.cuda_index).eval()

        # initial latent
        self.latent_z = nn.Parameter(torch.randn(self.gs_num, config.latent_z_dim))
        self.latent_f = nn.Parameter(torch.randn(self.avatars_num, len(part_id_list), config.latent_f_dim))
        self.w = torch.nn.Parameter(torch.zeros(self.latent_f.shape[0], 11, 1).cuda(self.cuda_index))

        self.Encoder = psp_Encoder(50, 'ir_se', config)
        # trunc_normal_(self.latent_z, std=.02)
        # trunc_normal_(self.latent_f, std=.02)

        self.flame_param = nn.ParameterDict()

        self.mlp1s = nn.ModuleList()
        self.mlp2s = nn.ModuleList()

        for part_id in part_id_list:
            self.mlp1s.append(
                MLP(config.latent_f_dim + config.latent_z_dim, 11, config.mlp1_hidden_dim, config.mlp1_hidden_layers))
            self.mlp2s.append(
                MLP(config.latent_f_dim + config.latent_z_dim + 3 + 11, config.latent_h_dim, config.mlp2_hidden_dim,
                    config.mlp2_hidden_layers))

        self.cnn = CNN(config.latent_h_dim, 3, config.cnn_hidden_channels, config.cnn_hidden_layers)

    def get_opt_param(self, lr=0.001):
        param_groups = [
            {"params": [self.latent_z, self.latent_f], "lr": lr},
            {"params": [p for mlp in self.mlp1s for p in mlp.parameters()], "lr": lr},
            {"params": [p for mlp in self.mlp2s for p in mlp.parameters()], "lr": lr},
            {"params": list(self.cnn.parameters()), "lr": lr},
        ]

        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):
                    # 原文是添加arap loss同时对static_offset优化，暂时没有这个loss，为保持稳定采取gsavatars的方法，以下三个参数不优化
                    if key in {"shape", "static_offset", "dynamic_offset"}:
                        # if key in {"dynamic_offset"}:
                        value.requires_grad = False
                    else:
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, nn.ParameterDict):
                    collect_params(value, lr)

        # collect_params(self.flame_param, lr=1e-5)

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

        # xyz_data = uv_vertices_shape[0].cpu().detach().numpy()

        # # 保存为 .xyz 文件
        # with open("output.xyz", "w") as file:
        #     for point in xyz_data:
        #         # 写入每行的 x, y, z 坐标
        #         file.write(f"{point[0]} {point[1]} {point[2]}\n")

        # self.cano_vertices_xyz = uv_vertices_shape[0] # for cano init
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

    def global2local_xyz_one_batch(self, xyz_global, binding):
        xyz_local = (xyz_global - self.face_center.clone().detach()[0, binding,
                                  :]) / self.face_scaling.clone().detach()[0, binding, :]
        xyz_local = torch.bmm(self.face_orien_mat.clone().detach()[0, binding, :, :].transpose(1, 2),
                              xyz_local[..., None]).squeeze(-1)
        return xyz_local

    def global2local_scale_one_batch(self, global_scaling, binding):
        local_scaling = global_scaling / self.face_scaling.clone().detach()[0, binding, :]
        return torch.log(local_scaling)

    def local2global_xyz(self, local_xyz, indices):
        xyz = torch.matmul(self.face_orien_mat[:, self.binding[indices], :, :], local_xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[:, self.binding[indices], :] + self.face_center[:, self.binding[indices], :]

    def local2global_rotation(self, local_rotation, indices):
        rot = self.rotation_activation(local_rotation, dim=-1)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[:, self.binding[indices], :], dim=-1)
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
        self.optimizer = torch.optim.Adam([self.gsrgb], lr=0.001)

    def fintune_w_setup(self):

        for param in self.parameters():
            param.requires_grad = False
        self.w.requires_grad = True
        self.optimizer = torch.optim.Adam([self.w], lr=0.001)

    def fintune_network_setup(self):
        self.w.requires_grad = False
        self.latent_f.requires_grad = True
        self.latent_z.requires_grad = True
        for i in range(11):
            if i == 3 or i == 4:
                self.mlp1s[i].requires_grad = False
                self.mlp2s[i].requires_grad = False
            else:
                self.mlp1s[i].requires_grad = True
                self.mlp2s[i].requires_grad = True

        param_groups = [
            {'params': self.latent_z, 'lr': 0.001},
            {'params': self.latent_f, 'lr': 0.00001},
            {'params': self.mlp1s.parameters(), 'lr': 0.00001},
            {'params': self.mlp2s.parameters(), 'lr': 0.00001},
            {'params': self.cnn.parameters(), 'lr': 0.00001}
        ]

        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):
                    if key in {"dynamic_offset", "static_offset"}:
                        value.requires_grad = False
                    elif key in {"translation"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-6})  # 1e-6
                    elif key in {"expr"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-5})
                    elif key in {"shape"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-5})  # 1e-5
                    else:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, nn.ParameterDict):
                    collect_params(value, lr)

        collect_params(self.flame_param, lr=1e-5)  # 1e-5
        self.optimizer = torch.optim.Adam(param_groups)

    def fintune_network_setup_custom(self):
        self.w.requires_grad = False
        self.latent_f.requires_grad = True
        self.latent_z.requires_grad = True
        for i in range(11):
            if i == 3 or i == 4:
                self.mlp1s[i].requires_grad = False
                self.mlp2s[i].requires_grad = False
            else:
                self.mlp1s[i].requires_grad = True
                self.mlp2s[i].requires_grad = True

        param_groups = [
            {'params': self.latent_z, 'lr': 0.001},
            {'params': self.latent_f, 'lr': 0.00001},
            {'params': self.mlp1s.parameters(), 'lr': 0.00001},
            {'params': self.mlp2s.parameters(), 'lr': 0.00001},
            {'params': self.cnn.parameters(), 'lr': 0.00001}
        ]

        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):
                    if key in {"dynamic_offset", "static_offset"}:
                        value.requires_grad = False
                    elif key in {"translation"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-6})  # 1e-6
                    elif key in {"expr"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-2})
                    elif key in {"shape"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-3})  # 1e-5
                    else:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, nn.ParameterDict):
                    collect_params(value, lr)

        collect_params(self.flame_param, lr=1e-5)  # 1e-5
        self.optimizer = torch.optim.Adam(param_groups)

    def fintune_network_setup_custom_flame(self):
        param_groups = []

        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):
                    if key in {"dynamic_offset", "static_offset"}:
                        value.requires_grad = False
                    elif key in {"translation"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-6})  # 1e-6
                    elif key in {"expr"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-3})
                    elif key in {"shape"}:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": 1e-4})  # 1e-5
                    else:
                        value.requires_grad = True
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, nn.ParameterDict):
                    collect_params(value, lr)

        collect_params(self.flame_param, lr=1e-5)  # 1e-5
        self.optimizer = torch.optim.Adam(param_groups)

    def fintune_network_icp_setup(self):
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.latent_z, 'lr': 0.001},  # 为 latent_z 设置学习率 0.001
        #     {'params': self.latent_f, 'lr': 0.00001},  # 为 latent_f 设置学习率 0.00001
        #     {'params': self.mlp1s.parameters(), 'lr': 0.00001},  # 为 mlp1s 设置学习率 0.00001
        #     {'params': self.mlp2s.parameters(), 'lr': 0.00001},  # 为 mlp2s 设置学习率 0.00001
        #     {'params': self.cnn.parameters(), 'lr': 0.00001}  # 为 cnn 设置学习率 0.00001
        # ])

        param_groups = [
            {'params': self.latent_z, 'lr': 0.001},  # 为 latent_z 设置学习率 0.001
            {'params': self.latent_f, 'lr': 0.00001},  # 为 latent_f 设置学习率 0.00001
            {'params': self.mlp1s.parameters(), 'lr': 0.00001},  # 为 mlp1s 设置学习率 0.00001
            {'params': self.mlp2s.parameters(), 'lr': 0.00001},  # 为 mlp2s 设置学习率 0.00001
            {'params': self.cnn.parameters(), 'lr': 0.00001}  # 为 cnn 设置学习率 0.00001
        ]

        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):
                    # if key in {"shape", "static_offset", "dynamic_offset"}:
                    if key in {"dynamic_offset"}:
                        value.requires_grad = False
                    else:
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, dict):
                    collect_params(value, lr)

        collect_params(self.flame_param, lr=1e-3)
        self.optimizer = torch.optim.Adam(param_groups)

    def fintune_network_icp(self, cameras, background: torch.tensor, usecnn=False, tar_xyz=None, tar_label=None):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params([cameras[0]])
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []
        flame_part = []

        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)

        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            flame_part.append(torch.zeros_like(indices) + part_index)
            all_indices.append(indices)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]

            # if part_index == 3 or part_index == 4:
            #     latent_f.detach()

            B, d = latent_f.shape
            N, D = latent_z.shape

            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[..., 7:10]
            gs_opacity = gs_attr[..., -1].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, indices)
            global_xyzs.append(global_position)
            e = global_position - cano_vertices_xyz[:, indices, :]
            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)
            h[..., :3] = torch.sigmoid(h[..., :3])
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)

        flame_part = torch.cat(flame_part)
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
        gt_rgb = []
        I_opecity = []
        gt_opecity = []
        v_filter = []
        # gt_ref=[]
        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[0], global_rotation[0], global_scale[0], global_alpha[0],
                                         global_features[0], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            # gt_ref.append(camera.ref_image.cuda(self.cuda_index))
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])

        # ===================================================
        def save_tensor_as_xyz(tensor, file_path):
            """
            将 PyTorch Tensor 点云保存为 .xyz 文件。
            :param tensor: torch.Tensor, 点云数据，形状为 (N, 3)，每行是 (x, y, z) 点坐标。
            :param file_path: str, 保存的 .xyz 文件路径。
            """
            # 确保输入是二维 Tensor，且每行有三个坐标
            if tensor.dim() != 2 or tensor.size(1) != 3:
                raise ValueError("点云 Tensor 必须是 (N, 3) 的形状")

            # 转为 NumPy 数组
            points = tensor.clone().cpu().detach().numpy()

            # 写入 .xyz 文件
            with open(file_path, 'w') as f:
                for point in points:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
            print(f"点云已保存到: {file_path}")

        # ===================================================
        # part_list=["forehead","nose","eye","teeth","lip","ear","hair","boundary","neck","face","other"]

        face_index_headgap = self.flame_model.mask.get_fid_except_region(["eyeballs", "teeth", "hair", "boundary"])
        mask = torch.isin(self.binding[all_indices], face_index_headgap)  # 得到一个布尔张量
        indices_flame = torch.nonzero(mask, as_tuple=True)[0]

        # 使用索引获取 `src_pcd`
        src_pcd = global_xyz[0, indices_flame, :]

        # 2. 优化处理 `tar_label`
        values_gs = [0]  # 条件值
        mask_gs = torch.zeros_like(tar_label, dtype=torch.bool)
        for value in values_gs:
            mask_gs |= (tar_label == value)
        indices_gs = torch.nonzero(mask_gs, as_tuple=True)[0]

        # 使用索引获取 `tar_pcd`
        tar_pcd = tar_xyz[indices_gs]

        # save_tensor_as_xyz(src_pcd, "xyz_headgap.xyz")
        # save_tensor_as_xyz(tar_pcd, "xyz_gsavatars.xyz")

        # distances = torch.cdist(src_pcd, tar_pcd)
        distances = torch.cdist(tar_pcd, src_pcd)
        closest_idx = torch.argmin(distances, dim=1)
        matched_src = src_pcd[closest_idx]

        # 局部对齐 Loss（点到点欧几里得距离）
        point_loss = torch.mean((tar_pcd - matched_src) ** 2)

        # 全局对齐 Loss（倒角距离）
        global_loss = chamfer_distance(src_pcd, tar_pcd)
        losses.append((0.2 * point_loss + (1 - 0.8) * global_loss) * 10)

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter = torch.stack(v_filter, dim=0)
        # gt_ref = torch.stack(gt_ref, dim=0)
        if usecnn:
            I = self.cnn(I_fea)
        #     losses.append(rec_loss(I,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean())
        #     losses.append(rec_loss(I,gt_ref,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).sum()* self.config.lambda_ref)
        # else:
        #     losses.append(rec_loss(I_rgb,gt_ref,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).sum()* self.config.lambda_ref)

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

    def inference_icp(self, cameras, background: torch.tensor, usecnn=False, tar_xyz=None, tar_label=None):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params([cameras[0]])
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []
        flame_part = []

        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)

        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            flame_part.append(torch.zeros_like(indices) + part_index)
            all_indices.append(indices)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]

            # if part_index == 3 or part_index == 4:
            #     latent_f.detach()

            B, d = latent_f.shape
            N, D = latent_z.shape

            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[..., 7:10]
            gs_opacity = gs_attr[..., -1].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, indices)
            global_xyzs.append(global_position)
            e = global_position - cano_vertices_xyz[:, indices, :]
            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)
            h[..., :3] = torch.sigmoid(h[..., :3])
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)

        flame_part = torch.cat(flame_part)
        all_indices = torch.cat(all_indices)
        local_scalings = torch.cat(local_scalings, dim=1)
        local_xyz = torch.cat(local_xyz, dim=1)
        local_rots = torch.cat(local_rots, dim=1)
        gs_op = torch.cat(gs_alpha, dim=1)
        # global_alpha=self.opacity_activation(gs_op)
        global_features = torch.cat(gs_features, dim=1)
        global_xyz = torch.cat(global_xyzs, dim=1)

        # global_rotation=self.local2global_rotation(local_rots,all_indices)
        # global_scale=self.local2global_scale(local_scalings,all_indices)

        # ===================================================
        def save_tensor_as_xyz(tensor, file_path):
            """
            将 PyTorch Tensor 点云保存为 .xyz 文件。
            :param tensor: torch.Tensor, 点云数据，形状为 (N, 3)，每行是 (x, y, z) 点坐标。
            :param file_path: str, 保存的 .xyz 文件路径。
            """
            # 确保输入是二维 Tensor，且每行有三个坐标
            if tensor.dim() != 2 or tensor.size(1) != 3:
                raise ValueError("点云 Tensor 必须是 (N, 3) 的形状")

            # 转为 NumPy 数组
            points = tensor.clone().cpu().detach().numpy()

            # 写入 .xyz 文件
            with open(file_path, 'w') as f:
                for point in points:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
            print(f"点云已保存到: {file_path}")

        # ===================================================

        tar_binding = torch.zeros(tar_xyz.shape[0]).long().cuda() - 1

        face_index_headgap = self.flame_model.mask.get_fid_except_region(["eyeballs", "teeth", "hair", "boundary"])
        mask = torch.isin(self.binding[all_indices], face_index_headgap)  # 得到一个布尔张量
        indices_flame = torch.nonzero(mask, as_tuple=True)[0]
        src_pcd = global_xyz[0, indices_flame, :]

        temp_binding = self.binding[all_indices][indices_flame]
        value_gs = 0  # 假设只有一个值
        mask_gs = (tar_label == value_gs)  # 直接创建布尔掩码
        indices_gs = torch.nonzero(mask_gs, as_tuple=True)[0]  # 获取索引
        tar_pcd = tar_xyz[indices_gs]  # 根据索引获取对应的点云数据

        # save_tensor_as_xyz(tar_pcd, "xyz_gsavatars.xyz")

        distances = torch.cdist(tar_pcd, src_pcd)  # (N, M)
        closest_idx = torch.argmin(distances, dim=1)  # (N,)
        tar_binding[indices_gs] = temp_binding[closest_idx]
        # =================================================================
        value_gs_905 = 3  # 假设只有一个值
        mask_gs = (tar_label == value_gs_905)  # 直接创建布尔掩码
        indices_gs_905 = torch.nonzero(mask_gs, as_tuple=True)[0]  # 获取索引
        tar_binding[indices_gs_905] = 905
        # =================================================================
        # =================================================================
        value_gs_3000 = 4  # 假设只有一个值
        mask_gs = (tar_label == value_gs_3000)  # 直接创建布尔掩码
        indices_gs_3000 = torch.nonzero(mask_gs, as_tuple=True)[0]  # 获取索引
        tar_binding[indices_gs_3000] = 3000
        # =================================================================

        return [local_xyz[0].clone(), global_features[0, :, :3].clone(), gs_op[0].clone(), local_scalings[0].clone(),
                local_rots[0].clone(), self.binding[all_indices].clone()], tar_binding

    def fintune_network(self, cameras, background: torch.tensor, usecnn=False, finetune_mouth=False):

        gt_rgb = []
        gt_opecity = []
        for index, camera in enumerate(cameras):
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)

        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params([cameras[0]])
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)

        patch_image = torch.ones((gt_rgb.shape[0], gt_rgb.shape[1],
                                  gt_rgb.shape[2], (gt_rgb.shape[2] - gt_rgb.shape[3]) // 2)).cuda()

        input_image = torch.cat([patch_image, gt_rgb, patch_image], dim=3)
        input_image = F1.resize(input_image, [256, 256])
        latent = self.Encoder(input_image)
        w = torch.mean(latent, dim=1)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []

        softmax_weights = F.softmax(self.w, dim=0)
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)

        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(indices)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]

            if part_index == 3:  # teeth
                latent_z = latent_z.clone().detach()
                latent_f = latent_f.clone().detach()

            if part_index == 4:
                if not finetune_mouth:
                    latent_z = latent_z.clone().detach()
                latent_f = latent_f.clone().detach()

            B, d = latent_f.shape
            N, D = latent_z.shape
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[..., 7:10]
            gs_opacity = gs_attr[..., -1].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, indices)
            global_xyzs.append(global_position)
            e = global_position - cano_vertices_xyz[:, indices, :]
            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)
            h = torch.sigmoid(h)
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
        gt_rgb = []
        mouth_masks = []
        I_opecity = []
        gt_opecity = []
        I_depth = []
        v_filter = []
        I_normal = []
        gt_rgb_raw = []
        fea_raw = []
        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[index], global_rotation[index], global_scale[index],
                                         global_alpha[index], global_features[index], background)
            I_rgb.append(rendering["render"])
            I_depth.append(rendering["depth"])
            I_normal.append(rendering["normal"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(torch.cat((rendering["render"], rendering["feature"]), dim=0))
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            if hasattr(camera, 'original_camera'):
                with torch.no_grad():
                    temp = gs_render.render(camera.original_camera, global_xyz[index], global_rotation[index],
                                            global_scale[index], global_alpha[index], global_features[index],
                                            background)
                    fea_raw.append(torch.cat((temp["render"], temp["feature"]), dim=0))
                    gt_rgb_raw.append(camera.original_camera.original_image.cuda(self.cuda_index))

            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            if hasattr(camera, 'one_hot_label'):
                mouth_masks.append(camera.one_hot_label[2].cuda(self.cuda_index))
            else:
                mouth_masks.append(torch.zeros_like(rendering["opacity"])[0])

            v_filter.append(rendering["visibility_filter"])
        if len(gt_rgb_raw) != 0:
            gt_rgb_raw = torch.stack(gt_rgb_raw, dim=0)
            fea_raw = torch.stack(fea_raw, dim=0)

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter = torch.stack(v_filter, dim=0)
        gt_mask_mouth = torch.stack(mouth_masks, dim=0).unsqueeze(1)
        if usecnn:
            if len(gt_rgb_raw) == 0:
                I = self.cnn(I_fea)
                losses.append(
                    rec_loss(I, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                             self.lpips_fn))
            else:
                I = self.cnn(fea_raw)
                losses.append(
                    rec_loss(I, gt_rgb_raw, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                             self.lpips_fn))
                #     I=self.cnn(I_fea)
            #     _, _, h2, w2 = gt_rgb_raw_t.size()
            #     # 使用 interpolate 将 a 的大小调整为与 b 相同
            #     I = F.interpolate(I, size=(h2, w2), mode='bilinear', align_corners=False)
            #     losses.append(rec_loss(I,gt_rgb_raw_t,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.lpips_fn))
            # losses.append(rec_loss(I*gt_mask_mouth,gt_rgb*gt_mask_mouth,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.lpips_fn)* self.config.lambda_m)
        else:
            pass
            # losses.append(rec_loss(I_rgb*gt_mask_mouth,gt_rgb*gt_mask_mouth,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.lpips_fn)* self.config.lambda_m)

        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.lpips_fn))
        losses.append(l1_loss(I_opecity, gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(
            F.relu(local_xyz[v_filter].norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings[v_filter]) - self.config.threshold_scale).norm(
            dim=-1).mean() * self.config.lambda_s)
        if usecnn:
            images = I[0]
        else:
            images = I_rgb[0]

        minigs = MiniGaussianModel(global_xyz[0], global_features[0, :, :3], global_alpha[0], global_scale[0],
                                   global_rotation[0], self.binding[all_indices])
        return sum(losses), [images, I_depth[0], I_normal[0]], [local_xyz[0].clone(), global_features[0, :, :].clone(),
                                                                gs_op[0].clone(), local_scalings[0].clone(),
                                                                local_rots[0].clone(),
                                                                self.binding[all_indices].clone()], minigs

    def inference(self, cameras, background: torch.tensor, usecnn=False):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params([cameras[0]])
        # update flame
        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []

        softmax_weights = F.softmax(self.w, dim=0)
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)

        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(indices)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]

            if part_index == 3 or part_index == 4:
                latent_z = latent_z.clone().detach()
                latent_f = latent_f.clone().detach()

            B, d = latent_f.shape
            N, D = latent_z.shape
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[..., 7:10]
            gs_opacity = gs_attr[..., -1].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, indices)
            global_xyzs.append(global_position)
            e = global_position - cano_vertices_xyz[:, indices, :]
            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)
            h = torch.sigmoid(h)
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

        return [local_xyz, local_rots, local_scalings, global_alpha[0], global_features[0], all_indices]

    def inference_render(self, camera, attr, background, usecnn=True):
        flame_parea, _, _ = self.merge_flame_params([camera])
        self.update_mesh_by_param_dict_torch(flame_parea)
        global_xyz = self.local2global_xyz(attr[0], attr[5])
        global_rotation = self.local2global_rotation(attr[1], attr[5])
        global_scale = self.local2global_scale(attr[2], attr[5])

        rendering = gs_render.render(camera, global_xyz[0], global_rotation[0], global_scale[0], attr[3], attr[4],
                                     background)
        I_fea = torch.cat((rendering["render"], rendering["feature"]), dim=0).unsqueeze(0)
        if usecnn:
            I_fea = self.cnn(I_fea)

        return I_fea[0]

    def fintune_w(self, cameras: Camera, background: torch.tensor, usecnn=False):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []

        softmax_weights = F.softmax(self.w, dim=0)
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)

        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(indices)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]
            B, d = latent_f.shape
            N, D = latent_z.shape
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[..., 7:10]
            gs_opacity = gs_attr[..., -1].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, indices)
            global_xyzs.append(global_position)
            e = global_position - cano_vertices_xyz[:, indices, :]
            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)
            h = torch.sigmoid(h)
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
        gt_rgb = []
        mouth_masks = []
        I_opecity = []
        gt_opecity = []
        v_filter = []
        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[index], global_rotation[index], global_scale[index],
                                         global_alpha[index], global_features[index], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(torch.cat((rendering["render"], rendering["feature"]), dim=0))
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            if hasattr(camera, 'one_hot_label'):
                mouth_masks.append(camera.one_hot_label[2].cuda(self.cuda_index))
            else:
                mouth_masks.append(torch.zeros_like(rendering["opacity"])[0])

            v_filter.append(rendering["visibility_filter"])

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter = torch.stack(v_filter, dim=0)
        gt_mask_mouth = torch.stack(mouth_masks, dim=0).unsqueeze(1)

        if usecnn:
            I = self.cnn(I_fea)
            losses.append(rec_loss(I, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                                   self.lpips_fn))
            # losses.append(rec_loss(I*gt_mask_mouth,gt_rgb*gt_mask_mouth,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.lpips_fn)* self.config.lambda_m)
        else:
            pass
            # losses.append(rec_loss(I_rgb*gt_mask_mouth,gt_rgb*gt_mask_mouth,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.lpips_fn)* self.config.lambda_m)

        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.lpips_fn))
        losses.append(l1_loss(I_opecity, gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(
            F.relu(local_xyz[v_filter].norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings[v_filter]) - self.config.threshold_scale).norm(
            dim=-1).mean() * self.config.lambda_s)
        if usecnn:
            images = I[0]
        else:
            images = I_rgb[0]
        return sum(losses), images, [local_xyz[0].clone(), global_features[0, :, :3].clone(), gs_op[0].clone(),
                                     local_scalings[0].clone(), local_rots[0].clone(),
                                     self.binding[all_indices].clone()]

    def create_gsavatars_model(self, avatars_id, xyz, rgb, opacity, scaling, rotation, binding, sh_degree):
        # C0 = 0.28209479177387814
        gaussians = FlameGaussianModel(sh_degree, 0, False)
        gaussians.flame_param = self.flame_param[avatars_id]
        gaussians._xyz = xyz
        gaussians._features_dc = RGB2SH(rgb[..., :3].unsqueeze(1))  # /C0
        num_dim = ((sh_degree + 1) ** 2) - 1
        gaussians._features_seg = torch.zeros((gaussians._features_dc.shape[0], 1, gaussians.seg_num_classes),
                                              device=xyz.device)
        gaussians._features_rest = torch.zeros(
            (gaussians._features_dc.shape[0], num_dim, gaussians._features_dc.shape[2]), device=xyz.device)
        gaussians._opacity = opacity
        gaussians._scaling = scaling
        gaussians._rotation = rotation
        gaussians.binding = binding

        return gaussians, rgb[..., 3:]

    def forward(self, cameras: Camera, background: torch.tensor, usecnn=True):
        flame_parea, list_avatar_ids, cano_vertices_xyz = self.merge_flame_params(cameras)
        self.update_mesh_by_param_dict_torch(flame_parea)

        gs_alpha = []
        gs_features = []
        local_xyz = []
        local_rots = []
        all_indices = []
        local_scalings = []
        global_xyzs = []

        latent_f_B = self.latent_f[list_avatar_ids]

        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            all_indices.append(indices)
            latent_z = self.latent_z[indices]
            latent_f = latent_f_B[:, part_index, :]
            B, d = latent_f.shape
            N, D = latent_z.shape
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1 = self.mlp1s[part_index]
            mlp2 = self.mlp2s[part_index]

            gs_attr = mlp1(input)

            local_position = gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling = gs_attr[..., 7:10]
            gs_opacity = gs_attr[..., -1].unsqueeze(-1)
            global_position = self.local2global_xyz(local_position, indices)
            global_xyzs.append(global_position)
            e = global_position - cano_vertices_xyz[:, indices, :]
            input2 = torch.cat((input, e, gs_attr), dim=2)
            h = mlp2(input2)
            h = torch.sigmoid(h)
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
        gt_rgb = []
        mouth_masks = []
        I_opecity = []
        gt_opecity = []
        v_filter = []
        for index, camera in enumerate(cameras):
            rendering = gs_render.render(camera, global_xyz[index], global_rotation[index], global_scale[index],
                                         global_alpha[index], global_features[index], background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(torch.cat((rendering["render"], rendering["feature"]), dim=0))
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            if hasattr(camera, 'one_hot_label'):
                mouth_masks.append(camera.one_hot_label[2].cuda(self.cuda_index))
            else:
                mouth_masks.append(torch.zeros_like(rendering["opacity"])[0])

            v_filter.append(rendering["visibility_filter"])

        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea = torch.stack(I_fea, dim=0)
        I_opecity = torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter = torch.stack(v_filter, dim=0)
        gt_mask_mouth = torch.stack(mouth_masks, dim=0).unsqueeze(1)
        if usecnn:
            I = self.cnn(I_fea)
            losses.append(rec_loss(I, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                                   self.lpips_fn))
            losses.append(
                rec_loss(I * gt_mask_mouth, gt_rgb * gt_mask_mouth, self.config.lambda_l1, self.config.lambda_ssmi,
                         self.config.lambda_lpips, self.lpips_fn) * self.config.lambda_m)

        losses.append(
            rec_loss(I_rgb * gt_mask_mouth, gt_rgb * gt_mask_mouth, self.config.lambda_l1, self.config.lambda_ssmi,
                     self.config.lambda_lpips, self.lpips_fn) * self.config.lambda_m)

        losses.append(rec_loss(I_rgb, gt_rgb, self.config.lambda_l1, self.config.lambda_ssmi, self.config.lambda_lpips,
                               self.lpips_fn))
        losses.append(l1_loss(I_opecity, gt_opecity).mean() * self.config.lambda_alpha)
        losses.append(
            F.relu(local_xyz[v_filter].norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings[v_filter]) - self.config.threshold_scale).norm(
            dim=-1).mean() * self.config.lambda_s)
        if usecnn:
            images = [I_rgb[0], I[0], gt_rgb[0], I_opecity[0], gt_opecity[0]]
        else:
            images = [I_rgb[0], gt_rgb[0], I_opecity[0], gt_opecity[0]]
        return sum(losses), images, [local_xyz[0].clone(), global_features[0, :, :3].clone(), gs_op[0].clone(),
                                     local_scalings[0].clone(), local_rots[0].clone(),
                                     self.binding[all_indices].clone()]

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

    def load_meshes(self, train_meshes, test_meshes, avatarid, load_static_offset=True, face_offset_to_zero=False):
        meshes = {**train_meshes, **test_meshes}
        pose_meshes = meshes

        self.num_timesteps = max(pose_meshes) + 1  # required by viewers
        num_verts = self.flame_model.v_template.shape[0]

        # if not self.disable_flame_static_offset:
        #     static_offset = torch.zeros([num_verts, 3])
        # else:
        if not load_static_offset or 'static_offset' not in meshes[0]:
            static_offset = torch.zeros([num_verts, 3])
        else:
            static_offset = torch.from_numpy(meshes[0]['static_offset'])

            if face_offset_to_zero:
                face_v_id = self.flame_model.mask.get_vid_by_region(["hair", "boundary", "neck"])
                static_offset[:, face_v_id, :] = 0

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

        flame_param_nn = nn.ParameterDict({
            key: nn.Parameter(value).float().cuda(self.cuda_index) if isinstance(value, torch.Tensor) else value
            for key, value in flame_param.items()
        }).cuda(self.cuda_index)

        self.flame_param[avatarid] = flame_param_nn
        self.set_cano_vertices_xyz(flame_param_nn, avatarid)

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

