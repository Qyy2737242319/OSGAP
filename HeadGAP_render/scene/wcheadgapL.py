from re import I
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flame_model.flame import FlameHead
from utils.graphics_utils import compute_face_orientation
from utils.general_utils import Pytorch3dRasterizer,face_vertices_gen
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from scene.cameras import Camera
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from gaussian_renderer import gs_render,gs_render_raw
from utils.loss_utils import rec_loss,l1_loss
from torch.cuda.amp import autocast
from scene import FlameGaussianModel
from core.utils.network_util import set_requires_grad, trunc_normal_
import torch.nn.init as init
class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_c,hidden_layers):
        super(CNN, self).__init__()
        
        # 初始化通道数设置，逐步递增
       # channels = input_channels
        layers = []
        
        self.input_layer = nn.Conv2d(input_channels, hidden_c, kernel_size=3, stride=1, padding=1)
        
        # 添加隐藏层
        for i in range(hidden_layers-1):
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
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=4,dropout_rate=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        
  
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        
        #self._initialize_weights()
    
    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
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

    def __init__(self, input_dim, output_dim,hidden_dim, num_layers):
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



class WCHeadGAPModelL(nn.Module):
    def __init__(self, config,avatar_id_list,part_id_list,cuda_index=0,n_shape=300, n_expr=100):
        super(WCHeadGAPModelL, self).__init__()
        
        
        self.cuda_index=cuda_index
        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda(self.cuda_index)
  
        #some info
        self.config=config
        self.uv_size=config.uv_size
        self.avatars_list=avatar_id_list
        self.avatars_num=len(avatar_id_list)
        
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
       
        
        #initial sth.
        self.initial_binding_infor(config.uv_size,1)
        self.initial_part_info(part_id_list)
        self.cano_vertices_xyz={}
        
        
        #initial latent
        self.latent_z = nn.Parameter(torch.randn(self.gs_num, config.latent_z_dim))
       
        self.latent_f = nn.Parameter(torch.randn(self.avatars_num,len(part_id_list),config.latent_f_dim))

        # trunc_normal_(self.latent_z, std=.02)
        # trunc_normal_(self.latent_f, std=.02)
 
        self.flame_param=nn.ParameterDict()
        
        self.mlp1s = nn.ModuleList()
        self.mlp2s = nn.ModuleList()
       
        for part_id in part_id_list:
            self.mlp1s.append(MLP(config.latent_f_dim+config.latent_z_dim, 11, config.mlp1_hidden_dim,config.mlp1_hidden_layers))
            self.mlp2s.append(MLP(config.latent_f_dim+config.latent_z_dim+3+11, config.latent_h_dim, config.mlp2_hidden_dim, config.mlp2_hidden_layers))

        self.cnn=CNN(config.latent_h_dim,3,config.cnn_hidden_channels,config.cnn_hidden_layers)
        
    
    def get_opt_param(self,lr=0.001):
        #indices = torch.where(self.gs_part_infor == 9)[0].tolist()
        
        param_groups = [
            {"params": [self.latent_z, self.latent_f], "lr": lr}, 
            {"params": [p for mlp in self.mlp1s for p in mlp.parameters()], "lr": lr},  
            {"params": [p for mlp in self.mlp2s for p in mlp.parameters()], "lr": lr},  
            {"params": list(self.cnn.parameters()), "lr": lr},  
        ]
        def collect_params(param_dict, lr):
            for key, value in param_dict.items():
                if isinstance(value, nn.Parameter):
                    if key in {"shape", "static_offset", "dynamic_offset"}:
                    #if key in {"dynamic_offset"}:
                        value.requires_grad=False
                    else:
                        param_groups.append({"params": [value], "lr": lr})
                elif isinstance(value, dict):
                    collect_params(value, lr)
        
        collect_params(self.flame_param, lr=1e-5)
        
        return param_groups
        
         
    
    def initial_part_info(self,part_id_list):
        mask_faces=self.flame_model.mask.get_headgap_part(part_id_list).to(self.binding.device)
        self.gs_part_infor=mask_faces[self.binding]
        
    def set_cano_vertices_xyz(self,flame_param,name):
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
            batch_size=1
            uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
            face_vertices_shape = face_vertices_gen(verts, self.flame_model.faces.expand(batch_size, -1, -1))
            verts_uvs=self.flame_model.verts_uvs
            verts_uvs = verts_uvs * 2 - 1
            verts_uvs[..., 1] = - verts_uvs[..., 1]
            verts_uvs=verts_uvs[None]
            verts_uvs = torch.cat([verts_uvs, verts_uvs[:, :, 0:1] * 0. + 1.], -1)
            rast_out, pix_to_face, bary_coords = uv_rasterizer(verts_uvs.expand(batch_size, -1, -1),
                                            self.flame_model.textures_idx.expand(batch_size, -1, -1),
                                            face_vertices_shape)
            uvmask = rast_out[:, -1].unsqueeze(1)
            uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
            uvmask_flaten_idx = (uvmask_flaten[:]>0)
        
            
            uv_vertices_shape = rast_out[:, :3]
            uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
            uv_vertices_shape = uv_vertices_shape_flaten[uvmask_flaten_idx].unsqueeze(0)
            
            assert uv_vertices_shape[0].size(0) == self.gs_num, f"There is a problem with sampling"
            self.cano_vertices_xyz[name]=uv_vertices_shape[0]
        
   
    def initial_binding_infor(self,uv_size,batch_size):
        uv_rasterizer = Pytorch3dRasterizer(uv_size)
        face_vertices_shape = face_vertices_gen(self.flame_model.v_template.expand(batch_size, -1, -1), self.flame_model.faces.expand(batch_size, -1, -1))
        verts_uvs=self.flame_model.verts_uvs
        verts_uvs = verts_uvs * 2 - 1
        verts_uvs[..., 1] = - verts_uvs[..., 1]
        verts_uvs=verts_uvs[None]
        verts_uvs = torch.cat([verts_uvs, verts_uvs[:, :, 0:1] * 0. + 1.], -1)
        rast_out, pix_to_face, bary_coords = uv_rasterizer(verts_uvs.expand(batch_size, -1, -1),
                                        self.flame_model.textures_idx.expand(batch_size, -1, -1),
                                        face_vertices_shape)
        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
        uvmask_flaten_idx = (uvmask_flaten[:]>0)

        pix_to_face_flaten = pix_to_face[0].clone().view(-1) # batch=1
        self.binding = pix_to_face_flaten[uvmask_flaten_idx] # pix to face idx
        #self.pix_to_v_idx = self.flame_model.faces.expand(batch_size, -1, -1)[0, self.binding, :] # pix to vert idx

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
        uv_vertices_shape = uv_vertices_shape_flaten[uvmask_flaten_idx].unsqueeze(0)

        
        # xyz_data = uv_vertices_shape[0].cpu().detach().numpy()

        # # 保存为 .xyz 文件
        # with open("output.xyz", "w") as file:
        #     for point in xyz_data:
        #         # 写入每行的 x, y, z 坐标
        #         file.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        #self.cano_vertices_xyz = uv_vertices_shape[0] # for cano init
        self.gs_num=uv_vertices_shape.shape[1]
                 
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
    
    def local2global_xyz(self,local_xyz,indices):
        xyz = torch.matmul(self.face_orien_mat[:, self.binding[indices], :, :], local_xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[:, self.binding[indices], :] + self.face_center[:, self.binding[indices], :]
    
    def local2global_rotation(self,local_rotation,indices):
        rot = self.rotation_activation(local_rotation,dim=-1)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[:, self.binding[indices], :],dim=-1)
        # rot=local_rotation
        # face_orien_quat=self.face_orien_quat[:, self.binding[indices], :]
        return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot))) 

    def local2global_scale(self,local_scale,indices):
        local_scale=self.scaling_activation(local_scale)
        return local_scale * self.face_scaling[:, self.binding[indices], :]#+0.001
    
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

    def merge_flame_params(self,camera_list):
        # 获取第一个 Camera 对象的 flame_param 结构，用于初始化 merged_dict
        sample_flame_param =  self.flame_param[camera_list[0].avatar_id]
        merged_dict = {key: [] for key in sample_flame_param.keys()}
        list_avatar_ids=[]
        cano_vertices_xyz=[]
        # 遍历所有 Camera 对象，收集每个 flame_param 的值
        for camera in camera_list:
            cano_vertices_xyz.append(self.cano_vertices_xyz[camera.avatar_id])
            list_avatar_ids.append(self.avatars_list.index(camera.avatar_id) if camera.avatar_id in self.avatars_list else -1)
            flame_param =self.flame_param[camera.avatar_id]
            for key, value in flame_param.items():
                if torch.is_tensor(value):
                    if key=="dynamic_offset":
                        continue
                    if value.dim()==1:#shape
                        merged_dict[key].append(value)
                    else:
                        if value.shape[0] == 1:#stasistic_offset
                            merged_dict[key].append(value.squeeze(0))
                        else:
                            merged_dict[key].append(value[camera.timestep])

        merged_dict = {
        key: torch.stack(value_list, axis=0) if len(value_list) > 0 else None
        for key, value_list in merged_dict.items()
        }


        return merged_dict,list_avatar_ids, torch.stack(cano_vertices_xyz, dim=0)
    
    
    def fintune_rgb_setup(self):
        temp=self.gs_features[0,:,0:3].clone()
        self.gsrgb=torch.nn.Parameter(temp)
        self.optimizer = torch.optim.Adam([self.gsrgb], lr=0.001)
    
    def fintune_w_setup(self):
        self.w = torch.nn.Parameter(torch.zeros(self.latent_f.shape[0], 11, 1).cuda(self.cuda_index))
        self.optimizer = torch.optim.Adam([self.w], lr=0.001)
    
    def fintune_network_setup(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.latent_z, 'lr': 0.001},  # 为 latent_z 设置学习率 0.001
            {'params': self.latent_f, 'lr': 0.00001},  # 为 latent_f 设置学习率 0.00001
            {'params': self.mlp1s.parameters(), 'lr': 0.00001},  # 为 mlp1s 设置学习率 0.00001
            {'params': self.mlp2s.parameters(), 'lr': 0.00001},  # 为 mlp2s 设置学习率 0.00001
            {'params': self.cnn.parameters(), 'lr': 0.00001}  # 为 cnn 设置学习率 0.00001
        ])
    
    def fintune_network(self, cameras,background: torch.tensor,usecnn=False):
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params([cameras[0]])
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
      
        gs_alpha=[]
        gs_features=[]
        local_xyz=[]
        local_rots=[]
        all_indices=[]
        local_scalings=[]
        global_xyzs=[]
        
        
        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)
        
        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            
            all_indices.append(indices)
            latent_z =self.latent_z[indices]
            latent_f = latent_f_B[:,part_index,:]
            
            if part_index == 3 or part_index == 4:
                latent_f.detach()
            
            B,d=latent_f.shape
            N,D=latent_z.shape
            
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1=self.mlp1s[part_index]
            mlp2=self.mlp2s[part_index]
            
            gs_attr=mlp1(input)
  
            local_position=gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling=gs_attr[..., 7:10]
            gs_opacity=gs_attr[..., -1].unsqueeze(-1)
            global_position=self.local2global_xyz(local_position,indices)
            global_xyzs.append(global_position)
            e=global_position-cano_vertices_xyz[:,indices,:]
            input2=torch.cat((input,e,gs_attr), dim=2)
            h=mlp2(input2)
            h[..., :3] = torch.sigmoid(h[..., :3])
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)
        
        all_indices=torch.cat(all_indices)
        local_scalings=torch.cat(local_scalings, dim=1) 
        local_xyz=torch.cat(local_xyz, dim=1)
        local_rots=torch.cat(local_rots, dim=1)
        gs_op=torch.cat(gs_alpha, dim=1)
        global_alpha=self.opacity_activation(gs_op)
        global_features=torch.cat(gs_features, dim=1)
        global_xyz=torch.cat(global_xyzs, dim=1)
        global_rotation=self.local2global_rotation(local_rots,all_indices)
        global_scale=self.local2global_scale(local_scalings,all_indices)

        losses=[]
        I_fea=[]
        I_rgb=[]
        gt_rgb=[]
        I_opecity=[]
        gt_opecity=[]
        v_filter=[]
        gt_ref=[]
        for index, camera in enumerate(cameras):
            rendering=gs_render.render(camera,global_xyz[0],global_rotation[0],global_scale[0],global_alpha[0],global_features[0],background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_ref.append(camera.ref_image.cuda(self.cuda_index))
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])
            
        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea=torch.stack(I_fea, dim=0)
        I_opecity=torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter= torch.stack(v_filter, dim=0)
        gt_ref = torch.stack(gt_ref, dim=0)
        if usecnn:
            I=self.cnn(I_fea)
            losses.append(rec_loss(I,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean())
            losses.append(rec_loss(I,gt_ref,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).sum()* self.config.lambda_ref)
        else:
            losses.append(rec_loss(I_rgb,gt_ref,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).sum()* self.config.lambda_ref)
        
        losses.append(rec_loss(I_rgb,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean()) 
        #losses.append(l1_loss(I_opecity,gt_opecity).mean()*self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings) - self.config.threshold_scale).norm(dim=-1).mean() * self.config.lambda_s)
    
        if usecnn:    
            images=I
        else:
            images=I_rgb
        return sum(losses),images,[local_xyz[0].clone(),global_features[0,:,:3].clone(),gs_op[0].clone(),local_scalings[0].clone(),local_rots[0].clone(),self.binding[all_indices].clone()]
    
    def fintune_w(self, cameras: Camera,background: torch.tensor,usecnn=False):
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params([cameras[0]])
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
      
        gs_alpha=[]
        gs_features=[]
        local_xyz=[]
        local_rots=[]
        all_indices=[]
        local_scalings=[]
        global_xyzs=[]
        
        
        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0)
        
        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            
            all_indices.append(indices)
            latent_z =self.latent_z[indices]
            latent_f = latent_f_B[:,part_index,:]
            B,d=latent_f.shape
            N,D=latent_z.shape
            
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1=self.mlp1s[part_index]
            mlp2=self.mlp2s[part_index]
            
            gs_attr=mlp1(input)
  
            local_position=gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling=gs_attr[..., 7:10]
            gs_opacity=gs_attr[..., -1].unsqueeze(-1)
            global_position=self.local2global_xyz(local_position,indices)
            global_xyzs.append(global_position)
            e=global_position-cano_vertices_xyz[:,indices,:]
            input2=torch.cat((input,e,gs_attr), dim=2)
            h=mlp2(input2)
            h[..., :3] = torch.sigmoid(h[..., :3])
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)
        
        all_indices=torch.cat(all_indices)
        local_scalings=torch.cat(local_scalings, dim=1) 
        local_xyz=torch.cat(local_xyz, dim=1)
        local_rots=torch.cat(local_rots, dim=1)
        gs_op=torch.cat(gs_alpha, dim=1)
        global_alpha=self.opacity_activation(gs_op)
        global_features=torch.cat(gs_features, dim=1)
        global_xyz=torch.cat(global_xyzs, dim=1)
        global_rotation=self.local2global_rotation(local_rots,all_indices)
        global_scale=self.local2global_scale(local_scalings,all_indices)

        losses=[]
        I_fea=[]
        I_rgb=[]
        gt_rgb=[]
        I_opecity=[]
        gt_opecity=[]
        v_filter=[]
        for index, camera in enumerate(cameras):
            rendering=gs_render.render(camera,global_xyz[0],global_rotation[0],global_scale[0],global_alpha[0],global_features[0],background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])
            
        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea=torch.stack(I_fea, dim=0)
        I_opecity=torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter= v_filter[0]
        
        if usecnn:
            I=self.cnn(I_fea)
            losses.append(rec_loss(I,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean())
        losses.append(rec_loss(I_rgb,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean()) 
        #losses.append(l1_loss(I_opecity,gt_opecity).mean()*self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings) - self.config.threshold_scale).norm(dim=-1).mean() * self.config.lambda_s)


        if usecnn:    
            images=I
        else:
            images=I_rgb
        return sum(losses),images

  
    
    def create_gsavatars_model(self,avatars_id,xyz,rgb,opacity,scaling,rotation,binding):
        gaussians = FlameGaussianModel(0, False)
        gaussians.flame_param=self.flame_param[avatars_id]
        gaussians._xyz=xyz
        gaussians._features_dc=rgb.unsqueeze(1)
        gaussians._features_rest=torch.zeros((rgb.shape[0],0,rgb.shape[1]))
        gaussians._opacity=opacity
        gaussians._scaling=scaling
        gaussians._rotation=rotation
        gaussians.binding=binding
        
        return gaussians
    
        
    def forward(self, cameras: Camera,background: torch.tensor,useseg=True):
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params(cameras)
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
      
        gs_alpha=[]
        gs_features=[]
        local_xyz=[]
        local_rots=[]
        all_indices=[]
        local_scalings=[]
        global_xyzs=[]
        
        latent_f_B=self.latent_f[list_avatar_ids]
        
        for part_index in range(11):
            indices = torch.where(self.gs_part_infor == part_index)[0]
            
            all_indices.append(indices)
            latent_z =self.latent_z[indices]
            latent_f = latent_f_B[:,part_index,:]
            B,d=latent_f.shape
            N,D=latent_z.shape
            
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1=self.mlp1s[part_index]
            mlp2=self.mlp2s[part_index]
            
            gs_attr=mlp1(input)
  
            local_position=gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling=gs_attr[..., 7:10]
            gs_opacity=gs_attr[..., -1].unsqueeze(-1)
            global_position=self.local2global_xyz(local_position,indices)
            global_xyzs.append(global_position)
            e=global_position-cano_vertices_xyz[:,indices,:]
            input2=torch.cat((input,e,gs_attr), dim=2)
            h=mlp2(input2)
            h[..., :3] = torch.sigmoid(h[..., :3])
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)
        
        all_indices=torch.cat(all_indices)
        local_scalings=torch.cat(local_scalings, dim=1) 
        local_xyz=torch.cat(local_xyz, dim=1)
        local_rots=torch.cat(local_rots, dim=1)
        gs_op=torch.cat(gs_alpha, dim=1)
        global_alpha=self.opacity_activation(gs_op)
        global_features=torch.cat(gs_features, dim=1)
        global_xyz=torch.cat(global_xyzs, dim=1)
        global_rotation=self.local2global_rotation(local_rots,all_indices)
        global_scale=self.local2global_scale(local_scalings,all_indices)

        losses=[]
        I_fea=[]
        I_rgb=[]
        gt_rgb=[]
        I_opecity=[]
        gt_opecity=[]
        gt_label=[]
        v_filter=[]
        for index, camera in enumerate(cameras):
            rendering=gs_render.render(camera,global_xyz[index],global_rotation[index],global_scale[index],global_alpha[index],global_features[index],background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            if useseg:
                gt_label.append(camera.seglabel.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])
            
        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea=torch.stack(I_fea, dim=0)
        I_opecity=torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter= torch.stack(v_filter, dim=0)
        if useseg:
            gt_label=torch.stack(gt_label, dim=0).float()
            I_label=F.softmax(I_fea[:, 3:14, :, :], dim=1)
            losses.append(l1_loss(I_label,gt_label).mean())
        losses.append(rec_loss(I_rgb,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean()) 
        losses.append(l1_loss(I_opecity,gt_opecity).mean()*self.config.lambda_alpha)
        losses.append(F.relu(local_xyz[v_filter].norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(self.scaling_activation(local_scalings[v_filter]) - self.config.threshold_scale).norm(dim=-1).mean() * self.config.lambda_s)


        if useseg:    
            images=[I_rgb[0], gt_rgb[0] ,I_opecity[0],gt_opecity[0],I_label[0],gt_label[0]]
        else:
            images=[I_rgb[0], gt_rgb[0] ,I_opecity[0],gt_opecity[0]]
        return sum(losses),images,[local_xyz[0].clone(),global_features[0,:,:3].clone(),gs_op[0].clone(),local_scalings[0].clone(),local_rots[0].clone(),self.binding[all_indices].clone()]

    def forward_all_view(self, cameras: Camera,background: torch.tensor):
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params([cameras[0]])
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
        # gs_xyz=[]
        # gs_rotation=[]
        # gs_scale=[]
        gs_alpha=[]
        gs_features=[]
        local_xyz=[]
        local_rots=[]
        all_indices=[]
        local_scalings=[]
        
        # latent_f_B=[self.latent_f[key].unsqueeze(0) for key in list_avatar_ids if key in self.latent_f]
        # latent_f_B=torch.cat(latent_f_B, dim=0)
        
        latent_f_B=self.latent_f[list_avatar_ids]
        
        for part_index in range(11):
            #prepare latent by part id
            indices = torch.where(self.gs_part_infor == part_index)[0]
            
            all_indices.append(indices)
            #latent_z =torch.nn.functional.normalize(self.latent_z[indices],dim=-1)
            latent_z =self.latent_z[indices]
            latent_f = latent_f_B[:,part_index,:]
            B,d=latent_f.shape
            N,D=latent_z.shape
            
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)

            # Step 3: 在最后一个维度上拼接
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            

            mlp1=self.mlp1s[part_index]
            mlp2=self.mlp2s[part_index]
            
            gs_attr=mlp1(input)
            #gs_attr = torch.tanh(gs_attr)
            
            local_position=gs_attr[..., :3]
            rot_delta_0 = gs_attr[..., 3:7]
            rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
            rot_delta_v = rot_delta_0[..., 1:]
            local_rot= torch.cat((rot_delta_r, rot_delta_v), dim=-1)
            local_scaling=self.scaling_activation(gs_attr[..., 7:10])
            gs_opacity=self.opacity_activation(gs_attr[..., -1].unsqueeze(-1))
            
            
            global_xyz=self.local2global_xyz(local_position,indices)
            e=global_xyz-cano_vertices_xyz[:,indices,:]
          
            
            input2=torch.cat((input,e,local_position,local_rot,local_scaling,gs_opacity), dim=2)
            h=mlp2(input2)
            #h = F.relu(h)
            local_scalings.append(local_scaling)
            local_xyz.append(local_position)
            local_rots.append(local_rot)
            gs_alpha.append(gs_opacity)
            gs_features.append(h)
        
        all_indices=torch.cat(all_indices)
        local_scalings=torch.cat(local_scalings, dim=1) 
        local_xyz=torch.cat(local_xyz, dim=1)
        local_rots=torch.cat(local_rots, dim=1)
        gs_alpha=torch.cat(gs_alpha, dim=1)
        gs_features=torch.cat(gs_features, dim=1)
        
        
        gs_xyz=self.local2global_xyz(local_xyz,all_indices)
        gs_rotation=self.local2global_rotation(local_rots,all_indices)
        gs_scale=self.local2global_scale(local_scalings,all_indices)
        
        losses=[]
        I_fea=[]
        I_rgb=[]
        gt_rgb=[]
        I_opecity=[]
        gt_opecity=[]
        v_filter=[]
        for index, camera in enumerate(cameras):
            rendering=gs_render.render(camera,gs_xyz[0],gs_rotation[0],gs_scale[0],gs_alpha[0],gs_features[0],background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
            v_filter.append(rendering["visibility_filter"])
            
        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea=torch.stack(I_fea, dim=0)
        #I_rgb=I_fea[:,:3,:,:]
        I_opecity=torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
        v_filter= torch.stack(v_filter, dim=0)
        I=self.cnn(I_fea)
    
    
        losses.append(rec_loss(I,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean())
        losses.append(rec_loss(I_rgb,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean()) 
        losses.append(l1_loss(I_opecity,gt_opecity).mean()*self.config.lambda_alpha)
        losses.append(F.relu(local_xyz.norm(dim=-1) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        losses.append(F.relu(local_scalings - self.config.threshold_scale).norm(dim=-1).mean() * self.config.lambda_s)
    
            
        images=[I_rgb[0], I[0] , gt_rgb[0] ,I_opecity[0],gt_opecity[0]]
        #images=[I_rgb[0], torch.zeros_like(I_rgb[0]) , gt_rgb[0] ,I_opecity[0],gt_opecity[0]]
        return sum(losses),images,gs_xyz[0].clone().squeeze(0).cpu()
    
    def inference_get_gaussian_attr(self, cameras: Camera):
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params(cameras)
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
        gs_xyz=[]
        gs_rotation=[]
        gs_scale=[]
        gs_alpha=[]
        gs_features=[]
        local_xyz=[]
        local_scale=[]
        local_rotation=[]
        gs_indices=[]
        
        softmax_weights = F.softmax(self.w, dim=0)  # 计算 softmax 权重，形状 [k, p, 1]
        latent_f_B = torch.sum(softmax_weights * self.latent_f, dim=0).unsqueeze(0).repeat(len(cameras), 1, 1)
        
        
        for part_index in range(11):
            #prepare latent by part id
            indices = torch.where(self.gs_part_infor == part_index)[0]
            latent_z =self.latent_z[indices]
            latent_f = latent_f_B[:,part_index,:]
            B,d=latent_f.shape
            N,D=latent_z.shape
            
            latent_z_expanded = latent_z.unsqueeze(0).expand(B, N, D)
            latent_f_expanded = latent_f.unsqueeze(1).expand(B, N, d)

            # Step 3: 在最后一个维度上拼接
            input = torch.cat([latent_z_expanded, latent_f_expanded], dim=-1)

            mlp1=self.mlp1s[part_index]
            mlp2=self.mlp2s[part_index]
            
            gs_attr=mlp1(input)
            
            global_xyz=self.local2global_xyz(gs_attr[..., :3],indices)
            e=global_xyz-cano_vertices_xyz[:,indices,:]
            
            input2=torch.cat((input,e,gs_attr), dim=2)
            h=mlp2(input2)
            
            local_xyz.append(gs_attr[..., :3])
            local_scale.append(gs_attr[..., 7:10])
            gs_indices.append(indices)
            gs_xyz.append(global_xyz)
            local_rotation.append(gs_attr[..., 3:7])
            gs_rotation.append(self.local2global_rotation(gs_attr[..., 3:7],indices))
            gs_scale.append(self.local2global_scale(gs_attr[..., 7:10],indices))
            gs_alpha.append(self.opacity_activation(gs_attr[..., -1].unsqueeze(-1)))
            gs_features.append(h)
            
      
        self.gs_alpha=torch.cat(gs_alpha, dim=1)
        self.gs_features=torch.cat(gs_features, dim=1)
        self.local_xyz=torch.cat(local_xyz, dim=1)
        self.local_scale=torch.cat(local_scale, dim=1)
        self.local_rotation=torch.cat(gs_rotation, dim=1)
        self.gs_indices=torch.cat(gs_indices, dim=0)
        
        return self.local2global_xyz(self.local_xyz,self.gs_indices),self.binding[self.gs_indices],self.gs_alpha
    
    def load_meshes(self, train_meshes, test_meshes,avatarid):
        meshes = {**train_meshes, **test_meshes}
        pose_meshes = meshes
        
        self.num_timesteps = max(pose_meshes) + 1  # required by viewers
        num_verts = self.flame_model.v_template.shape[0]

        # if not self.disable_flame_static_offset:
        static_offset = torch.from_numpy(meshes[0]['static_offset'])
        if static_offset.shape[0] != num_verts:
            static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
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
            flame_param[k] =nn.Parameter(v.float().cuda(self.cuda_index))
        
        self.flame_param[avatarid]=flame_param
        self.set_cano_vertices_xyz(flame_param,avatarid)

    def inference(self, cameras: Camera,background: torch.tensor):
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params(cameras)
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
        gs_xyz=[]
        gs_rotation=[]
        gs_scale=[]
        gs_alpha=[]
        gs_features=[]
        local_xyz=[]
        local_scale=[]
        
        
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
            
        gs_xyz=self.local2global_xyz(self.local_xyz,self.gs_indices)
        gs_rotation=self.local2global_rotation(self.local_rotation,self.gs_indices)
        gs_scale=self.local2global_scale(self.local_scale,self.gs_indices)
        gs_alpha=self.gs_alpha
        gs_features=self.gs_features

        
        I_fea=[]
        I_rgb=[]
        gt_rgb=[]
        I_opecity=[]
        gt_opecity=[]
        
        for index, camera in enumerate(cameras):
            rendering=gs_render.render(camera,gs_xyz[index],gs_rotation[index],gs_scale[index],gs_alpha[index],gs_features[index],background)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(camera.original_image.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
        
        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea=torch.stack(I_fea, dim=0)
        I_opecity=torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
       
        I=self.cnn(I_fea)
    
        images=[I_rgb[0], I[0] , gt_rgb[0] ,I_opecity[0],gt_opecity[0]]
        return images
    
    
    def finetune_rgb(self, cameras: Camera,background: torch.tensor,gtimage):
        
        flame_parea,list_avatar_ids,cano_vertices_xyz=self.merge_flame_params(cameras)
        #update flame
        self.update_mesh_by_param_dict_torch(flame_parea)
      
            
        gs_xyz=self.local2global_xyz(self.local_xyz,self.gs_indices)
        gs_rotation=self.local2global_rotation(self.local_rotation,self.gs_indices)
        gs_scale=self.local2global_scale(self.local_scale,self.gs_indices)
        gs_alpha=self.gs_alpha
        gs_features=self.gs_features
        gs_rgb=self.gsrgb
        
        I_fea=[]
        I_rgb=[]
        gt_rgb=[]
        I_opecity=[]
        gt_opecity=[]
        
        for index, camera in enumerate(cameras):
            rendering=gs_render.render(camera,gs_xyz[index],gs_rotation[index],gs_scale[index],gs_alpha[index],gs_features[index],background,gs_rgb)
            I_rgb.append(rendering["render"])
            I_opecity.append(rendering["opacity"])
            I_fea.append(rendering["feature"])
            gt_rgb.append(gtimage.cuda(self.cuda_index))
            gt_opecity.append(camera.mask.cuda(self.cuda_index))
        
        I_rgb = torch.stack(I_rgb, dim=0)
        I_fea=torch.stack(I_fea, dim=0)
        I_opecity=torch.stack(I_opecity, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        gt_opecity = torch.stack(gt_opecity, dim=0)
       
        #I=self.cnn(I_fea)
    
        losses=[]
        #losses.append(rec_loss(I,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean())
        losses.append(rec_loss(I_rgb,gt_rgb,self.config.lambda_l1,self.config.lambda_ssmi,self.config.lambda_lpips,self.cuda_index).mean()) 
        #losses.append(l1_loss(I_opecity,gt_opecity).mean()*self.config.lambda_alpha)
        #losses.append(F.relu(local_xyz.norm(dim=2) - self.config.threshold_xyz).mean() * self.config.lambda_mui)
        #losses.append(F.relu(torch.exp(local_scale) - self.config.threshold_scale).norm(dim=2).mean() * self.config.lambda_s)
    
            
        images=[I_rgb[0], None , gt_rgb[0] ,I_opecity[0],gt_opecity[0]]
        return sum(losses),images,gs_xyz[0].clone().squeeze(0).cpu()
        
