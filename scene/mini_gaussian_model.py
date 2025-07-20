import torch


class MiniGaussianModel:
    def __init__(self, xyz, rgb, opacity, scaling, rotation, binding):
        self.is_static = True
        self.mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
        self.binding = binding
        self._xyz = xyz
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        self._rgb = rgb
        self._seg = torch.zeros((self.get_rgb.shape[0], 0, 0), device=xyz.device)
        self.active_sh_degree = 0

    def set_part_mask_by_face_id(self, face_id, is_static=True, mask_outliers=None):
        self.mask = torch.isin(self.binding, face_id)
        if mask_outliers is not None:
            self.mask = self.mask & mask_outliers
        self.is_static = is_static

    @property
    def get_binding(self):
        return self.binding[self.mask]

    @property
    def get_xyz(self):
        if self.is_static:
            return self._xyz[self.mask].clone().detach()
        else:
            return self._xyz[self.mask]

    @property
    def get_opacity(self):
        if self.is_static:
            return self._opacity[self.mask].clone().detach()
        else:
            return self._opacity[self.mask]

    @property
    def get_scaling(self, is_static=True):
        if is_static:
            return self._scaling[self.mask].clone().detach()
        else:
            return self._scaling[self.mask]

    @property
    def get_rotation(self):
        if self.is_static:
            return self._rotation[self.mask].clone().detach()
        else:
            return self._rotation[self.mask]

    @property
    def get_rgb(self):
        if self.is_static:
            return self._rgb[self.mask].clone().detach()
        else:
            return self._rgb[self.mask]

    @property
    def get_seg(self):
        if self.is_static:
            return self._seg[self.mask].clone().detach()
        else:
            return self._seg[self.mask]