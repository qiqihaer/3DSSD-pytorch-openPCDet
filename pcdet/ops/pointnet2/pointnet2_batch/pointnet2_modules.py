from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
import pcdet.utils.SSD as SSD
import pcdet.ops.pointnet2.pointnet2_3DSSD.pointnet2_utils as pointnet2_3DSSD


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None, flag_SSD=False) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if flag_SSD is False:
            if new_xyz is None:
                new_xyz = pointnet2_utils.gather_operation(
                    xyz_flipped,
                    pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                ).transpose(1, 2).contiguous() if self.npoint is not None else None
        else:
            features_SSD = torch.cat([xyz, features.transpose(1, 2)], dim=-1)
            features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
            features_for_fps_distance = features_for_fps_distance.contiguous()
            fps_idx_1 = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, self.npoint)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, fps_idx_1
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method



class PointnetSAModuleMSG_SSD(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool', out_channle=-1, fps_type='D-FPS', fps_range=-1,
                 dilated_group=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()
        self.fps_types = fps_type
        self.fps_ranges = fps_range
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

        if out_channle != -1 and len(self.mlps) > 0:
            in_channel = 0
            for mlp_tmp in mlps:
                in_channel += mlp_tmp[-1]
            shared_mlps = []
            shared_mlps.extend([
                nn.Conv1d(in_channel, out_channle, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channle),
                nn.ReLU()
            ])
            self.out_aggregation = nn.Sequential(*shared_mlps)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None, ctr_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        if ctr_xyz is None:
            last_fps_end_index = 0
            fps_idxes = []
            for i in range(len(self.fps_types)):
                fps_type = self.fps_types[i]
                fps_range = self.fps_ranges[i]
                npoint = self.npoint[i]
                if npoint == 0:
                    continue
                if fps_range == -1:
                    xyz_tmp = xyz[:, last_fps_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
                else:
                    xyz_tmp = xyz[:, last_fps_end_index:fps_range, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:fps_range, :]
                    last_fps_end_index += fps_range
                if fps_type == 'D-FPS':
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                elif fps_type == 'F-FPS':
                    # features_SSD = xyz_tmp
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                elif fps_type == 'FS':
                    # features_SSD = xyz_tmp
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx_1 = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    fps_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)  # [bs, npoint * 2]
                fps_idxes.append(fps_idx)
            fps_idxes = torch.cat(fps_idxes, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, fps_idxes
            ).transpose(1, 2).contiguous() if self.npoint is not None else None
        else:
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)
            new_features = self.out_aggregation(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, fps_idxes).contiguous()

        return new_xyz, new_features


class Vote_layer(nn.Module):
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        for i in range(len(mlp_list)):
            shared_mlps = []

            shared_mlps.extend([
                nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_list[i]),
                nn.ReLU()
            ])
            pre_channel = mlp_list[i]
        self.mlp_modules = nn.Sequential(*shared_mlps)

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.min_offset = torch.tensor(max_translate_range).float().view(1, 1, 3)

    def forward(self, xyz, features):

        new_features = self.mlp_modules(features)
        ctr_offsets = self.ctr_reg(new_features)

        ctr_offsets = ctr_offsets.transpose(1, 2)

        min_offset = self.min_offset.repeat((xyz.shape[0], xyz.shape[1], 1)).to(xyz.device)

        limited_ctr_offsets = torch.where(ctr_offsets < min_offset, min_offset, ctr_offsets)
        min_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(limited_ctr_offsets > min_offset, min_offset, limited_ctr_offsets)
        xyz = xyz + limited_ctr_offsets
        return xyz, new_features, ctr_offsets


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
