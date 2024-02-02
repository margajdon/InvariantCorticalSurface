import torch
import torch.nn as nn
from pointnet_util import PointNetSetAbstraction
from transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, normal, points, fps_index=None):
        return self.sa(xyz, normal, points, fps_index)

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, d_points = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.input_dim
        nneighbor_bias = cfg.nneighbor_bias

        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.transformer_dim, nneighbor_bias, npoints, cfg.pos_enc)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            npoints = npoints // 4
            self.transition_downs.append(TransitionDown(int(npoints), nneighbor, [channel // 2, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.transformer_dim, nneighbor_bias, npoints, cfg.pos_enc))
        self.nblocks = nblocks
    
    def forward(self, xyz, normals, init_features):
        points = self.transformer1(xyz, normals, self.fc1(init_features.cuda()))[0]

        xyz_normals_feats = [(xyz, normals, points)]
        fps_indices = []
        for i in range(self.nblocks):
            xyz, normals, points, fps_index = self.transition_downs[i](xyz, normals, points)
            points = self.transformers[i](xyz, normals, points)[0]
            xyz_normals_feats.append((xyz, normals, points))
            fps_indices.append(fps_index)
        return points, xyz_normals_feats, fps_indices


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, xyz, normals, init_features, scan_age):
        points, _, _ = self.backbone(xyz, normals, init_features)
        res = self.fc2(points.mean(1))
        return res
    
class PointTransformerReg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        nblocks = cfg.nblocks
        input_dim = (32 * 2 ** nblocks) + 1 if cfg.scan_age else 32 * 2 ** nblocks
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.nblocks = nblocks

    def forward(self, xyz, normals, init_features, scan_age=None):
        points, _, _ = self.backbone(xyz, normals, init_features)
        points = points.mean(1)
        if scan_age is not None:
            points = torch.hstack([points, scan_age.unsqueeze(1)])
        res = self.fc2(points)
        return res