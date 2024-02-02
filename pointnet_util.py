import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


# reference https://github.com/yanx27/Pointnet_Pointnet2_pytorch, modified by Yang You
# modified again by Marga Don

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, nsample, xyz, normals, points, fps_idx=None):
    """
    Input:
        npoint:
        nsample: default 16
        xyz: input points position data, [B, N, 3]
        normals: input points normal data [B, N, 3]
        points: input points data, [B, N, D]
        fps_index: work with predefined fps_index
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_normal: sampled points normal data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
        fps_index: fps index used
    """
    if fps_idx is None:
        fps_idx = farthest_point_sample(xyz, npoint)  
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    new_normals = index_points(normals, fps_idx)
    torch.cuda.empty_cache()

    # Return the features of the 16 nearest neighbors for each sampled point
    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    new_points = index_points(points, idx) # [B, npoint, nsample, D]
    
    return new_xyz, new_normals, new_points, fps_idx


def sample_and_group_all(xyz, normals, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        normals: input points normal data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_normals: sampled points normal data [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    new_normals = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_normals = normals.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        new_normals = torch.cat([grouped_normals, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
        new_normals = grouped_normals
    return new_xyz, new_normals, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, normals, points, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            normal: input points normal data, [B, N, C]
            points: input points data, [B, N, C]
            fps_idx: allow for pre-determined fps_idx
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_normal: samped points normal data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
            fps_idx: used fps_idx
        """
        new_xyz, new_normals, new_points, fps_index = sample_and_group(self.npoint, self.nsample, xyz, normals, points, fps_idx=fps_idx)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_normals: sampled points normals data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pool
        new_points = torch.max(new_points, 2)[0].transpose(1, 2) # [B, npoint, D]
        return new_xyz, new_normals, new_points, fps_index
