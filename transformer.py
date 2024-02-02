from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, npoint_dim, pos_enc_choice) -> None:
        super().__init__()
        self.npoint_dim = npoint_dim
        self.pos_enc_choice = pos_enc_choice
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        # MLP for attention
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # MLP for positional encoding
        first_dim = 1 if self.pos_enc_choice == 'R3' else 3
        self.fc_delta = nn.Sequential(
            nn.Linear(first_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        # Q, K and V matrices
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k if k < npoint_dim else npoint_dim
        
    # xyz: b x n x 3, normals: b x n x 3, features: b x n x f
    def forward(self, xyz, normals, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        knn_normals = index_points(normals, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        # Compute positional encoding
        if 'none' in self.pos_enc_choice:
            pos_enc = torch.zeros(k.shape).cuda()  # b x n x k x f

        elif self.pos_enc_choice == 'rel_dis':
            pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)

        elif self.pos_enc_choice == 'R3':
            pos_enc = self.fc_delta(self.biasR3(xyz, knn_xyz))

        elif self.pos_enc_choice == 'R3S2':
            pos_enc = self.fc_delta(self.biasR3S2(xyz, knn_xyz, normals, knn_normals))
        
        else:
            print(f'Bias {self.pos_enc_choice} not found when computing positional encoding.')
            quit(1)

        # Add positional encoding
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

    def biasR3(self, xyz, knn_xyz):
        # Calculate relative distances
        rel_pos = xyz[:, :, None] - knn_xyz
        bias = torch.einsum('bnkf, bnkf -> bnk', rel_pos, rel_pos).unsqueeze(dim=3)
        return bias

    def biasR3S2(self, xyz, knn_xyz, normals, knn_normals):
        eps = 1e-8
        b, n, k, _= knn_normals.shape
        normals =  normals.repeat(1,1,k).reshape(b, n, k, 3)
        
        # Compute invariances
        xyz_dir = xyz[:, :, None] - knn_xyz
        first_invar = torch.einsum('bnkf, bnkf -> bnk', knn_normals, xyz_dir)
        second_invar = knn_xyz - torch.einsum('bnkf, bnk -> bnkf', knn_normals, first_invar)

        result = torch.stack([
            first_invar,                                                    # n_j^T p_ji   
            torch.einsum('bnkf, bnkf -> bnk', second_invar, second_invar),  # ||p_ji - n_j(n_j^T p_ji)||^2
            torch.einsum('bnkf, bnkf -> bnk', normals, knn_normals)         # n_j^T n_i
            ], dim=-1)
        return result
    