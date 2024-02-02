import argparse
import torch
import torch.nn as nn
from model import TransitionDown
from transformer import TransformerBlock
from torch_geometric.datasets import ModelNet
from torch_geometric import transforms as T
import copy
from e3nn.o3 import rand_matrix
import torch_geometric as tg
from dataset import CorticalPointDataset, getCorticalSplits

class BackBone_Test(nn.Module):
    def __init__(self, pos_enc, d_points):
        super().__init__()

        npoints = 1024
        nblocks = 4
        nneighbor = 16
        nneighbor_bias = 16
        transformer_dim = 512

        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor_bias, npoints, pos_enc)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            npoints = npoints // 4
            self.transition_downs.append(TransitionDown(int(npoints), nneighbor, [channel // 2, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor_bias, npoints, pos_enc))
        self.nblocks = nblocks

    def forward(self, xyz1, normals1, init_features1, xyz2, normals2, init_features2):
        points1 = self.transformer1(xyz1, normals1, self.fc1(init_features1.cuda()))[0]
        points2 = self.transformer1(xyz2, normals2, self.fc1(init_features2.cuda()))[0]

        xyz_normals_feats1 = [(xyz1, normals1, points1)]
        xyz_normals_feats2 = [(xyz2, normals2, points2)]
        fps_indices = []
        for i in range(self.nblocks):
            xyz1, normals1, points1, fps_index1 = self.transition_downs[i](xyz1, normals1, points1, fps_index=None)
            xyz2, normals2, points2, fps_index2 = self.transition_downs[i](xyz2, normals2, points2, fps_index1)

            points1 = self.transformers[i](xyz1, normals1, points1)[0]
            points2 = self.transformers[i](xyz2, normals2, points2)[0]

            xyz_normals_feats1.append((xyz1, normals1, points1))
            xyz_normals_feats2.append((xyz2, normals2, points2))

            fps_indices.append((fps_index1, fps_index2))
        return points1, xyz_normals_feats1, points2, xyz_normals_feats2, fps_indices

def rotate(matrix, to_rotate):
    return torch.einsum('ji, in -> jn', matrix, to_rotate)

def prepare(x):
    return x.unsqueeze(0).float()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelnet', action='store_true', default=False)
    parser.add_argument('--root')
    args = parser.parse_args()

    tg.seed_everything(42)

    if args.modelnet:
        transforms = T.Compose([
            T.NormalizeScale(),
            T.Center(),
            T.SamplePoints(1024, include_normals=True)])
        data = ModelNet('./modelnet', name='40', transform=transforms, train=False)
    else:
        train_split, _, _ = getCorticalSplits(args.root)
        data = CorticalPointDataset(args.root, split=train_split, npoint=1024)
    
    R = rand_matrix().cuda()
    R = R if args.modelnet else R.double()
    x1 = data[0].cuda()
    x2 = copy.deepcopy(x1).cuda()

    # Rotate x2
    x2.pos, x2.normal = rotate(R, x2.pos.T).T, rotate(R, x2.normal.T).T

    bb = BackBone_Test(pos_enc='R3', d_points=3 if args.modelnet else 9).cuda()
    x1.x = torch.ones(x1.pos.shape) if args.modelnet else x1.x
    x2.x = torch.ones(x2.pos.shape) if args.modelnet else x2.x
    p1, feats1, p2, feats2, fps_idxs = bb(prepare(x1.pos), prepare(x1.normal), prepare(x1.x), prepare(x2.pos), prepare(x2.normal), prepare(x2.x))

    last1, last2 = feats1[-1][0].squeeze(), feats2[-1][0].squeeze()

    fps_checks = []
    xyz_max, normals_max, points_max = [], [], []
    for i in range(len(fps_idxs)):
        fps_checks.append(torch.allclose(*fps_idxs[i]))
        
        r_xyz1, r_norm1 = rotate(R.float(), feats1[i][0].T.squeeze()).T, rotate(R.float(), feats1[i][1].T.squeeze()).T

        xyz_max.append(round(torch.max(torch.abs(r_xyz1 - feats2[i][0])).item(), 5))
        normals_max.append(round(torch.max(torch.abs(r_norm1 - feats2[i][1])).item(), 5))

    print('If invariant to rotation: all should be True / close to 0:')
    print(f'fps indices check: {fps_checks}')
    print(f'xyz check: {xyz_max}')
    print(f'normals check: {normals_max}')