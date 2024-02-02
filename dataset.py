import numpy as np
import os
from torch.utils.data import Dataset, random_split
import torch
from pointnet_util import farthest_point_sample, pc_normalize, index_points
import json
import random
from torch_geometric.data import Data

def getScales(root):
    pos_scales = []
    all_features = []
    
    for sub_ses in os.listdir(root):
        mesh = torch.load(f'{root}/{sub_ses}')

        # Positions
        pos = mesh.pos
        centroid = torch.mean(pos, dim=0)
        pos -= centroid
        scale = torch.max(torch.sqrt(torch.sum(abs(pos)**2,dim=-1)))
        pos_scales.append(scale)

        # Features
        all_features.append(mesh.x)

    all_features = torch.cat(all_features)
    f_scales = (torch.mean(all_features, dim=0), torch.std(all_features, dim=0))

    # Position scales
    pos_scale = max(pos_scales)

    return pos_scale, f_scales

def getCorticalSplits(root, train_size=0.8):
    assert train_size < 1
    val_test_size = (1 - train_size) / 2

    all_sub_ses = np.array(os.listdir(root))
    n_datapoints = len(all_sub_ses)
    train_cutoff = int(n_datapoints * train_size)
    val_cutoff = int(n_datapoints * (train_size + val_test_size))

    indices = np.arange(n_datapoints)
    np.random.shuffle(indices)
    train_indices = indices[:train_cutoff]
    val_indices = indices[train_cutoff:val_cutoff]
    test_indices = indices[val_cutoff:]

    assert len(train_indices) + len(val_indices) + len(test_indices) == n_datapoints

    train_data = all_sub_ses[train_indices]
    val_data = all_sub_ses[val_indices]
    test_data = all_sub_ses[test_indices]

    return train_data, val_data, test_data

class CorticalPointDataset(Dataset):
    def __init__(self, root, split, npoint, task='birth age', pos_scale=None, feature_scales=None, device='cuda'):
        self.npoint = npoint
        self.root = root
        self.split = split
        self.task = task
        self.device = device
        self.pos_scale = pos_scale
        self.feature_scales = feature_scales

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        mesh_path = f'{self.root}/{self.split[index]}'
        mesh = torch.load(mesh_path)
        pos, normals, features = mesh.pos, mesh.normal, mesh.x
        label, scan_age = mesh.labels[self.task], mesh.labels['scan age']

        ## Centering and scaling positions
        centroid = torch.mean(pos, dim=0)
        pos -= centroid
        local_scale = torch.max(torch.sqrt(torch.sum(abs(pos)**2,dim=-1)))
        pos = pos / self.pos_scale if self.pos_scale is not None else pos / local_scale

        pos = pos.to(self.device).unsqueeze(0)
        fps_index = farthest_point_sample(pos, self.npoint).cpu()
        index_ = lambda x: index_points(x.unsqueeze(0), fps_index).squeeze()
        pos, normals, features = index_(pos.cpu().squeeze()), index_(mesh.normal), index_(mesh.x)

        ## Scaling features if applicable
        if self.feature_scales is not None:
            mean, std = self.feature_scales
            features = (features - mean)/std

        return Data(x=features, pos=pos, normal=normals, y=label, scan_age=scan_age)

class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        random.seed(42)
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        train_val_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        random.shuffle(train_val_ids)
        shape_ids['train'], shape_ids['val'] = np.split(np.array(train_val_ids), [int(len(train_val_ids)*0.90)])

        assert (split == 'train' or split == 'test' or split == 'val')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    root = './pointcloud_data'
    train_data, val_data, test_data = getCorticalSplits(root)
    train_data = CorticalPointDataset(root, train_data, 1024, task='birth age')
    ldr = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    import time
    from tqdm import tqdm
    start = time.time()

    for (pos, normals, features, labels) in tqdm(ldr):
        continue
    print(round(time.time() - start, 3))