import numpy as np
import torch
import logging
from tqdm import tqdm
import wandb
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import random
import os
import argparse
from model import PointTransformerCls
from datetime import date

import argparse
from dataset import getCorticalSplits, CorticalPointDataset, getScales
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from model import PointTransformerReg
from tqdm import tqdm
import wandb
import copy
import os
import logging
from datetime import date

def test(args, model, loader, device):
    model.eval()
    metric, std = 0, 0
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            if args.modelnet:
                batch.x = batch.pos if args.features == 'xyz' else torch.ones(batch.pos.shape)

            batch_size, feature_channels = len(batch.y), batch.x.shape[-1]
            process = lambda x, c: x.reshape(batch_size, args.num_point, c).to(device).float()
            pos, normals, features = process(batch.pos, 3), process(batch.normal, 3), process(batch.x, feature_channels)
            target = batch.y.to(device)
            scan_age = batch.scan_age.to(device) if args.scan_age else None

            if args.xyz_features:
                features = torch.cat([pos, features], dim=-1)
         
            pred = model(pos, normals, features, scan_age).squeeze()

            # Metrics
            if args.modelnet:
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                metric += correct.item() / batch_size
            else:
                abs_error = torch.abs(pred - target)
                metric += torch.mean(abs_error)   
                if batch_size != 1:
                    std += torch.std(abs_error)

    metric /= len(loader)
    std /= len(loader)
    return metric, std

def getCorticalDataset(args, root, device):
    train_split, val_split, test_split = getCorticalSplits(root)
    pos_scale, feat_scales = getScales(root)
    pos_scale = pos_scale if args.scale_pos_relative else None
    feat_scales = feat_scales if args.scale_features else None

    print(f'Using scales: pos {pos_scale}, feat {feat_scales}')
    train_data = CorticalPointDataset(root, train_split, args.num_point, args.task, pos_scale, feat_scales, device)
    val_data = CorticalPointDataset(root, val_split, args.num_point, args.task, pos_scale, feat_scales, device)
    test_data = CorticalPointDataset(root, test_split, args.num_point, args.task, pos_scale, feat_scales, device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def getModelNetDataset(args):
    # Set transforms
    train_transforms = [T.NormalizeScale(),
                        T.Center(),
                        T.RandomJitter(0.1),
                        T.RandomScale((0.8, 1.25)),
                        T.SamplePoints(args.num_point, include_normals=True)]
    
    rotation_transforms = [T.RandomRotate((-180, 180), axis=i) for i in [0,1,2]]
    test_transforms = [ T.NormalizeScale(),
                        T.Center(),
                        T.SamplePoints(args.num_point, include_normals=True)]

    logging.info(f'Using train transforms: {train_transforms}')
    train_transforms, stationary_transforms  = T.Compose(train_transforms), T.Compose(test_transforms)
    rotated_transforms = T.Compose(rotation_transforms + test_transforms)

    # Load data
    train_val_data = ModelNet('./modelnet', name='40', transform=train_transforms, train=True)
    train_val_data = train_val_data.shuffle()
    val_cutoff = int(args.val_size* len(train_val_data))
    train_data = train_val_data[:val_cutoff]
    val_data = train_val_data[val_cutoff:]

    trainDataLoader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valDataLoader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Load rotated and non-rotated test data
    stationary_test_data = ModelNet('./modelnet', name='40', transform=stationary_transforms, train=False)
    stationary_testDataLoader =  DataLoader(stationary_test_data, batch_size=args.batch_size, shuffle=False)
    rotated_test_data = ModelNet('./modelnet', name='40', transform=rotated_transforms, train=False)
    rotated_testDataLoader = DataLoader(rotated_test_data, batch_size=args.batch_size, shuffle=False)

    return trainDataLoader, valDataLoader, stationary_testDataLoader, rotated_testDataLoader

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(args):
    set_seed(args.seed)

    '''Setup Wandb and local logging'''
    task_name = args.task.replace(' ', '-')
    wandb.init(project=f'test', config=args)
    logging_folder = f'./log/{task_name}/{wandb.run.name}_{date.today()}'
    try:
        os.mkdir(logging_folder)
    except:
        print(f'Folder {logging_folder} already exists, overwriting contents.')

    '''Check hyperparameters'''
    logging.basicConfig(filename=f'{logging_folder}/train_log.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    print(f'Args: {args}')

    '''Load dataset and model'''
    logging.info('Load dataset ...')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.modelnet:
        train_loader, val_loader, test_loader, rot_test_loader = getModelNetDataset(args)
        args.num_class = 40
        args.input_dim = 3

        model = PointTransformerCls(args).to(device)
    else:
        root = './pointcloud_data_new'
        train_loader, val_loader, test_loader = getCorticalDataset(args, root, device)
        args.input_dim = 9 if args.diffusion else 4 ## 5 diffusion features + 4 structural features
        if args.xyz_features:
            args.input_dim += 3

        model = PointTransformerReg(args).to(device)

    '''Setup training'''
    criterion = nn.CrossEntropyLoss() if args.modelnet else nn.MSELoss()
    best_metric = 0 if args.modelnet else float('Inf')
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    logging.info('Start training')
    for epoch in range(args.epoch):
        logging.info(f'Epoch {epoch+1} ({epoch+1}/{args.epoch}):')
        '''Training'''
        model.train()
        avg_loss, train_metric, train_std = 0, 0, 0
        for batch_id, batch in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            if args.modelnet:
                batch.x = batch.pos if args.features == 'xyz' else torch.ones(batch.pos.shape)
            batch.y = batch.y.long() if args.modelnet else batch.y.float()

            batch_size, feature_channels = len(batch.y), batch.x.shape[-1]
            process = lambda x, c: x.reshape(batch_size, args.num_point, c).to(device).float()
            pos, normals, features = process(batch.pos, 3), process(batch.normal, 3), process(batch.x, feature_channels)
            target = batch.y.to(device).squeeze()
            scan_age = batch.scan_age.to(device) if args.scan_age else None

            if args.xyz_features:
                features = torch.cat([pos, features], dim=-1)

            optimizer.zero_grad()
            pred = model(pos, normals, features, scan_age).squeeze()
            loss = criterion(pred, target)  

            # Metrics
            avg_loss += loss.item() / batch_size
            if args.modelnet:
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                train_metric += correct.item() / batch_size
            else:        
                abs_error = torch.abs(pred - target)
                train_metric += torch.mean(abs_error)   
                if batch_size != 1:
                    train_std += torch.std(abs_error)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            loss.backward()
            optimizer.step()
        scheduler.step()

        '''Validation'''
        val_metric, val_std = test(args, model, val_loader, device)
        better = (args.modelnet and val_metric > best_metric) or (not args.modelnet and val_metric < best_metric)
        if better:
            logging.info('Save model...')
            best_metric = val_metric
            best_model = copy.deepcopy(model)
            savepath = f'{logging_folder}/best_model.pth'
            state = {
                    'epoch': epoch,
                    'metric': best_metric,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(state, savepath)

        # Log to wandb
        wandb.log({'train_metric': train_metric / len(train_loader), 
                   'train_loss': avg_loss / len(train_loader), 
                   'train_std': train_std / len(train_loader),
                   'val_metric': val_metric,
                   'val_std': val_std,
                   'best_metric': best_metric
                })

        '''Testing'''
        test_metric, test_std = test(args, best_model, test_loader, device)
        wandb.log({'test_metric': test_metric, 'test_std': test_std})
        logging.info('End of training...')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General training
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--task', choices=['birth age', 'bayley score'], default='birth age')
    parser.add_argument('--modelnet', action='store_true', default=False)
    parser.add_argument('--batch_size', default=32, type=int)  
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)  
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--gradient_clip', default=10, type=int)

    # Transformer
    parser.add_argument('--pos_enc', default='R3S2', choices=['R3S2', 'R3', 'none', 'rel_dis']) 
    parser.add_argument('--num_point', default=1024, type=int)
    parser.add_argument('--nblocks', default=4, type=int)
    parser.add_argument('--nneighbor', default=16, type=int, help='how many neighbors to use in downsampling operation')
    parser.add_argument('--nneighbor_bias', default=16, type=int, help='how many neighbors to consider for computing the bias')
    parser.add_argument('--transformer_dim', default=512, type=int)

    # ModelNet
    parser.add_argument('--features', choices=['xyz', 'ones'], default='ones', type=str)
    parser.add_argument('--val_size', default=0.85, type=float)   

    # Cortical
    parser.add_argument('--xyz_features', action='store_true', default=False)
    parser.add_argument('--diffusion', action='store_true', default=False, help='Enable to use the diffusion features')
    parser.add_argument('--scan_age', action='store_true', default=False, help='Enable to use the scan age in the prediction head')
    parser.add_argument('--scale_pos_relative', action='store_true', default=False,
        help='If enabled, normalizes cortical surfaces wrt the largest volume in dataset. Else, normalizes all cortical data individually.')
    parser.add_argument('--scale_features', action='store_true', default=False,
        help='If enabled, scales all features to N(0, 1). Otherwise keeps their raw values.')

    args = parser.parse_args()
    main(args)