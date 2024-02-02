import torch
import argparse
from fsl.data.gifti import loadGiftiMesh
import numpy as np
import open3d as o3d
import nibabel as nib
import os
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data
import torch_geometric.transforms as t

def getPosNormals(sub_ses, midthickness_folder):
    _, _, vertices_L, _ = loadGiftiMesh(f'{midthickness_folder}/{sub_ses}_left_midthickness.sym.40k_fs_LR.surf.gii')
    _, _, vertices_R, _ = loadGiftiMesh(f'{midthickness_folder}/{sub_ses}_right_midthickness.sym.40k_fs_LR.surf.gii')
    vertices = torch.cat([torch.Tensor(vertices_L[0]), torch.Tensor(vertices_R[0])])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.numpy())
    pcd.estimate_normals()

    points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    return torch.from_numpy(points), torch.from_numpy(normals)

def getStructural(sub_ses, structural_folder):
    cor_thickness_L = nib.load(f'{structural_folder}/cortical_thickness/{sub_ses}.corr_thickness.ico6.L.shape.gii').agg_data()
    cor_thickness_R = nib.load(f'{structural_folder}/cortical_thickness/{sub_ses}.corr_thickness.ico6.R.shape.gii').agg_data()
    cor_thickness = torch.cat([torch.Tensor(cor_thickness_L), torch.Tensor(cor_thickness_R)])

    curvature_L = nib.load(f'{structural_folder}/curvature/{sub_ses}.curvature.ico6.L.shape.gii').agg_data()
    curvature_R = nib.load(f'{structural_folder}/curvature/{sub_ses}.curvature.ico6.R.shape.gii').agg_data()
    curvature = torch.cat([torch.Tensor(curvature_L), torch.Tensor(curvature_R)])

    myelin_L = nib.load(f'{structural_folder}/myelin_map/{sub_ses}.myelin_map.ico6.L.shape.gii').agg_data()
    myelin_R = nib.load(f'{structural_folder}/myelin_map/{sub_ses}.myelin_map.ico6.R.shape.gii').agg_data()
    myelin = torch.cat([torch.Tensor(myelin_L), torch.Tensor(myelin_R)])

    sulcal_L = nib.load(f'{structural_folder}/sulcal_depth/{sub_ses}.sulc.ico6.L.shape.gii').agg_data()
    sulcal_R = nib.load(f'{structural_folder}/sulcal_depth/{sub_ses}.sulc.ico6.R.shape.gii').agg_data()
    sulcal = torch.cat([torch.Tensor(sulcal_L), torch.Tensor(sulcal_R)])

    return torch.stack([cor_thickness, curvature, myelin, sulcal]).T

def getDiffusion(sub_ses, diffusion_folder):
    try:
        fa_L = nib.load(f'{diffusion_folder}/{sub_ses}.fa.ico6.L.shape.gii').agg_data()
        fa_R = nib.load(f'{diffusion_folder}/{sub_ses}.fa.ico6.R.shape.gii').agg_data()
        fa = torch.cat([torch.Tensor(fa_L), torch.Tensor(fa_R)])

        md_L = nib.load(f'{diffusion_folder}/{sub_ses}.md.ico6.L.shape.gii').agg_data()
        md_R = nib.load(f'{diffusion_folder}/{sub_ses}.md.ico6.R.shape.gii').agg_data()
        md = torch.cat([torch.Tensor(md_L), torch.Tensor(md_R)])

        ICVF_L = nib.load(f'{diffusion_folder}/{sub_ses}.ICVF.ico6.L.shape.gii').agg_data()
        ICVF_R = nib.load(f'{diffusion_folder}/{sub_ses}.ICVF.ico6.R.shape.gii').agg_data()
        ICVF = torch.cat([torch.Tensor(ICVF_L), torch.Tensor(ICVF_R)])

        ISOVF_L = nib.load(f'{diffusion_folder}/{sub_ses}.ISOVF.ico6.L.shape.gii').agg_data()
        ISOVF_R = nib.load(f'{diffusion_folder}/{sub_ses}.ISOVF.ico6.R.shape.gii').agg_data()
        ISOVF = torch.cat([torch.Tensor(ISOVF_L), torch.Tensor(ISOVF_R)])

        OD_L = nib.load(f'{diffusion_folder}/{sub_ses}.OD.ico6.L.shape.gii').agg_data()
        OD_R = nib.load(f'{diffusion_folder}/{sub_ses}.OD.ico6.R.shape.gii').agg_data()
        OD = torch.cat([torch.Tensor(OD_L), torch.Tensor(OD_R)])

        return torch.stack([fa, md, ICVF, ISOVF, OD]).T

    ## Some subjects have diffusion metrics missing, return None in that case to skip them
    except:
        return None

def getLabels(sub_ses, all_labels):
    return all_labels[all_labels['sub_ses'] == sub_ses][['sub_ses', 'birth age', 'scan age', 'bayley score']]   

def main(args):
    # Make sure target folder exists
    save_dir = f'{args.save_dir}'
    try: 
        os.makedirs(save_dir)
    except:
        print(f'{save_dir} already exists, overwriting data inside it.')

    # Load all labels
    all_labels = pd.read_csv(f'{args.src_dir}/train_labels.csv')
    all_labels['sub_ses'] = all_labels['ids'] + '_' + all_labels['session'].astype(str)

    # Load data 
    structural_folder = f'{args.src_dir}/structural'
    midthickness_dir = f'{structural_folder}/midthickness'
    diffusion_folder = f'{args.src_dir}/diffusion'
    pb = tqdm(total=len(os.listdir(midthickness_dir))//2) 
    for filename in os.listdir(midthickness_dir):
        identifiers = filename.split('_')
        subject = identifiers[0].split('-')[1]
        session = identifiers[1].split('-')[1]

        # Do one pass over each subject_session
        # only for left, bc the naming is the same otherwise
        if 'left' in identifiers:
            pos, normals = getPosNormals(f'sub-{subject}_ses-{session}', midthickness_dir)
            structural_features = getStructural(f'{subject}_{session}', structural_folder)
            diffusion_features = getDiffusion(f'{subject}_{session}', diffusion_folder)

            # Deal with subjects with missing diffusion metrics
            if diffusion_features is None:
                continue
            
            # Combine all features and get labels
            features = torch.cat([structural_features, diffusion_features], dim=1)
            labels = getLabels(f'{subject}_{session}', all_labels).squeeze().to_dict()

            ## Make into torch geometric mesh and save
            mesh = Data(x=features, pos=pos, normal=normals, labels=labels)
            torch.save(mesh, f'{save_dir}/{subject}_{session}.pt')
            pb.update()
    pb.close()

    print(f'Succesfully saved files.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='./cortical_data', type=str,
        help='The directory that contains the source data.')
    parser.add_argument('--save_dir', default='./pointcloud_data', type=str,
        help='The directory to save to.')
    
    args = parser.parse_args()
    main(args)
