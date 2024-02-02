Code for the PointTransformer was taken from https://github.com/qq456cvb/Point-Transformers and adjusted by Marga Don for the purposes of this project.

## Process the data
To convert the cortical data to point clouds, first make sure to download the data. It should be structured as such: 

```
├── cortical_data
│   ├── diffusion
│   ├── structural
│   │   ├── cortical_thickness
│   │   ├── curvature
│   │   ├── midthickness
│   │   ├── myelin_map
│   │   ├── sulcal_depth
│   ├── train_labels.csv

```

To preprocess the data, run
```
python process_cortical_data.py --src_dir {SRC_DIR} --save_dir {SAVE_DIR}
```
where `SRC_DIR` is the path to the directory where you saved the data and `SAVE_DIR` is the path to the directory where you want to save the processed data. 
This step should take about 2 minutes.

## Training
Run 
```
python train.py
```
to train the model. 

The code includes Weights & Biases (wandb) logging code, either log in to your wandb account or uncomment the lines relating to wandb.