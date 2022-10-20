'''
config.py
Written by Li Jiang
Modified by Su Zhengyuan
'''

import argparse
import yaml
import os
from easydict import EasyDict as edict

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/3DUNetWithText_scannet_default.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            if isinstance(v, dict):
                v = edict(v)
            setattr(args_cfg, k, v)

    return args_cfg

cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.training_name, cfg.training_name))

TRAIN_NAME = cfg.training_name
try:
    verbose = 'verbose' in cfg.options
    dist_flag = 'distributed' in cfg.options
except Exception as e:
    print("Fallback due to", e)
    verbose = False
    dist_flag = False
    local_rank = 0
    
text_flag = cfg.has_text
pseudo_label_flag = cfg.label == 'pseudo'
subcloud_flag = cfg.label == 'subcloud'
if text_flag:
    max_seq_len = cfg.text_data.max_seq_len
    cropped_texts = cfg.text_data.cropped_texts

scale=cfg.pointcloud_data.scale  #Voxel size = 1/scale - 5cm
val_reps=cfg.pointcloud_data.val_reps # Number of test views, 1 or more
batch_size=cfg.pointcloud_data.batch_size
elastic_deformation=cfg.pointcloud_data.elastic_deformation

dimension = cfg.pointcloud_model.dimension
full_scale = cfg.pointcloud_model.full_scale #Input field size
if subcloud_flag:
    in_radius = cfg.in_radius

try:
    use_gt = cfg.use_gt
except:
    use_gt = False