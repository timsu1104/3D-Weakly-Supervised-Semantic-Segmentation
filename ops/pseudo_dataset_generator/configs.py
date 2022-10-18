import os.path as osp
import argparse
import yaml
import os
from easydict import EasyDict as edict

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/cabinet_with_white_bg.yaml', help='path to config file')
    parser.add_argument('--device_num', type=int, default=1, help='index of cuda device')
    parser.add_argument('--total_device', type=int, default=1, help='total number of cuda devices')

    ### pretrain
    # parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    # args_cfg, args_device_num, args_total_device = parser.parse_args()
    args_cfg = parser.parse_args()
    print('type: ', type(args_cfg))
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            if isinstance(v, dict):
                v = edict(v)
            setattr(args_cfg, k, v)

    #return args_cfg.config, args_cfg.device_num, args_cfg.total_device
    return args_cfg

# cfg, device_num, total_device = get_parser()
cfg = get_parser()
# setattr(cfg, 'exp_path', os.path.join('exp', cfg.training_name, cfg.training_name))

# TRAIN_NAME = cfg.training_name
# try:
#     verbose = 'verbose' in cfg.options
#     dist_flag = 'distributed' in cfg.options
# except Exception as e:
#     print("Fallback due to", e)
#     verbose = False
#     dist_flag = False
    
# text_flag = cfg.has_text
# pseudo_label_flag = cfg.label == 'pseudo'
# subcloud_flag = cfg.label == 'subcloud'
# if text_flag:
#     max_seq_len = cfg.text_data.max_seq_len
#     cropped_texts = cfg.text_data.cropped_texts

# scale=cfg.pointcloud_data.scale  #Voxel size = 1/scale - 5cm
# val_reps=cfg.pointcloud_data.val_reps # Number of test views, 1 or more
# batch_size=cfg.pointcloud_data.batch_size
# elastic_deformation=cfg.pointcloud_data.elastic_deformation

# dimension = cfg.pointcloud_model.dimension
# full_scale = cfg.pointcloud_model.full_scale #Input field size
# if subcloud_flag:
#     in_radius = cfg.in_radius

###############################
# Modify this part
# self.folder = "../../dataset/pseudo_images" # where you save your data
# cfg.folder = '/data/zhuhe/3DUNetWithText/datasets/pseudo_dataset'
# self.cls = "sofa_picture_with_white_bg"
# self.text_format = "sofa"

# self.blur_radius = 1
# self.blur_samples = 20
# ###############################

# self.Output_path = osp.join(self.folder, self.cls)

# class PseudoDatasetConfig:
#     def __init__(self) -> None:

#         ###############################
#         # Modify this part
#         # self.folder = "../../dataset/pseudo_images" # where you save your data
#         self.folder = '/data/zhuhe/3DUNetWithText/datasets/pseudo_dataset'
#         self.cls = "sofa_picture_with_white_bg"
#         self.text_format = "sofa"

#         self.blur_radius = 1
#         self.blur_samples = 20
#         ###############################

#         self.Output_path = osp.join(self.folder, self.cls)

# cfg = PseudoDatasetConfig()