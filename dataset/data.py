# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NUM_CLASSES = 20

import torch, numpy as np
import torch.utils.data
import multiprocessing as mp, time, json
import os, sys, glob, copy
from clip import tokenize
from easydict import EasyDict as edict
import pickle

sys.path.append(os.getcwd()) # HACK: add working directory
from utils.config import *
from dataset.dataset_utils.pc_util import *
from dataset.dataset_utils.wypr_util import rotate_aligned_boxes

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

train = []
val = []
train_files = glob.glob('dataset/ScanNet/train_processed/*.pth')
val_files = glob.glob('dataset/ScanNet/val_processed/*.pth')
box_path = 'ops/GeometricSelectiveSearch/gss/computed_proposal_scannet/fv'
wypr_path = 'dataset/ScanNet/scannet_all_points'

def collect_files(x:str):
    """
    [data, box, shapes, pseudo_label*, text*, KDTree*, scene_name]
    """
    x = x[0]
    data = torch.load(x)
    prefix = x[:-15]
    scene_name = prefix.split('/')[-1]
    box_file = os.path.join(box_path, scene_name + '_prop.npy') if not use_gt else os.path.join(wypr_path, scene_name + '_bbox.npy')
    box = np.load(box_file)
    shape_file = os.path.join(wypr_path, scene_name + '_shape.npy')
    shapes = np.load(shape_file)

    result = [data, box, shapes]
    if pseudo_label_flag:
        result.append(torch.load(os.path.join(cfg.pseudo_label_path, scene_name + cfg.pseudo_label_suffix)))
    if text_flag:
        result.append(json.load(open(prefix + '_text.json', 'r')))
    if subcloud_flag:
        result.append(pickle.load(open(prefix + '_KDTree.pkl', 'rb')))
    result.append(scene_name) # scene name
    return result

def collect_val_files(x:str):
    """
    [data, box, shapes, scene_name]
    """
    x = x[0]
    data = torch.load(x)
    prefix = x[:-15]
    scene_name = prefix.split('/')[-1]
    box_file = os.path.join(box_path, scene_name + '_prop.npy') if not use_gt else os.path.join(wypr_path, scene_name + '_bbox.npy')
    box = np.load(box_file)
    shape_file = os.path.join(wypr_path, scene_name + '_shape.npy')
    shapes = np.load(shape_file)

    result = [data, box, shapes]
    result.append(scene_name) # scene name
    return result

#generate sampling anchors for subclouds
def get_anchors(points):
    n_anchors = []
    x_max = points[:, 0].max()
    x_min = points[:, 0].min()
    y_max = points[:, 1].max()
    y_min = points[:, 1].min()
    z_max = points[:, 2].max()
    z_min = points[:, 2].min()
    x_step = np.floor((x_max - x_min) / in_radius) + 1
    y_step = np.floor((y_max - y_min) / in_radius) + 1
    z_step = np.floor((z_max - z_min) / in_radius) + 1
    x_num = np.linspace(x_min, x_max, int(x_step))
    y_num = np.linspace(y_min, y_max, int(y_step))
    z_num = np.linspace(z_min, z_max, int(z_step))
    for x in x_num:
        for y in y_num:
            for z in z_num:
                n_anchors.append([x, y, z])
    return np.array(n_anchors)

for x in torch.utils.data.DataLoader(
        train_files,
        collate_fn=collect_files, num_workers=mp.cpu_count() // 4):
    if not subcloud_flag:
        train.append(x)
    else:
        a, b, c = x[0]
        box = x[1]
        shapes = x[2]
        scene_name = x[-1]
        search_tree = x[-2]
        ind = 3
        if pseudo_label_flag:
            pseudo_label = x[ind]
            ind += 1
        assert not text_flag
        assert ind == len(x) - 2

        anchors = get_anchors(a) # NAnchors, 3
        noise = np.random.normal(scale=in_radius/10, size=anchors.shape)
        anchors = anchors + noise.astype(anchors.dtype)
        inds = search_tree.query_radius(anchors, in_radius)
        for ind in inds:
            if ind.shape[0] < 1000:
                continue
            if pseudo_label_flag:
                train.append((
                    (a[ind], b[ind], c[ind]), 
                    box,
                    shapes,
                    pseudo_label[ind],
                    scene_name
                ))
            else:
                train.append((
                    (a[ind], b[ind], c[ind]),
                    box,
                    shapes,
                    scene_name
                ))

for x in torch.utils.data.DataLoader(
        val_files,
        collate_fn=collect_val_files, num_workers=mp.cpu_count() // 4):
    val.append(x) # pc, scene_name
print('Training examples:', len(train))
print('Validation examples:', len(val))

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
MAX_NUM_PROP = 1000

def augment_input_wypr(coords, color, normals):
    point_cloud = coords
    pcl_color = (color - MEAN_COLOR_RGB) / 256.0
    
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1) 

    if normals is not None:
        point_cloud = np.concatenate([point_cloud, normals], 1) 

    return point_cloud, pcl_color

def sparseconvnet_augmentation_train(a, b, c, pseudo_label):
    m=np.eye(3)+np.random.randn(3,3)*0.1
    m[0][0]*=np.random.randint(0,2)*2-1
    m*=scale
    theta=np.random.rand()*2*np.pi
    rot=np.matmul(m,[[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    a=np.matmul(a,rot)
    # if elastic_deformation:
    #     a=elastic(a,6*scale//50,40*scale/50) # 16
    #     a=elastic(a,20*scale//50,160*scale/50) # 64
    m = a.min(0)
    M = a.max(0)
    length=M-m
    offset=-m + np.clip(full_scale-length-0.001,0,None) * np.random.rand(3) + np.clip(full_scale-length+0.001,None,0) * np.random.rand(3)
    a+=offset

    idxs = (a.min(1)>=0)*(a.max(1)<full_scale) # box in [0, full_scale]^3 Actual size is full_scale / scale = 200m
    a = a[idxs]
    b = b[idxs]
    c = c[idxs]
    if pseudo_label_flag:
        pseudo_label = pseudo_label[idxs]
    return a, b, c, pseudo_label

def wypr_augmentation(a, b, c, normals, props, shape_labels, augment=False):
    point_cloud, pcl_color = augment_input_wypr(a, b, normals)
    semantic_labels = c
    
    if props.shape[0] > MAX_NUM_PROP:
        choices = np.random.choice(props.shape[0], MAX_NUM_PROP, replace=False)
        props = props[choices]
    
    # proposals
    target_props = np.zeros((MAX_NUM_PROP, 6))
    target_props_mask = np.zeros((MAX_NUM_PROP))    
    if props is not None:
        target_props_mask[:props.shape[0]] = 1
        target_props[:props.shape[0],:] = copy.deepcopy(props[:,:6])
        
    # ------------------------------- SAMPLING ------------------------------       
    if cfg.pointcloud_data.num_points_sampled > 0:  
        point_cloud, choices = random_sampling(point_cloud,
            cfg.pointcloud_data.num_points_sampled, return_choices=True)     
        pcl_color = pcl_color[choices]
        semantic_labels = semantic_labels[choices]
        if shape_labels is not None:
            shape_labels = shape_labels[choices]

    # ------------------------------- DATA AUGMENTATION ------------------------------      
    if augment:
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:,0] = -1 * point_cloud[:,0] 
            target_props[:,0] = -1 * target_props[:,0]      
            
        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:,1] = -1 * point_cloud[:,1]
            target_props[:,1] = -1 * target_props[:,1] 
        
        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
        rot_mat = rotz(rot_angle)
        point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
        target_props = rotate_aligned_boxes(target_props, rot_mat)

     # ------------- DATA AUGMENTATION for consistency loss -----------------------      
        point_cloud_aug = copy.deepcopy(point_cloud)  
        target_props_aug = copy.deepcopy(target_props)  
        # Flipping along the YZ plane
        if np.random.random() > 0.8:
            point_cloud[:,0] = -1 * point_cloud[:,0]       
            target_props_aug[:,0] = -1 * target_props_aug[:,0]           
        # Flipping along the XZ plane
        if np.random.random() > 0.8:
            point_cloud[:,1] = -1 * point_cloud[:,1]
            target_props_aug[:,1] = -1 * target_props_aug[:,1]  
        # roatation
        rot_angle = (np.random.random()*np.pi/6)
        rot_mat = rotz(rot_angle)
        point_cloud_aug[:,0:3] = np.dot(point_cloud_aug[:,0:3], np.transpose(rot_mat))
        target_props_aug = rotate_aligned_boxes(target_props_aug, rot_mat)
        
    # scale
    min_s = 0.8;  max_s = 2 - min_s
    # scale = np.random.rand(point_cloud.shape[0]) * (max_s - min_s) + min_s
    # point_cloud_aug[:, :3] *= scale.reshape(-1, 1)
    scale = np.random.rand() * (max_s - min_s) + min_s
    point_cloud_aug[:, :3] *= scale
    target_props_aug *= scale
    # jittering
    jitter_min = 0.95; jitter_max = 2 - jitter_min
    jitter_scale = np.random.rand(point_cloud.shape[0]) * (jitter_max - jitter_min) + jitter_min
    point_cloud_aug[:, :3] *= jitter_scale.reshape(-1, 1)
    # # dropout  
    # num_aug_points_sampled = int(0.9 * point_cloud_aug.shape[0])
    # point_cloud_aug, choices_aug = cfg.random_sampling(point_cloud, num_aug_points_sampled, return_choices=True)  
    return point_cloud, point_cloud_aug, pcl_color, semantic_labels, shape_labels, target_props, target_props_aug

def trainMerge(tbl):
    locs=[]
    locs_aug=[]
    boxes = []
    boxes_aug = []
    shp=[]
    feats=[]
    labels=[]
    scene_labels = []
    scene_names = []
    batch_offsets = [0]
    batch_offsets_aug = [0]
    has_text = []
    texts = []

    for idx,i in enumerate(tbl):
        data = train[i]
        pc = data[0]
        box = torch.from_numpy(data[1])
        shapes = data[2]
        # print("Read", box.shape[0], "boxes.")
        # if box.shape[0] > 64: box = box[:64]
        # print("Keep", box.shape[0], "boxes.")
        scene_name = data[-1]
        ind = 3
        if pseudo_label_flag:
            pseudo_label = data[ind]
            ind += 1
        if text_flag:
            text = data[ind]
        assert ind == len(data) - 1, f"{ind} {len(data)}"
        a, (b, normal), c = pc # a - coords, b - colors, c - label
        
        a, a_aug, b, c, shape_labels, prop, prop_aug = wypr_augmentation(a, b, c, normal, box, shapes, augment=True)
        
        a = torch.from_numpy(a)
        a_aug = torch.from_numpy(a_aug)
        prop = torch.from_numpy(prop)
        prop_aug = torch.from_numpy(prop_aug)

        scene_label_inds = np.unique(c).astype('int')
        scene_label_inds = scene_label_inds[scene_label_inds >= 0]
        scene_label = np.zeros(NUM_CLASSES)
        scene_label[scene_label_inds] = 1.

        if text_flag and len(text) > 0:
            has_text.append(idx)
            text = tokenize(text)
            texts.append(text)

        locs.append(torch.cat([a,torch.LongTensor(a.shape[0], 1).fill_(idx)],1))
        locs_aug.append(torch.cat([a_aug,torch.LongTensor(a_aug.shape[0], 1).fill_(idx)],1))
        boxes.append(torch.cat([prop[:, :6],torch.LongTensor(prop.shape[0], 1).fill_(idx)],1))
        boxes_aug.append(torch.cat([prop_aug[:, :6],torch.LongTensor(prop_aug.shape[0], 1).fill_(idx)],1))
        shp.append(torch.from_numpy(shape_labels))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c if not pseudo_label_flag else pseudo_label)) 
        scene_labels.append(torch.from_numpy(scene_label))
        if not pseudo_label_flag:
            scene_names.append(scene_name)
        batch_offsets.append(batch_offsets[-1] + a.shape[0])
        batch_offsets_aug.append(batch_offsets[-1] + a_aug.shape[0])

    locs = torch.cat(locs, 0)
    locs_aug = torch.cat(locs_aug, 0)
    boxes = torch.cat(boxes, 0)
    boxes_aug = torch.cat(boxes_aug, 0)
    shp = torch.cat(shp, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0) # B, N
    scene_labels = torch.stack(scene_labels) # B, NumClasses
    texts = torch.stack(texts) if len(has_text) > 0 else torch.tensor(-1) # B, NumText, LenSeq
    has_text = torch.tensor(has_text).long()

    input_batch = {
            'coords': locs,
            'coords_aug': locs_aug,
            'feature': feats,
            'batch_offsets': batch_offsets,
            'batch_offsets_aug': batch_offsets_aug,
            'boxes': boxes, # (NumBoxes, 6)
            'boxes_aug': boxes_aug, # (NumBoxes, 6)
            'shapes': shp
            }

    return edict({
        'x': edict(input_batch), 
        'y_orig': labels.long(), 
        'y': scene_labels, 
        'text': [texts, has_text],
        'id': tbl,
        'scene_names': scene_names
        })
train_data_loader = torch.utils.data.DataLoader(
    list(range(len(train))),
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=4, 
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

valOffsets=[0]
valLabels=[]
valScenes=[]
for idx,x in enumerate(val):
    x, box, shape, scene_name = x
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
    valScenes.append(scene_name)
valLabels=np.hstack(valLabels)

def valMerge(tbl):
    locs=[]
    boxes=[]
    shapes=[]
    feats=[]
    labels=[]
    scene_names = []
    scene_labels = []
    point_ids=[]

    for idx,i in enumerate(tbl):
        pc, box, shape, scene_name = val[i]
        a, (b, normal), c = pc

        a, a_aug, b, c, shape_labels, prop, prop_aug = wypr_augmentation(a, b, c, normal, box, shape, augment=True)
        
        a=torch.from_numpy(a).long()
        prop = torch.from_numpy(prop)

        scene_label_inds = np.unique(c).astype('int')
        scene_label_inds = scene_label_inds[scene_label_inds >= 0]
        scene_label = np.zeros(NUM_CLASSES)
        scene_label[scene_label_inds] = 1

        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        boxes.append(torch.cat([prop,torch.LongTensor(prop.shape[0],1).fill_(idx)],1))
        
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        shapes.append(torch.from_numpy(shape_labels))
        scene_names.append(scene_name)
        scene_labels.append(torch.from_numpy(scene_label))
        point_ids.append(torch.from_numpy(np.arange(a.shape[0])+valOffsets[i]))
    locs=torch.cat(locs,0)
    boxes=torch.cat(boxes,0)
    feats=torch.cat(feats,0)
    shapes=torch.cat(shapes,0)
    labels=torch.cat(labels,0)
    scene_labels = torch.stack(scene_labels) # B, NumClasses
    point_ids=torch.cat(point_ids,0)

    input_batch = {
        'coords': locs,
        'feature': feats,
        'boxes': boxes,
        'shapes': shapes
        }

    return {
        'x': edict(input_batch), 
        'y_orig': labels.long(), 
        'y': scene_labels.long(), 
        'id': tbl,
        'scene_name': scene_names, 
        'point_ids': point_ids}
        
val_data_loader = torch.utils.data.DataLoader(
    list(range(len(val))),
    batch_size=batch_size,
    collate_fn=valMerge,
    num_workers=4,
    shuffle=False,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

if __name__ == '__main__':
    for batch in train_data_loader:
        print(batch.keys())
        for k, v in batch['x'].items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())
            else:
                print(k, v)
        break
    for batch in val_data_loader:
        print(batch.keys())
        for k, v in batch['x'].items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())
            else:
                print(k, v)
        break