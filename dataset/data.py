# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NUM_CLASSES = 20

import torch, numpy as np
import torch.utils.data
import multiprocessing as mp, time, json
import os, sys, glob
from clip import tokenize
from easydict import EasyDict as edict
import pickle

sys.path.append(os.getcwd()) # HACK: add working directory
from utils.config import *

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

train = []
val = []
train_files = glob.glob('dataset/ScanNet/train_processed/*.pth')
val_files = glob.glob('dataset/ScanNet/val_processed/*.pth')
box_path = 'ops/GeometricSelectiveSearch/gss/computed_proposal_scannet/fv'
gt_box_path = 'dataset/ScanNet/scannet_all_points'
shape_path = 'ops/GeometricSelectiveSearch/shape_det/cgal_output'

def collect_files(x:str):
    """
    [data, pseudo_label*, text*, KDTree*, scene_name]
    """
    x = x[0]
    data = torch.load(x)
    prefix = x[:-15]
    scene_name = prefix.split('/')[-1]
    box_file = os.path.join(box_path, scene_name + '_prop.npy') if not use_gt else os.path.join(gt_box_path, scene_name + '_bbox.npy')
    box = np.load(box_file)
    shape_file = os.path.join(shape_path, scene_name + '.txt')
    with open(shape_file, 'r') as f:
        shapes = f.readlines()
    shapes = [np.array([int(id) for id in shape.split(' ')[:-1]]) for shape in shapes]

    result = [data, box, shapes]
    if pseudo_label_flag:
        result.append(torch.load(os.path.join(cfg.pseudo_label_path, scene_name + cfg.pseudo_label_suffix)))
    if text_flag:
        result.append(json.load(open(prefix + '_text.json', 'r')))
    if subcloud_flag:
        result.append(pickle.load(open(prefix + '_KDTree.pkl', 'rb')))
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
        collate_fn=lambda x: (torch.load(x[0]), x[0][:-15].split('/')[-1]), num_workers=mp.cpu_count() // 4):
    val.append(x) # pc, scene_name
print('Training examples:', len(train))
print('Validation examples:', len(val))


def trainMerge(tbl):
    locs=[]
    boxes = []
    feats=[]
    labels=[]
    scene_labels = []
    scene_names = []
    batch_offsets = [0]
    has_text = []
    texts = []
    align_matrices = []
    centers = []
    rots = []
    offsets = []

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
        #TODO: debug here
        (a, center), b, c, align_mat = pc # a - coords, b - colors, c - label
        
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
        a = torch.from_numpy(a).long()

        scene_label_inds = np.unique(c).astype('int')
        scene_label_inds = scene_label_inds[scene_label_inds >= 0]
        scene_label = np.zeros(NUM_CLASSES)
        scene_label[scene_label_inds] = 1.

        if text_flag and len(text) > 0:
            has_text.append(idx)
            text = tokenize(text)
            texts.append(text)

        locs.append(torch.cat([a,torch.LongTensor(a.shape[0], 1).fill_(idx)],1))
        boxes.append(torch.cat([box[:, :6],torch.LongTensor(box.shape[0], 1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c if not pseudo_label_flag else pseudo_label)) 
        scene_labels.append(torch.from_numpy(scene_label))
        align_matrices.append(torch.from_numpy(align_mat).float())
        centers.append(torch.from_numpy(center).float())
        rots.append(torch.from_numpy(np.linalg.inv(rot)).float())
        offsets.append(torch.from_numpy(offset).float())
        if not pseudo_label_flag:
            scene_names.append(scene_name)
        batch_offsets.append(batch_offsets[-1] + np.sum(idxs))

    locs = torch.cat(locs, 0)
    boxes = torch.cat(boxes, 0)
    shapes = [torch.from_numpy(shape) for shape in shapes]
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0) # B, N
    scene_labels = torch.stack(scene_labels) # B, NumClasses
    texts = torch.stack(texts) if len(has_text) > 0 else torch.tensor(-1) # B, NumText, LenSeq
    has_text = torch.tensor(has_text).long()
    align_matrices = torch.stack(align_matrices)
    centers = torch.stack(centers)
    rots = torch.stack(rots)
    offsets = torch.stack(offsets)

    input_batch = {
            'coords': locs,
            'feature': feats,
            'batch_offsets': batch_offsets,
            'boxes': boxes, # (NumBoxes, 6+1)
            'shapes': shapes,
            'transform': [align_matrices, centers, rots, offsets]
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
    x, scene_name = x
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
    valScenes.append(scene_name)
valLabels=np.hstack(valLabels)

def valMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    scene_names = []
    scene_labels = []
    point_ids=[]

    for idx,i in enumerate(tbl):
        pc, scene_name = val[i]
        (a, _), b, c, _= pc

        m=np.eye(3)
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*np.pi
        m=np.matmul(m,[[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset

        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        a=torch.from_numpy(a).long()

        scene_label_inds = np.unique(c).astype('int')
        scene_label_inds = scene_label_inds[scene_label_inds >= 0]
        scene_label = np.zeros(NUM_CLASSES)
        scene_label[scene_label_inds] = 1

        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        scene_names.append(scene_name)
        scene_labels.append(torch.from_numpy(scene_label))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    scene_labels = torch.stack(scene_labels) # B, NumClasses
    point_ids=torch.cat(point_ids,0)

    input_batch = {
        'coords': locs,
        'feature': feats,
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
        break
    for batch in val_data_loader:
        print(batch.keys())
        break