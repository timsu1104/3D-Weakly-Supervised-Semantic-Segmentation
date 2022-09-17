# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NUM_CLASSES = 20

import torch, numpy as np
import torch.utils.data, scipy.ndimage
import multiprocessing as mp, time, json
import os, sys, glob

from .dataset_utils import text_transform

sys.path.append(os.getcwd()) # HACK: add working directory
from utils.config import cfg

scale=cfg.pointcloud_data.scale  #Voxel size = 1/scale - 5cm
val_reps=cfg.pointcloud_data.val_reps # Number of test views, 1 or more
batch_size=cfg.pointcloud_data.batch_size
elastic_deformation=cfg.pointcloud_data.elastic_deformation

max_seq_len = cfg.text_data.max_seq_len
cropped_texts = cfg.text_data.cropped_texts

tokenize = text_transform(max_seq_len, cropped_texts)

dimension=3
full_scale=4096 #Input field size

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

train = []
val = []
for x in torch.utils.data.DataLoader(
        glob.glob('dataset/ScanNet/train_processed/*.pth'),
        collate_fn=lambda x: (torch.load(x[0]), json.load(open(x[0][:-15] + '_text.json', 'r'))), num_workers=mp.cpu_count()):
    train.append(x)
for x in torch.utils.data.DataLoader(
        glob.glob('dataset/ScanNet/val_processed/*.pth'),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    val.append(x)
print('Training examples:', len(train))
print('Validation examples:', len(val))

#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag

def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    scene_labels = []
    batch_offsets = [0]
    has_text = []
    texts = []

    for idx,i in enumerate(tbl):
        pc, text = train[i]
        a, b, c = pc # a - coords, b - colors, c - label

        m=np.eye(3)+np.random.randn(3,3)*0.1
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*np.pi
        m=np.matmul(m,[[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)
        if elastic_deformation:
            a=elastic(a,6*scale//50,40*scale/50) # 16
            a=elastic(a,20*scale//50,160*scale/50) # 64
        m = a.min(0)
        M = a.max(0)
        length=M-m
        offset=-m + np.clip(full_scale-length-0.001,0,None) * np.random.rand(3) + np.clip(full_scale-length+0.001,None,0) * np.random.rand(3)
        a+=offset

        idxs = (a.min(1)>=0)*(a.max(1)<full_scale) # box in [0, full_scale]^3 Actual size is full_scale / scale = 200m
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]
        a = torch.from_numpy(a).long()

        scene_label_inds = np.unique(c).astype('int')
        scene_label_inds = scene_label_inds[scene_label_inds >= 0]
        scene_label = np.zeros(NUM_CLASSES)
        scene_label[scene_label_inds] = 1.

        if len(text) > 0:
            has_text.append(idx)
            text = tokenize(text)
            texts.append(text)

        locs.append(torch.cat([a,torch.LongTensor(a.shape[0], 1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c))
        scene_labels.append(torch.from_numpy(scene_label))
        batch_offsets.append(batch_offsets[-1] + np.sum(idxs))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0) # B, N
    scene_labels = torch.stack(scene_labels) # B, NumClasses
    texts = torch.stack(texts) if len(has_text) > 0 else torch.tensor(-1) # B, NumText, LenSeq
    has_text = torch.tensor(has_text).long()
    return {
        'x': [locs, feats, batch_offsets], 
        'y_orig': labels.long(), 
        'y': scene_labels, 
        'text': [texts, has_text],
        'id': tbl
        }
train_data_loader = torch.utils.data.DataLoader(
    list(range(len(train))),
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=20, 
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

valOffsets=[0]
valLabels=[]
for idx,x in enumerate(val):
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
valLabels=np.hstack(valLabels)

def valMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    scene_labels = []
    point_ids=[]

    for idx,i in enumerate(tbl):
        a,b,c=val[i]

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
        scene_labels.append(torch.from_numpy(scene_label))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    scene_labels = torch.stack(scene_labels) # B, NumClasses
    point_ids=torch.cat(point_ids,0)
    return {'x': [locs,feats], 'y_orig': labels.long(), 'y': scene_labels.long(), 'id': tbl, 'point_ids': point_ids}
val_data_loader = torch.utils.data.DataLoader(
    list(range(len(val))),
    batch_size=batch_size,
    collate_fn=valMerge,
    num_workers=20,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
