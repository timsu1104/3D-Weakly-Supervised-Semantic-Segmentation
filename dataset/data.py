# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# num of semantic classes
NUM_CLASSES = 20

import torch, numpy as np
import torch.utils.data, scipy.ndimage
import multiprocessing as mp, time, json
import os, sys, glob
from clip import tokenize

# from .dataset_utils import text_transform

sys.path.append(os.getcwd()) # HACK: add working directory
from utils.config import cfg

# default setting -- val_reps = 1, batch_size = 8, elastic_deformation = False
scale = cfg.pointcloud_data.scale  #Voxel size = 1/scale - 5cm
val_reps = cfg.pointcloud_data.val_reps # Number of test views, 1 or more
batch_size = cfg.pointcloud_data.batch_size
elastic_deformation = cfg.pointcloud_data.elastic_deformation

text_flag = cfg.has_text
# if cfg.label == "scene_level" => pseudo_label_flag = 1
pseudo_label_flag = cfg.label == 'scene_level'
if text_flag:
    # default setting for clip model -- 120
    max_seq_len = cfg.text_data.max_seq_len
    #cropped_texts = cfg.text_data.cropped_texts

    # tokenize = text_transform(max_seq_len, cropped_texts)

dimension=3
full_scale=4096 #Input field size

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

train = []
val = []
train_files = glob.glob('dataset/ScanNet/train_processed/*.pth')
val_files = glob.glob('dataset/ScanNet/val_processed/*.pth')

def printDataLoader(x):
    print("x:", x)
    print("load x[0]:", torch.load(x[0]))
    print(" load json", json.load(open(x[0][:-15] + '_text.json', 'r')))
    print(x[0].split('/')[-1][:-15])
    return (torch.load(x[0]), json.load(open(x[0][:-15] + '_text.json', 'r')), x[0].split('/')[-1][:-15])


if pseudo_label_flag:
    if text_flag:
        for x in torch.utils.data.DataLoader(
                train_files,
                collate_fn=lambda x: (torch.load(x[0]), json.load(open(x[0][:-15] + '_text.json', 'r')), x[0].split('/')[-1][:-15]), 
                # collate_fn=printDataLoader,
                num_workers=mp.cpu_count() // 4):
            """ 
                element in train -- 
                1. point cloud
                2. texts in json
                3. scene name
            """
            train.append(x)
    else:
        for x in torch.utils.data.DataLoader(
                train_files,
                # if there's no text, json won't be loaded
                collate_fn=lambda x: (torch.load(x[0]), x[0].split('/')[-1][:-15]),
                num_workers=mp.cpu_count() // 4):
            """ 
                element in train -- 
                1. point cloud
                2. scene name
            """
            train.append(x)
else:
    if text_flag:
        for x in torch.utils.data.DataLoader(
                train_files,
                # ???
                collate_fn=lambda x: (torch.load(x[0]), torch.load(os.path.join(cfg.pseudo_label_path, x[0].split('/')[-1][:-15] + cfg.pseudo_label_suffix)), json.load(open(x[0][:-15] + '_text.json', 'r'))),
                num_workers=mp.cpu_count() // 4):
            train.append(x)
    else:
        for x in torch.utils.data.DataLoader(
                train_files,
                collate_fn=lambda x: (torch.load(x[0]), torch.load(os.path.join(cfg.pseudo_label_path, x[0].split('/')[-1][:-15] + cfg.pseudo_label_suffix))),
                num_workers=mp.cpu_count() // 4):
            train.append(x)

for x in torch.utils.data.DataLoader(
        val_files,
        # validation -- only load point cloud
        collate_fn=lambda x: torch.load(x[0]), 
        num_workers=mp.cpu_count() // 4):
    val.append(x)
print('Training examples:', len(train))
print('Validation examples:', len(val))

#Elastic distortion
# point cloud pre-processing
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

# collate_fn of the final data_loader
def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    scene_labels = []
    scene_names = []
    # num of points in each pc(each scene)
    batch_offsets = [0]
    # record scenes with texts
    has_text = []
    texts = []

    for idx,i in enumerate(tbl):
        # print(idx, i)
        # due to shuffling, idx is not always equivalent to i
        if text_flag:
            if pseudo_label_flag:
                pc, text, scene_name = train[i]
            else:
                pc, pseudo_label, text = train[i]
        else:
            if pseudo_label_flag:
                pc, scene_name = train[i]
            else:
                pc, pseudo_label = train[i]
            # if text_flag = 0 => set text as an empty list
            text = []
        a, b, c = pc # a -> coords, b -> colors, c -> label

        # np.eye -- identity matrix
        m=np.eye(3)+np.random.randn(3,3)*0.1
        # print(np.eye(3))
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
        # coordnate offset
        offset=-m + np.clip(full_scale-length-0.001,0,None) * np.random.rand(3) + np.clip(full_scale-length+0.001,None,0) * np.random.rand(3)
        # pc shift
        a+=offset

        # print("min",a.min(0),a.min(1))
        # print("max",a.max(0),a.max(1))
        idxs = (a.min(1)>=0)*(a.max(1)<full_scale) # box in [0, full_scale]^3 Actual size is full_scale / scale = 200m
        # exclude points out of bound
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]
        # pseudo_label_flag = 1 -> no pseudo_label
        if not pseudo_label_flag:
            # exclude pseudo_labels out of bound
            pseudo_label = pseudo_label[idxs]
        # convert a from numpy to torch tensor, dtype = long
        a = torch.from_numpy(a).long()

        # unique scene_label set
        scene_label_inds = np.unique(c).astype('int')
        # exclude invalid value(negative values)
        scene_label_inds = scene_label_inds[scene_label_inds >= 0]
        scene_label = np.zeros(NUM_CLASSES)
        # 1 -- this class exists
        scene_label[scene_label_inds] = 1.

        if len(text) > 0:
            has_text.append(idx)
            text = tokenize(text, truncate = True)
            # print(len(text[0]))
            text = torch.cat([text, torch.LongTensor(text.shape[0], 1).fill_(idx)], 1)
            texts.append(text)

        locs.append(torch.cat([a,torch.LongTensor(a.shape[0], 1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c if pseudo_label_flag else pseudo_label)) 
        scene_labels.append(torch.from_numpy(scene_label))
        if pseudo_label_flag:
            scene_names.append(scene_name)
        batch_offsets.append(batch_offsets[-1] + np.sum(idxs))

    # print(batch_offsets)
    
    # all points in one batch concat as one tensor
    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0) 
    labels = torch.cat(labels, 0) # one point, one label
    scene_labels = torch.stack(scene_labels) 
    # texts = torch.stack(texts) if len(has_text) > 0 else torch.tensor(-1) 
    texts = torch.cat(texts, 0) if len(has_text) > 0 else torch.tensor(-1) 
    # store the index of scenes that have texts 
    has_text = torch.tensor(has_text).long()
    
    return {
        'x': [locs, feats, batch_offsets], # locs -- B, N, 4   feats -- B, N, 3   batch_offsets -- 
        'y_orig': labels.long(), # B, N
        'y': scene_labels,       # B, BatchSize, NumClasses
        'text': [texts, has_text], # texts -- B, NumText(<=NumScenes), LenSeq   has_text -- B, NumTexts
        'id': tbl,
        'scene_names': scene_names # B, NumScenes
        }
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
    num_workers=4,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
