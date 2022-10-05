# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sparseconvnet as scn
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from dataset.data import train_data_loader
import models # register the classes
from utils import stats
from utils.config import cfg
from utils.registry import MODEL_REGISTRY
from matplotlib import pyplot as plt

TRAIN_NAME = cfg.training_name
# THRESHOLDS = torch.linspace(10, 100, 10).numpy()
# THRESHOLDS = torch.linspace(0.72, 0.74, 9).numpy()
THRESHOLDS = np.linspace(1.0, 0.6, 41)

use_cuda = torch.cuda.is_available()
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path

model_, model_meta = MODEL_REGISTRY.get(cfg.model_name)
model=model_(cfg.pointcloud_model, cfg.text_model) if cfg.has_text else model_(cfg.pointcloud_model)
if use_cuda:
    model=model.cuda()

training_epoch=scn.checkpoint_restore(model,exp_name,'model',use_cuda)
optimizer = optim.Adam(model.parameters())
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))

with torch.no_grad():
    model.eval()
    labelnum_prop = []
    correct_prop = []
    for thresh in THRESHOLDS:
        start = time.time()
        num_pseudo_labels = 0
        total_label_num = 0
        correct_num = 0
        for i, batch in enumerate(tqdm(train_data_loader)):
            if use_cuda:
                batch['x'].feature=batch['x'].feature.cuda()
                batch['y_orig']=batch['y_orig'].cuda()
                batch['y']=batch['y'].cuda() # scene_label
            predictions=model(batch['x'])
            pseudo_labels, num = stats.get_pseudo_labels(predictions, batch['y'], batch['x'].batch_offsets, threshold=thresh, show_stats=False)
            num_pseudo_labels += num
            total_label_num += pseudo_labels.size(0)
            correct, _ = stats.assess_label_quality(pseudo_labels, batch['y_orig'])
            correct_num += correct
        end = time.time()
        print(f"Using Thresh={thresh}. \nTotal elapsed {end-start}s, generated {num_pseudo_labels} labels ({num_pseudo_labels / total_label_num * 100}%), out of which {correct_num} are correct ({correct_num/num_pseudo_labels * 100}%)")
        labelnum_prop.append((num_pseudo_labels / total_label_num).cpu())
        correct_prop.append((correct_num/num_pseudo_labels).cpu())

if not os.path.exists('visualization/'):
    os.makedirs('visualization/')
    
plt.figure()
plt.plot(THRESHOLDS, labelnum_prop, label='Label number')
plt.plot(THRESHOLDS, correct_prop, label='Label precision')
plt.legend()
plt.savefig("visualization/Statistics.jpg")
