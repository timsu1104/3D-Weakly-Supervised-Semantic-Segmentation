# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from dataset.data import train_data_loader, train_files
import models # register the classes
from utils import stats
from utils.config import cfg
from utils.registry import MODEL_REGISTRY

use_cuda = torch.cuda.is_available()
saving_path = osp.join(cfg.path, cfg.training_name)
os.makedirs(saving_path, exist_ok=True)
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
    thresh=cfg.threshold
    start = time.time()
    num_pseudo_labels = 0
    total_label_num = 0
    correct_num = 0
    for i, batch in enumerate(tqdm(train_data_loader)):
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
            batch['y_orig']=batch['y_orig'].cuda()
        predictions=model(batch['x'][:-1])
        pseudo_labels, num = stats.get_pseudo_labels(predictions, threshold=thresh, show_stats=False)
        num_pseudo_labels += num
        total_label_num += pseudo_labels.size(0)
        correct, _ = stats.assess_label_quality(pseudo_labels, batch['y_orig'])
        correct_num += correct
        stats.store_pseudo_label(pseudo_labels.cpu(), batch['scene_names'], batch['x'][-1], saving_path, cfg.suffix)

    end = time.time()
    print(f"Using Thresh={thresh}. \nTotal elapsed {end-start}s, generated {num_pseudo_labels} labels ({num_pseudo_labels / total_label_num * 100}%), out of which {correct_num} are correct ({correct_num/num_pseudo_labels * 100}%)")
