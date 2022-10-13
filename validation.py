# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import sparseconvnet as scn
import time
import warnings

from dataset.data import val_data_loader, val, valOffsets, valLabels, valScenes
import models # register the classes
from utils import iou, vis_seg
from utils.config import cfg, visual_flag
from utils.registry import MODEL_REGISTRY

# Setups
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN_NAME = cfg.training_name

use_cuda = torch.cuda.is_available()
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path

model_, model_meta = MODEL_REGISTRY.get(cfg.model_name)
model = model_(cfg.pointcloud_model, cfg.text_model) if cfg.has_text else model_(cfg.pointcloud_model)
if use_cuda:
    model=model.cuda()

training_epochs=cfg.epochs
training_epoch=scn.checkpoint_restore(model,exp_name,'model',use_cuda)
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))

with torch.no_grad():
    model.eval()
    store=torch.zeros(valOffsets[-1],20)
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    for rep in range(1, 1+cfg.pointcloud_data.val_reps):
        for i,batch in enumerate(val_data_loader):
            if use_cuda:
                batch['x'].feature=batch['x'].feature.cuda()
                batch['y_orig']=batch['y_orig'].cuda()
            predictions=model(batch['x'])
            store.index_add_(0,batch['point_ids'], predictions.cpu())
        print(
            training_epoch,
            rep,
            'Val MegaMulAdd', scn.forward_pass_multiplyAdd_count/len(val)/1e6, 
            'MegaHidden', scn.forward_pass_hidden_states/len(val)/1e6,
            'time', time.time() - start, 's'
            )
        mean_iou = iou.evaluate(store.max(1)[1].numpy(),valLabels)
    
    # visualize
    if visual_flag:
        vis_seg.write_seg_result(valScenes, store.max(1)[1].numpy(), valOffsets, multiprocess=True)
