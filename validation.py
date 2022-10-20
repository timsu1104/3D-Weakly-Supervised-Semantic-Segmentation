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

from dataset.data import val_data_loader, val, valOffsets, valLabels
import models # register the classes
from utils import iou
from utils.config import cfg
from utils.registry import MODEL_REGISTRY
from utils.ap_helper import APCalculator
from dataset.data import class2type

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
    store = torch.zeros(valOffsets[-1])
    ap_calculator = APCalculator(class2type_map=class2type)
    start = time.time()
    for rep in range(1,1+cfg.pointcloud_data.val_reps):
        for i,batch in enumerate(val_data_loader):
            if use_cuda:
                batch['x'].coords=batch['x'].coords.cuda()
                batch['x'].feature=batch['x'].feature.cuda()
                batch['x'].boxes=batch['x'].boxes.cuda() # B, R, 6
                batch['x'].shapes=batch['x'].shapes.cuda()
                batch['y_orig']=batch['y_orig'].cuda()
            pred_seg, pred_det = model(batch['x'])
                
            # segmentation
            store.index_add_(0, batch['point_ids'], pred_seg.view(-1).cpu())
            
            # detection
            gt_boxes = batch['gt_box']
            batch_gt_map_cls = [
                list(zip(target_bbox_semcls, target_bbox))
                for target_bbox, target_bbox_mask, target_bbox_semcls in gt_boxes]
            batch_pred_map_cls = [
                list(zip(batch_pred_cls, batch_boxes, batch_pred_score)) 
                for batch_pred_cls, batch_boxes, batch_pred_score in zip(pred_det[1], batch['x'].boxes.cpu(), pred_det[0])]
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        print(
            training_epoch,
            rep,
            'time', time.time() - start, 's'
            )
        mean_iou = iou.evaluate(store.max(1)[1].numpy(),valLabels)
        
        print('-'*10, 'prop: iou_thresh: 0.25', '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print('eval %s: %f'%(key, metrics_dict[key]))
