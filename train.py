# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sparseconvnet as scn
import time
import warnings

from utils import iou, loss, stats
from utils.config import *

if dist_flag:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group("nccl")
    
import models # register the classes
from utils.registry import MODEL_REGISTRY, LOSS_REGISTRY
from utils.ap_helper import APCalculator
from dataset.data import class2type

# Setups
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

use_cuda = torch.cuda.is_available()
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path
if local_rank == 0:
    writer = SummaryWriter(os.path.join('exp', TRAIN_NAME))

model_, model_meta = MODEL_REGISTRY.get(cfg.model_name)
model = model_(cfg.pointcloud_model, cfg.text_model) if cfg.has_text else model_(cfg.pointcloud_model)
training_epochs=cfg.epochs
training_epoch=scn.checkpoint_restore(model,exp_name,'model',use_cuda)

if dist_flag:
    model = torch.nn.parallel.DistributedDataParallel(model.to(device),
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)
else:
    if use_cuda:
        model=model.cuda()

from dataset.data import NUM_CLASSES, train_data_loader, val_data_loader, train, val, valOffsets, valLabels

# optimizer = optim.Adam(model.parameters())
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.lr}], lr=cfg.lr)
print("Start from epoch", training_epoch)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.98)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, (0.1)**(1/100))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=training_epoch)
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))

for epoch in range(training_epoch, training_epochs+1):
    if local_rank == 0:
        print("Starting epoch", epoch)
    model.train()
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss = 0
    if verbose: print("Inference started.")
    e_iter = time.time()
    for i, batch in enumerate(train_data_loader):
        s_iter = time.time()
        if verbose: print("Data fetch elapsed {}s".format(s_iter - e_iter))
        optimizer.zero_grad()
        if use_cuda:
            batch['x']['feature'] = batch['x']['feature'].cuda()
            batch['text'][0] = batch['text'][0].cuda()
            batch['text'][1] = batch['text'][1].cuda()
            batch['y'] = batch['y'].cuda()
            batch['y_orig'] = batch['y_orig'].cuda()

        loss = 0
        
        scene_names_with_texts = batch.get('scene_names_with_texts', None)

        global_logits, meta = model((batch['x'], batch['text']), scene_names_with_texts, istrain=True)
        if cfg.loss.Classification:
            cls_loss, cls_meta = LOSS_REGISTRY.get('Classification')
            loss += cls_loss(global_logits, batch['y'])
            if cfg.label == 'pseudo':
                loss += cls_loss(meta, batch['y_orig'])
        if cfg.has_text and cfg.loss.TextContrastive: 
            contrastive_loss, contrastive_meta = LOSS_REGISTRY.get('TextContrastive')
            loss += contrastive_loss(*meta)
        e1_iter = time.time()
        if verbose: print("Forwarding elapsed {}s".format(e1_iter - s_iter))
            
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        e_iter = time.time()
        if verbose: print("Backward elapsed {}s".format(e_iter - e1_iter))
        if verbose: print("Model running elapsed {}s".format(e_iter-s_iter))
        
    lr_scheduler.step()
    if local_rank == 0:
        if dist_flag:
            save_model = model.module
        else:
            save_model = model
        print(
            epoch,
            'Train loss', train_loss/(i+1), 
            'MegaMulAdd', scn.forward_pass_multiplyAdd_count/len(train)/1e6, 
            'MegaHidden', scn.forward_pass_hidden_states/len(train)/1e6,
            'time', time.time() - start, 's'
            )
        writer.add_scalar("Train Loss", train_loss/(i+1), epoch)
        scn.checkpoint_save(save_model,exp_name,'model',epoch, use_cuda)
        if verbose: print("Checkpoint saved.")

    if local_rank == 0 and (stats.is_power2(epoch) or epoch % 32 == 0):
        with torch.no_grad():
            if dist_flag:
                val_model = model.module
            else:
                val_model = model
            val_model.eval()
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
                    pred_seg, pred_det = val_model(batch['x'])
                        
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
                    epoch,
                    rep,
                    'time', time.time() - start, 's'
                    )
                mean_iou = iou.evaluate(store.numpy(),valLabels)
                
                print('-'*10, 'prop: iou_thresh: 0.25', '-'*10)
                metrics_dict = ap_calculator.compute_metrics()
                for key in metrics_dict:
                    print('eval %s: %f'%(key, metrics_dict[key]))
                    
                writer.add_scalar("Validation accuracy", mean_iou, epoch)

writer.close()
