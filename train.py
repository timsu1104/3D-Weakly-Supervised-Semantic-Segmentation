# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
import time

from dataset.data import train_data_loader, val_data_loader, train, val, valOffsets, valLabels
import models # register the classes
from utils import iou, loss
from utils.config import cfg
from utils.registry import MODEL_REGISTRY, LOSS_REGISTRY

TRAIN_NAME = cfg.training_name

use_cuda = torch.cuda.is_available()
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path

model=MODEL_REGISTRY.get(cfg.model_name)(cfg.pointcloud_model, cfg.text_model)
if use_cuda:
    model=model.cuda()

training_epochs=cfg.epochs
training_epoch=scn.checkpoint_restore(model,exp_name,'model',use_cuda)
optimizer = optim.Adam(model.parameters())
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))

for epoch in range(training_epoch, training_epochs+1):
    print("Starting epoch", epoch)
    model.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss = 0
    print("Inference started.")
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['x'][2] = batch['x'][2].cuda()
            batch['text'][0] = batch['text'][0].cuda()
            batch['text'][1] = batch['text'][1].cuda()
            batch['y'] = batch['y'].cuda()
        global_logits, global_feats, text_feats, has_text = model((batch['x'], batch['text']), istrain=True)

        loss = 0
        if cfg.loss.Classification:
            loss += LOSS_REGISTRY.get('Classification')(global_logits, batch['y'])
        if cfg.loss.TextContrastive: 
            loss += LOSS_REGISTRY.get('TextContrastive')(global_feats, text_feats, has_text)
            
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(
        epoch,
        'Train loss', train_loss/(i+1), 
        'MegaMulAdd', scn.forward_pass_multiplyAdd_count/len(train)/1e6, 
        'MegaHidden', scn.forward_pass_hidden_states/len(train)/1e6,
        'time', time.time() - start, 's'
        )
    scn.checkpoint_save(model,exp_name,'model',epoch, use_cuda)
    print("Checkpoint saved.")

    if scn.is_power2(epoch):
        with torch.no_grad():
            model.eval()
            store=torch.zeros(valOffsets[-1],20)
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+cfg.pointcloud_data.val_reps):
                for i,batch in enumerate(val_data_loader):
                    if use_cuda:
                        batch['x'][1]=batch['x'][1].cuda()
                        batch['y_orig']=batch['y_orig'].cuda()
                    predictions=model(batch['x'])
                    store.index_add_(0,batch['point_ids'],predictions.cpu())
                print(
                    epoch,
                    rep,
                    'Val MegaMulAdd', scn.forward_pass_multiplyAdd_count/len(val)/1e6, 
                    'MegaHidden', scn.forward_pass_hidden_states/len(val)/1e6,
                    'time', time.time() - start, 's'
                    )
                iou.evaluate(store.max(1)[1].numpy(),valLabels)
