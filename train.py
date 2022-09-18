# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sparseconvnet as scn
import time
import warnings

from dataset.data import train_data_loader, val_data_loader, train, val, valOffsets, valLabels
import models # register the classes
from utils import iou, loss
from utils.config import cfg
from utils.registry import MODEL_REGISTRY, LOSS_REGISTRY

# Setups
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN_NAME = cfg.training_name

use_cuda = torch.cuda.is_available()
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path
writer = SummaryWriter(os.path.join('exp', TRAIN_NAME))

model_, model_meta = MODEL_REGISTRY.get(cfg.model_name)
model = model_(cfg.pointcloud_model, cfg.text_model) if cfg.has_text else model_(cfg.pointcloud_model)
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
            batch['text'][0] = batch['text'][0].cuda()
            batch['text'][1] = batch['text'][1].cuda()
            batch['y'] = batch['y'].cuda()

        loss = 0
        
        global_logits, contrastive_meta = model((batch['x'], batch['text']), istrain=True)
        if cfg.loss.Classification:
            cls_loss, cls_meta = LOSS_REGISTRY.get('Classification')
            loss += cls_loss(global_logits, batch['y'])
        if cfg.has_text and cfg.loss.TextContrastive: 
            contrastive_loss, contrastive_meta = LOSS_REGISTRY.get('TextContrastive')
            loss += contrastive_loss(*contrastive_meta)
            
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
    writer.add_scalar("Train Loss", train_loss/(i+1), epoch)
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
                mean_iou = iou.evaluate(store.max(1)[1].numpy(),valLabels)
                writer.add_scalar("Validation accuracy", mean_iou, epoch)

writer.close()
