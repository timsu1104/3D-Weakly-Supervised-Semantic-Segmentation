# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2

import os
import torch, data, iou
import torch.nn as nn
import torch.optim as optim
import sparseconvnet as scn
import time

TRAIN_NAME = 'scene_level_baseline'

use_cuda = torch.cuda.is_available()
if not os.path.exists('exp'):
    os.makedirs('exp')
if not os.path.exists(os.path.join('exp', TRAIN_NAME)):
    os.makedirs(os.path.join('exp', TRAIN_NAME))
exp_name=os.path.join('exp', TRAIN_NAME, TRAIN_NAME)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparseModel = scn.Sequential(
            scn.InputLayer(data.dimension,data.full_scale, mode=4),
            scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False),
            scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks),
            scn.BatchNormReLU(m),
            scn.OutputLayer(data.dimension)
        )
        self.linear = nn.Linear(m, 20)
    def forward(self, x, istrain=False):
        if istrain:
            batch_offsets = x[-1]
            B = batch_offsets.size(0) - 1
            out_feats = self.sparseModel(x[:-1]) # B, NumPts, C
            global_feats = []
            for idx in range(B):
                global_feats.append(torch.mean(out_feats[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
            global_feats = torch.stack(global_feats)
            assert global_feats.size(0) == B == 32, f"{global_feats.size(0)}"
            global_logits=self.linear(global_feats) # B, 20
        else:
            out_feats = self.sparseModel(x) 
            global_logits=self.linear(out_feats)

        return global_logits

unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epochs=512
training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
optimizer = optim.Adam(unet.parameters())
print('#classifier parameters', sum([x.nelement() for x in unet.parameters()]))

for epoch in range(training_epoch, training_epochs+1):
    unet.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    for i, batch in enumerate(data.train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['x'][2] = batch['x'][2].cuda()
            batch['y'] = batch['y'].cuda()
        predictions = unet(batch['x'], istrain=True)
        # print(predictions.size())
        # print(batch['y'].size())
        loss = criterion(predictions, batch['y'])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(
        epoch,
        'Train loss',train_loss/(i+1), 
        'MegaMulAdd',scn.forward_pass_multiplyAdd_count/len(data.train)/1e6, 
        'MegaHidden',scn.forward_pass_hidden_states/len(data.train)/1e6,
        'time',time.time() - start,'s'
        )
    scn.checkpoint_save(unet,exp_name,'unet',epoch, use_cuda)

    if scn.is_power2(epoch):
        with torch.no_grad():
            unet.eval()
            store=torch.zeros(data.valOffsets[-1],20)
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+data.val_reps):
                for i,batch in enumerate(data.val_data_loader):
                    if use_cuda:
                        batch['x'][1]=batch['x'][1].cuda()
                        batch['y_orig']=batch['y_orig'].cuda()
                    predictions=unet(batch['x'])
                    store.index_add_(0,batch['point_ids'],predictions.cpu())
                print(
                    epoch,
                    rep,
                    'Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count/len(data.val)/1e6, 
                    'MegaHidden', scn.forward_pass_hidden_states/len(data.val)/1e6,
                    'time',time.time() - start,
                    's'
                    )
                iou.evaluate(store.max(1)[1].numpy(),data.valLabels)
