# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch, data, iou
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
import time

from models import TextTransformer
from utils.config import cfg

TRAIN_NAME = cfg.training_name

use_cuda = torch.cuda.is_available()
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path

class Model(nn.Module):
    def __init__(self, pc_config, text_config):
        super().__init__()

        m = pc_config.m
        residual_blocks=pc_config.residual_blocks
        block_reps = pc_config.block_reps

        width = text_config.width
        vocab_size = text_config.vocab_size
        context_length = text_config.context_length
        layers = text_config.layers

        self.sparseModel = scn.Sequential(
            scn.InputLayer(data.dimension,data.full_scale, mode=4),
            scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False),
            scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks),
            scn.BatchNormReLU(m),
            scn.OutputLayer(data.dimension)
        )
        self.text_encoder = TextTransformer(context_length, width, layers, vocab_size)
        self.text_linear = nn.Linear(width, m)
        self.linear = nn.Linear(m, 20)
    def forward(self, x, istrain=False):
        if istrain:
            x, (text, has_text) = x

            BText, NumText, Length = text.size()
            text_feats = self.text_encoder(text.view(-1, Length), as_dict=True)['x'].view(BText, NumText, -1)
            text_feats = self.text_linear(text_feats)

            out_feats = self.sparseModel(x[:-1]) # B, NumPts, C

            batch_offsets = x[-1]
            B = batch_offsets.size(0) - 1
            global_feats = []
            for idx in range(B):
                global_feats.append(torch.mean(out_feats[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
            global_feats = torch.stack(global_feats)
            global_logits=self.linear(global_feats) # B, 20
            
            global_logits = (global_logits, global_feats, text_feats, has_text)
        else:
            out_feats = self.sparseModel(x) 
            global_logits=self.linear(out_feats)

        return global_logits

def contrastive_loss(pc: torch.Tensor, text: torch.Tensor, has_text):
    """
    pc: B, m
    text: B', num_text, m
    """
    assert text.ndim == 3, text.size()
    similarity = text @ pc.T # B', num_text, B
    num_text = similarity.size(1)
    labels = torch.tile(has_text[:, None], (1, num_text))
    contrast_loss = F.cross_entropy(similarity.transpose(1, 2), labels)
    return contrast_loss

unet=Model(cfg.pointcloud_model, cfg.text_model)
if use_cuda:
    unet=unet.cuda()

training_epochs=512
training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
optimizer = optim.Adam(unet.parameters())
print('#classifier parameters', sum([x.nelement() for x in unet.parameters()]))

for epoch in range(training_epoch, training_epochs+1):
    print("Starting epoch", epoch)
    unet.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss = 0
    print("Inference started.")
    for i, batch in enumerate(data.train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['x'][2] = batch['x'][2].cuda()
            batch['text'][0] = batch['text'][0].cuda()
            batch['text'][1] = batch['text'][1].cuda()
            batch['y'] = batch['y'].cuda()
        global_logits, global_feats, text_feats, has_text = unet((batch['x'], batch['text']), istrain=True)
        loss = F.multilabel_soft_margin_loss(global_logits, batch['y']) + contrastive_loss(global_feats, text_feats, has_text)
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
    print("Checkpoint saved.")

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
