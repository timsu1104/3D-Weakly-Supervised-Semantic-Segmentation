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
from torchvision import transforms, utils
from utils import iou, loss
from utils.config import *

from dataset.pseudo_loader import Pseudo_Images
from dataset.data import train_data_loader, val_data_loader, train, val, valOffsets, valLabels

import models # register the classes
from utils import iou, loss
from utils.config import cfg, verbose
from utils.registry import MODEL_REGISTRY, LOSS_REGISTRY
from itertools import cycle
from models.GanDiscriminator import NaiiveCNN
# Setups
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN_NAME = cfg.training_name

use_cuda = torch.cuda.is_available()
print("using cuda?")
print(use_cuda)
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path
writer = SummaryWriter(os.path.join('exp', TRAIN_NAME))

model_, model_meta = MODEL_REGISTRY.get(cfg.model_name)
model = model_(cfg.pointcloud_model, cfg.text_model) if cfg.has_text else model_(cfg.pointcloud_model)
if use_cuda:
    model=model.cuda()

training_epochs=cfg.epochs
training_epoch=scn.checkpoint_restore(model,exp_name,'model',use_cuda)

# optimizer = optim.Adam(model.parameters())
print(cfg)
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.lr}], lr=cfg.lr)

print("Start from epoch", training_epoch)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.98)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, (0.1)**(1/100))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=training_epoch)
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))
if cfg.loss.Gan:
    
    embed_width = MODEL_REGISTRY.get(cfg.pointcloud_model.name)[1].get('embed_length', lambda m : m)(cfg.pointcloud_model.m)
    projector_, proj_meta=MODEL_REGISTRY.get('Projector')
    projector = projector_(embed_width, out_channels=cfg.projector.mask_channels, resolution=cfg.projector.mask_res)
    discriminator=NaiiveCNN()
    if use_cuda:
        projector = projector.cuda()
        discriminator=discriminator.cuda()
    #load dataset, discriminator,optimizer for discriminator
    cls_lst=['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
    valid_cls=[False,False,  False    ,False,True ,False,   False,  False, False,   False,      False,    False,    False,  False,   False,          False,          False,    False,   False,  False]
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(cfg.projector.mask_res),#TODO:decide a size to cut
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    #TODO: I think some tricky normalize needs to be applied, like 1->0.999, 0->0.0001, or other?
    ])
    pseudo_dataset=Pseudo_Images('/home/shizhong/3DUNetWithText/dataset/pseudo_images',cls_lst,valid_cls,img_type='mask',transform=transforms.ToTensor())
    # discriminator = discriminator_()
    pseudo_mask_loader=torch.utils.data.DataLoader(pseudo_dataset,
                                             batch_size=32, shuffle=True,
                                             num_workers=4)
    pseudo_iter=iter(cycle(pseudo_mask_loader))
    #print(next(pseudo_iter)[0].shape)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.dis_lr)
    optimizer.add_param_group({'params':projector.parameters(),'initial_lr':cfg.lr})

    
for epoch in range(training_epoch, training_epochs+1):
    print("Starting epoch", epoch)
    model.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss = 0
    print("Inference started.")
    
    e_iter = 0
    for i, batch in enumerate(train_data_loader):
        s_iter = time.time()
        if verbose: print("Data fetch elapsed {}s".format(s_iter - e_iter))

        optimizer.zero_grad()
        if use_cuda:
            batch['x'].feature = batch['x'].feature.cuda()
            batch['x'].boxes = batch['x'].boxes.cuda()
            for i, t in enumerate(batch['x'].transform):
                batch['x'].transform[i] = t.cuda()
            batch['text'][0] = batch['text'][0].cuda()
            batch['text'][1] = batch['text'][1].cuda()
            batch['y'] = batch['y'].cuda()
            batch['y_orig'] = batch['y_orig'].cuda()

        loss = 0
        start = time.time()
        global_logits, point_logits,pseudo_class = model((batch['x'], batch['text']), istrain=True)
        print('backbone', time.time()-start)
        if cfg.loss.Classification:
            cls_loss, cls_meta = LOSS_REGISTRY.get('Classification')
            loss += cls_loss(global_logits, batch['y'])
            if cfg.label == 'pseudo':
                loss += cls_loss(meta, batch['y_orig'])
        if cfg.has_text and cfg.loss.TextContrastive: 
            contrastive_loss, meta = LOSS_REGISTRY.get('TextContrastive')
            loss += contrastive_loss(*meta)


        if cfg.loss.Gan=='Gan':
            
            # projector
            torch.cuda.empty_cache()
            projector.train()
            if use_cuda:
                batch['x'].coords = batch['x'].coords.cuda()
            start = time.time()
            gen_mask,gen_label = projector(batch['x'].coords, point_logits,pseudo_class, batch['x'].boxes, batch['x'].transform, cfg.projector.render_view) # (B, C, res, res)
            print('projection', time.time()-start)
            # print(gen_mask.size())
            print("GENMASK", gen_mask, gen_mask.size())
            discriminator.train()
            optimizer_d.zero_grad()
            gen_valid=discriminator(gen_mask,gen_label)
            g_loss=torch.log(1-gen_valid)
            if(g_loss.nelement()!=0):
                loss+=g_loss.mean()

            pseudo_image,cls_label=next(pseudo_iter)
            pseudo_image=pseudo_image.float()
            if use_cuda:
                pseudo_image=pseudo_image.cuda()
                cls_label=cls_label.cuda()
            you_want_multiple_update=0
            #TODO: This part might have some problem, you might need to update positive examples multiple times
            if you_want_multiple_update:
                #Do something
                None
            else:
                d_loss=-torch.log(discriminator(pseudo_image,cls_label)).mean()-torch.log(1-discriminator(gen_mask.detach(),gen_label)).mean()
                d_loss.backward()

            
        if cfg.loss.Gan=='WGanGP':
            #TODO:Complete WGanGP pipeline
            assert(cfg.loss.Gan=='WGanGP')



        if verbose: print("Forwarding elapsed {}s".format(e_iter - s_iter))

            
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if cfg.loss.Gan=='Gan':
            optimizer_d.step()
        e_iter = time.time()
    lr_scheduler.step()
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

    if scn.is_power2(epoch) or epoch % 32 == 0:
        with torch.no_grad():
            model.eval()
            store=torch.zeros(valOffsets[-1],20)
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+cfg.pointcloud_data.val_reps):
                for i,batch in enumerate(val_data_loader):
                    if use_cuda:
                        batch['x'].feature=batch['x'].feature.cuda()
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
