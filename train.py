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
#I can't make this run, also, I have changed dataset/__init__.py to NULL to run the program
dist_flag=0
local_rank=0
import test_blocks
if dist_flag:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group("nccl")
    
import models # register the classes
from utils.registry import MODEL_REGISTRY, LOSS_REGISTRY
from itertools import cycle

# Setups
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

use_cuda = torch.cuda.is_available()
print("using cuda?")
print(use_cuda)
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

from dataset.data import train_data_loader, val_data_loader, train, val, valOffsets, valLabels
from dataset.pseudo_loader import Pseudo_Images
# optimizer = optim.Adam(model.parameters())
print(cfg)
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.lr}], lr=cfg.lr)
print("Start from epoch", training_epoch)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.98)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, (0.1)**(1/100))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=training_epoch)
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))
if cfg.loss.Gan:
    #load dataset, discriminator,optimizer for discriminator
    cls_lst=['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
    valid_cls=[False,False,  False    ,False,True ,False,   False,  False, False,   False,      False,    False,    False,  False,   False,          False,          False,    False,   False,  False]
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    #TODO: I think some tricky normalize needs to be applied, like 1->0.999, 0->0.0001, or other?
    pseudo_dataset=Pseudo_Images('./dataset/pseudo_images/',cls_lst,valid_cls,img_type='mask',Transform=transforms.ToTensor())
    discriminator=MODEL_REGISTRY.get(cfg.loss.gan_name)
    pseudo_mask_loader=torch.utils.data.DataLoader(pseudo_dataset,
                                             batch_size=25, shuffle=True,
                                             num_workers=4) 
    pseudo_iter=iter(cycle(pseudo_mask_loader))
    optimizer_d = optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.dis_lr}], lr=cfg.dis_lr)

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
        print(batch)
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
        
        global_logits, meta = model((batch['x'], batch['text']), istrain=True)
        print(global_logits,meta)
        if cfg.loss.Classification:
            cls_loss, cls_meta = LOSS_REGISTRY.get('Classification')
            loss += cls_loss(global_logits, batch['y'])
            if cfg.label == 'pseudo':
                loss += cls_loss(meta, batch['y_orig'])
        if cfg.has_text and cfg.loss.TextContrastive: 
            contrastive_loss, meta = LOSS_REGISTRY.get('TextContrastive')
            loss += contrastive_loss(*meta)
        e1_iter = time.time()


        if cfg.loss.Gan=='Gan':
            discriminator.train()
            optimizer_d.zero_grad()
            #TODO: checkout the actuall functions
            boundingbox=GetBoundingBox(batch)
            cropped_data=Cropping(logits,meta,batch,boundingbox)
            masks,y=MaskProduce(cropped_data)#mask are logits
            gen_img=nn.Sigmoid()(mask)
            images=Projector(masks)
            gen_valid=discriminator(gen_img,y)
            g_loss=torch.log(1-gen_valid)
            g_loss.mean().backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            pseudo_image,cls_label=pseudo_iter.next()
            you_want_multiple_update=0
            #TODO: This part might have some problem, you might need to update positive examples multiple times
            if you_want_multiple_update:
                #Do something
            else:
                d_loss=-torch.log(discriminator(x,cls_label)).mean()-torch.log(1-discriminator(gen_img.detach(),y)).mean()
                d_loss.backward()
                optimizer_d.step()
            
        if cfg.loss.Gan=='WGanGP':
            #TODO:Complete WGanGP pipeline



        if verbose: print("Forwarding elapsed {}s".format(e1_iter - s_iter))
            
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        e_iter = time.time()
        if verbose: print("Backward elapsed {}s".format(e_iter - e1_iter))
        if verbose: print("Model running elapsed {}s".format(e_iter-s_iter))
        
    lr_scheduler.step()
    if local_rank == 0:
        print(
            epoch,
            'Train loss', train_loss/(i+1), 
            'MegaMulAdd', scn.forward_pass_multiplyAdd_count/len(train)/1e6, 
            'MegaHidden', scn.forward_pass_hidden_states/len(train)/1e6,
            'time', time.time() - start, 's'
            )
        writer.add_scalar("Train Loss", train_loss/(i+1), epoch)
        scn.checkpoint_save(model.module,exp_name,'model',epoch, use_cuda)
        if verbose: print("Checkpoint saved.")

    if local_rank == 0 and (scn.is_power2(epoch) or epoch % 32 == 0):
        with torch.no_grad():
            val_model = model.module()
            val_model.eval()
            store=torch.zeros(valOffsets[-1],20)
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+cfg.pointcloud_data.val_reps):
                for i,batch in enumerate(val_data_loader):
                    if use_cuda:
                        batch['x'].feature=batch['x'].feature.cuda()
                        batch['y_orig']=batch['y_orig'].cuda()
                    predictions=val_model(batch['x'])
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
