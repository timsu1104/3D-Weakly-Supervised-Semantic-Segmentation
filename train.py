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
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset.pseudo_loader import Pseudo_Images
from dataset.data import train_data_loader, val_data_loader, train, val, valOffsets, valLabels
from numpy import random
import models # register the classes
from utils import iou, loss
from utils.config import cfg, verbose
from utils.registry import MODEL_REGISTRY, LOSS_REGISTRY
from itertools import cycle
from models.GanDiscriminator import NaiiveCNN
# Setups
import numpy as np
import open3d as o3d
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN_NAME = cfg.training_name

random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

use_cuda = torch.cuda.is_available()
print("using cuda?")
print(use_cuda)
os.makedirs(os.path.join('exp', TRAIN_NAME), exist_ok=True)
exp_name=cfg.exp_path
writer = SummaryWriter(os.path.join('exp', TRAIN_NAME))
model_, model_meta = MODEL_REGISTRY.get(cfg.model_name)
model = model_(cfg.pointcloud_model, cfg.text_model) if cfg.has_text else model_(cfg.pointcloud_model)
model.load_state_dict(torch.load('/home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/exp/scene_level_with_fcnet_uppool/scene_level_with_fcnet_uppool-000000256-model.pth'), strict=False)
model.cuda()
if use_cuda:
    model=model.cuda()

training_epochs=cfg.epochs
training_epoch=scn.checkpoint_restore(model,exp_name,'model',use_cuda)
model.load_state_dict(torch.load('/home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/exp/scene_level_with_fcnet_uppool/scene_level_with_fcnet_uppool-000000256-model.pth'), strict=False)
model.cuda()

# optimizer = optim.Adam(model.parameters())
print(cfg)
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.lr}], lr=cfg.lr)
image_iter=0
print("Start from epoch", training_epoch)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.98)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, (0.1)**(1/100))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=training_epoch)
print('#classifier parameters', sum([x.nelement() for x in model.parameters()]))
if cfg.loss.Gan:
    
    embed_width = MODEL_REGISTRY.get(cfg.pointcloud_model.name)[1].get('embed_length', lambda m : m)(cfg.pointcloud_model.m)
    projector_, proj_meta=MODEL_REGISTRY.get('Projector')
    cropBox, crop_meta=MODEL_REGISTRY.get('cropBox')
    matting, crop_meta=MODEL_REGISTRY.get('DirectMattingModule')
    matting=matting(20,1)
    projector = projector_(embed_width, out_channels=cfg.projector.mask_channels, resolution=cfg.projector.mask_res)
    discriminator=NaiiveCNN()
    if use_cuda:
        projector = projector.cuda()
        discriminator=discriminator.cuda()
    #load dataset, discriminator,optimizer for discriminator
    cls_lst=['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
    valid_cls=[False,False,  False    ,False,True ,False,   False,  False, False,   False,      False,    False,    False,  False,   False,          False,          False,    False,   False,  False]
    data_transform = transforms.Compose([
        transforms.Resize(64),#TODO:decide a size to cut
        #transforms.RandomSizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    #TODO: I think some tricky normalize needs to be applied, like 1->0.999, 0->0.0001, or other?
    ])
    pseudo_dataset=Pseudo_Images('/home/shizhong/3DUNetWithText/dataset/pseudo_images',cls_lst,valid_cls,img_type='test',transform=data_transform)
    print("Size of pseudo dataset:")
    print(len(pseudo_dataset))
    # discriminator = discriminator_()
    pseudo_mask_loader=torch.utils.data.DataLoader(pseudo_dataset,
                                             batch_size=32, shuffle=True,
                                             num_workers=4)
    pseudo_iter=iter(cycle(pseudo_mask_loader))
    #print(next(pseudo_iter)[0].shape)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.dis_lr)
    optimizer.add_param_group({'params':projector.parameters(),'initial_lr':cfg.lr})

iteration_cnt=0
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
    first=0
    epoch_first=0
    for i, batch in enumerate(train_data_loader):
        iteration_cnt+=1
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
            cls_batch_loss=cls_loss(global_logits, batch['y'])
            loss += cls_batch_loss
            if cfg.label == 'pseudo':
                loss += cls_loss(meta, batch['y_orig'])
            writer.add_scalar("Classification_loss", cls_batch_loss.item(), iteration_cnt)
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
            

            cls_mask=torch.Tensor(valid_cls).cuda()
            gen_mask,gen_label = projector(batch['x'].coords, point_logits,pseudo_class, batch['x'].boxes, batch['x'].transform, cfg.projector.render_view) # (B, C, res, res)
            #print(batch['x'].boxes)
            print('projection', time.time()-start)
            gen_mask=gen_mask[torch.gather(cls_mask,0,gen_label).bool()]
            gen_label=gen_label[torch.gather(cls_mask,0,gen_label).bool()]
            #print(gen_label)
            #print(gen_mask)
            #print(gen_mask.size())
            #print("GENMASK", gen_mask, gen_mask.size())


            discriminator.train()
            gen_valid=discriminator(gen_mask,gen_label)
            #print(gen_valid)
            g_loss=torch.log(1-gen_valid)
            if(g_loss.nelement()!=0):
                if(iteration_cnt>100):
                    loss+=g_loss.mean()*0.05
                writer.add_scalar("Dicriminator Loss:Generator", g_loss.mean().item(), iteration_cnt)
            
            
            
        if cfg.loss.Gan=='WGanGP':
            #TODO:Complete WGanGP pipeline
            assert(cfg.loss.Gan=='WGanGP')



        if verbose: print("Forwarding elapsed {}s".format(e_iter - s_iter))

            
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if cfg.loss.Gan=='Gan':
            pseudo_image,cls_label=next(pseudo_iter)
            pseudo_image=pseudo_image.float()
            #print(pseudo_image.max())
            if use_cuda:
                pseudo_image=pseudo_image.cuda()
                cls_label=cls_label.cuda()
            you_want_multiple_update=0
            #TODO: This part might have some problem, you might need to update positive examples multiple times
            if you_want_multiple_update:
                #Do something
                None
            else:
                
                optimizer_d.zero_grad()
                d_loss=-torch.log(discriminator(pseudo_image,cls_label)).mean()
                writer.add_scalar("Dicriminator Loss: real(glide)", d_loss.mean().item(),iteration_cnt)
                adv_loss=-torch.log(1-discriminator(gen_mask.detach(),gen_label))
                if adv_loss.nelement()!=0:
                    d_loss+=adv_loss.mean()
                d_loss.backward()
            #print(i)
            #print(first)
            print(gen_mask.shape[0])
            if first==0 and gen_mask.shape[0]!=0:
                print('image generating...')
                first=1
                writer.add_image('target imgs', vutils.make_grid(pseudo_image, nrow=4),iteration_cnt)
                writer.add_image('gen imgs', vutils.make_grid(gen_mask, nrow=4),iteration_cnt)
                if epoch_first==0:
                #Visualization_test
                    ori_coords,original_else,cropped_coords, cropped_feats,batch_lens,dominate_class,box_class =cropBox(batch['x'].coords, point_logits, pseudo_class, batch['x'].boxes, batch['x'].transform,debug=True)
                    background,segmented_coords, segmented_feats,seg_box_class= matting(cropped_coords, cropped_feats,dominate_class,box_class)
                    epoch_first=1
                    pcd1 = o3d.geometry.PointCloud()
                    result=batch['x'].coords[ori_coords,:]
                    box_id=result[0][3]
                    for i in result[:,3]:
                        if cls_mask[box_class[i]]:
                            box_id=i
                            break
                    result=result[:,:3]
                    result2=background
                    result2=result2[result2[:,3]==box_id,:3]
                    result3=segmented_coords[segmented_coords[:,3]==box_id,:3]
                    #print(result,result2)
                    red=torch.tensor([1,0,0]).float()
                    green=torch.tensor([0,1,0]).float()
                    blue=torch.tensor([0,0,1]).float()
                    color=torch.cat([blue.repeat((result2.shape[0],1)),green.repeat((result3.shape[0],1))])
                    result=torch.cat([result2,result3])
                    pcd1.points = o3d.utility.Vector3dVector(result.cpu().numpy())
                    pcd1.colors = o3d.utility.Vector3dVector(color.numpy())
                    #o3d.visualization.draw_geometries([pcd1])
            if iteration_cnt%50==0:
                first=0
            optimizer_d.step()
        e_iter = time.time()
        writer.add_scalar("Train Loss", loss.item(), iteration_cnt)
    lr_scheduler.step()
    print(
        epoch,
        'Train loss', train_loss/(i+1), 
        'MegaMulAdd', scn.forward_pass_multiplyAdd_count/len(train)/1e6, 
        'MegaHidden', scn.forward_pass_hidden_states/len(train)/1e6,
        'time', time.time() - start, 's'
        )
    writer.add_scalar("Epoch Train Loss", train_loss/(i+1), epoch)
    scn.checkpoint_save(model,exp_name,'model',epoch, use_cuda)
    print("Checkpoint saved.")

    if scn.is_power2(epoch) or epoch % 2 == 0:
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
