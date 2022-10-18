import torch
import torch.nn as nn
from dataset.data import train_data_loader, val_data_loader, train, val, valOffsets, valLabels
from utils.registry import MODEL_REGISTRY
import sparseconvnet as scn
from utils.config import cfg
import models


def extract_text_feats(text_encoder, text_linear, path):
    use_cuda = torch.cuda.is_available()
    for i, batch in enumerate(train_data_loader):
        if use_cuda:
            # batch['x'][1] = batch['x'][1].cuda()
            batch['text'][0] = batch['text'][0].cuda()
            batch['text'][1] = batch['text'][1].cuda()
            # batch['scene_names'] = batch['scene_names'].cuda()
            # batch['y'] = batch['y'].cuda()
            # batch['y_orig'] = batch['y_orig'].cuda()
        text, has_text = batch['text']
        scene_names = batch['scene_names']
        if has_text.size(0) > 0:
                NumText, Length = text.size()
                
                text_without_index = text.view(-1, Length)[:,:-1]
                text_feats = text_encoder(text_without_index, as_dict=True)['x'].view(NumText, -1)
                text_feats = text_linear(text_feats)
                text_feats = torch.cat((text_feats,text[:,-1:]), 1)
        else:
            text_feats = -1
        
        print(scene_names)
        for j, scene in enumerate(scene_names):
            this_text_feats = text_feats[(text_feats[:,-1] == j).reshape((len(text_feats),)),:]
            # print(path+scene+'.pt')
            torch.save(this_text_feats, path+scene+'.pt')
    return


# config
pc_config = cfg.pointcloud_model
text_config = cfg.text_model
_, pc_meta = MODEL_REGISTRY.get(pc_config.name)
# text model -- current: clip transformer
text_model, _ = MODEL_REGISTRY.get(text_config.name)
# point cloud feature embedding dim
embed_width = pc_meta.get('embed_length', lambda m : m)(pc_config.m)

pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
text_encoder = text_model(**text_config)
text_linear = nn.Linear(text_config.width, embed_width)

path = '/home/zhuhe/3DUNetWithText/3DUNetWithText/dataset/text_feats/'
extract_text_feats(text_encoder, text_linear, path)