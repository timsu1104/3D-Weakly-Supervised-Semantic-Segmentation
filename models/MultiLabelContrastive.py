import torch
import torch.nn as nn

from dataset.data import NUM_CLASSES
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MultiLabelContrastive(nn.Module):
    def __init__(self, pc_config, text_config):
        super().__init__()

        # point cloud model
        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        # text model -- current: clip transformer
        text_model, _ = MODEL_REGISTRY.get(text_config.name)
        # point cloud feature embedding dim
        embed_width = pc_meta.get('embed_length', lambda m : m)(pc_config.m)

        self.pc_encoder = pc_model(**pc_config)
        self.text_encoder = text_model(**text_config)
        # nn.Linear(in_dim, out_dim, bias = False)
        # text and pc features should be of the same dim
        self.text_linear = nn.Linear(text_config.width, embed_width)
        # classification logits
        self.linear = nn.Linear(embed_width, NUM_CLASSES)

    def forward(self, x, istrain=False):
        if istrain:
            (coords, feats, batch_offsets), (text, has_text) = x

            if has_text.size(0) > 0:
                # BText, NumText, Length = text.size()
                NumText, Length = text.size()
                #text_feats = self.text_encoder(text.view(-1, Length), as_dict=True)['x'].view(BText, NumText, -1)
                
                # text_without_index = text.view(-1, Length)[:,:-1]
                text_without_index = text.view(-1, Length)[:,:-1]
                text_feats = self.text_encoder(text_without_index, as_dict=True)['x'].view(NumText, -1)
                # print(text_feats.size())
                text_feats = self.text_linear(text_feats)
                # print(text_feats.size())
                text_feats = torch.cat((text_feats,text[:,-1:]), 1)
            else:
                text_feats = -1

            # feature of each point extracted by pc_encoder(current -- sparseconvnet)
            out_feats = self.pc_encoder([coords, feats]) # B * NumPts, C

            # B -> batch_size
            # print(batch_offsets)
            B = len(batch_offsets) - 1
            global_feats = []
            for idx in range(B):
                global_feats.append(torch.mean(out_feats[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
            global_feats = torch.stack(global_feats)
            # print("global_feats_size", global_feats.size())
            global_logits=self.linear(global_feats) # B, 20
            
            global_logits = global_logits, (global_feats, text_feats, has_text)
        else:
            out_feats = self.pc_encoder(x) 
            global_logits=self.linear(out_feats)

        return global_logits


@MODEL_REGISTRY.register()
class MultiLabel(nn.Module):
    def __init__(self, pc_config):
        super().__init__()

        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        embed_width = pc_meta.get('embed_length', lambda m : m)(pc_config.m)

        self.pc_encoder = pc_model(**pc_config)
        self.linear = nn.Linear(embed_width, NUM_CLASSES)

    def forward(self, x, istrain=False):
        if istrain:
            (coords, feats, batch_offsets), _ = x

            out_feats = self.pc_encoder([coords, feats]) # B * NumPts, C

            B = len(batch_offsets) - 1
            global_feats = []
            for idx in range(B):
                global_feats.append(torch.mean(out_feats[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
            global_feats = torch.stack(global_feats)
            global_logits=self.linear(global_feats) # B, 20
            
            global_logits = global_logits, None
        else:
            out_feats = self.pc_encoder(x) 
            global_logits=self.linear(out_feats)

        return global_logits

@MODEL_REGISTRY.register()
class FullySupervised(nn.Module):
    def __init__(self, pc_config):
        super().__init__()
        
        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        embed_width = pc_meta.get('embed_length', lambda m : m)(pc_config.m)

        self.pc_encoder = pc_model(**pc_config)
        self.linear = nn.Linear(embed_width, NUM_CLASSES)

    def forward(self, x, istrain=False):
        if istrain:
            (coords, feats, batch_offsets), _ = x
            out_feats = self.pc_encoder([coords, feats]) # B * NumPts, C
            logits=self.linear(out_feats)

            B = len(batch_offsets) - 1
            global_logits = []
            for idx in range(B):
                global_logits.append(torch.mean(logits[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
            global_logits = torch.stack(global_logits) # B, 20
            
            global_logits = global_logits, logits
        else:
            out_feats = self.pc_encoder(x) 
            global_logits=self.linear(out_feats)

        return global_logits