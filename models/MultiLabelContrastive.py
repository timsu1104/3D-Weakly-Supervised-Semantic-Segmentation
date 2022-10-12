import torch
import torch.nn as nn

from dataset.data import NUM_CLASSES
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MultiLabelContrastive(nn.Module):
    def __init__(self, pc_config, text_config):
        super().__init__()

        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        text_model, _ = MODEL_REGISTRY.get(text_config.name)
        embed_width = pc_meta.get('embed_length', lambda m : m)(pc_config.m)

        self.pc_encoder = pc_model(**pc_config)
        self.text_encoder = text_model(**text_config)
        self.text_linear = nn.Linear(text_config.width, embed_width)
        self.linear = nn.Linear(embed_width, NUM_CLASSES)

    def forward(self, x, istrain=False):
        if istrain:
            pc_input, (text, has_text) = x
            batch_offsets = pc_input.batch_offsets

            if has_text.size(0) > 0:
                BText, NumText, Length = text.size()
                text_feats = self.text_encoder(text.view(-1, Length), as_dict=True)['x'].view(BText, NumText, -1)
                text_feats = self.text_linear(text_feats)
            else:
                text_feats = -1

            out_feats = self.pc_encoder(pc_input) # B * NumPts, C

            B = len(batch_offsets) - 1
            global_feats = []
            for idx in range(B):
                global_feats.append(torch.mean(out_feats[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
            global_feats = torch.stack(global_feats)
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
        # self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(embed_width, NUM_CLASSES)

    def forward(self, x, istrain=False):
        if istrain: x = x[0]

        out_feats, point_feats = self.pc_encoder(x, istrain)
        # out_feats = self.dropout(out_feats)
        pseudo_class=self.linear(point_feats)
        global_logits = self.linear(out_feats)

        if istrain: global_logits = global_logits, point_feats,pseudo_class
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
            pc_input, _ = x
            batch_offsets = pc_input.batch_offsets
            out_feats = self.pc_encoder(pc_input) # B * NumPts, C
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