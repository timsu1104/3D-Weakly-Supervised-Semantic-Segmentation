import torch
import torch.nn as nn

from dataset.data import dimension, full_scale
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MultiLabelContrastive(nn.Module):
    def __init__(self, pc_config, text_config):
        super().__init__()

        m = pc_config.m
        residual_blocks=pc_config.residual_blocks
        block_reps = pc_config.block_reps

        width = text_config.width
        vocab_size = text_config.vocab_size
        context_length = text_config.context_length
        layers = text_config.layers

        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        text_model, _ = MODEL_REGISTRY.get(text_config.name)
        embed_width = pc_meta.get('embed_length', lambda m : m)(m)

        self.pc_encoder = pc_model(m, dimension, full_scale, block_reps, residual_blocks)
        self.text_encoder = text_model(context_length, width, layers, vocab_size)
        self.text_linear = nn.Linear(width, embed_width)
        self.linear = nn.Linear(embed_width, 20)
    def forward(self, x, istrain=False):
        if istrain:
            (coords, feats, batch_offsets), (text, has_text) = x

            if has_text.size(0) > 0:
                BText, NumText, Length = text.size()
                text_feats = self.text_encoder(text.view(-1, Length), as_dict=True)['x'].view(BText, NumText, -1)
                text_feats = self.text_linear(text_feats)
            else:
                text_feats = -1

            out_feats = self.pc_encoder([coords, feats]) # B * NumPts, C
            # print(out_feats.size())

            # batch_offsets = x[-1]
            # print(batch_offsets)
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

        m = pc_config.m
        residual_blocks=pc_config.residual_blocks
        block_reps = pc_config.block_reps

        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        embed_width = pc_meta.get('embed_length', lambda m : m)(m)

        self.pc_encoder = pc_model(m, dimension, full_scale, block_reps, residual_blocks)
        self.linear = nn.Linear(embed_width, 20)

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

        m = pc_config.m
        residual_blocks=pc_config.residual_blocks
        block_reps = pc_config.block_reps

        pc_model, pc_meta = MODEL_REGISTRY.get(pc_config.name)
        embed_width = pc_meta.get('embed_length', lambda m : m)(m)

        self.pc_encoder = pc_model(m, dimension, full_scale, block_reps, residual_blocks)
        self.linear = nn.Linear(embed_width, 20)

    def forward(self, x, istrain=False):
        if istrain:
            (coords, feats, batch_offsets), _ = x
            out_feats = self.pc_encoder([coords, feats]) # B * NumPts, C
            global_logits=self.linear(out_feats)
            global_logits = global_logits, None
        else:
            out_feats = self.pc_encoder(x) 
            global_logits=self.linear(out_feats)

        return global_logits