from typing import List
import torch
from torch import nn
import sparseconvnet as scn

from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class SparseConvNet(nn.Module):
    def __init__(self, m, dimension, full_scale, block_reps, residual_blocks):
        super().__init__()
        self.encoder = scn.Sequential(
            scn.InputLayer(dimension, full_scale, mode=4),
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False),
            scn.UNet(
                dimension, 
                block_reps, 
                [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], 
                residual_blocks
                ),
            scn.BatchNormReLU(m),
            scn.OutputLayer(dimension)
        )
    
    def forward(self, x: List[torch.Tensor]):
        """
        Parameters
        -------------
        x: [coords, feats], 
            coords: (B * N, 3)
            feats: (B * N, C)

        Return
        -----------
        out_feats: torch.Tensor, (B, N, m)
        """
        coords, feats = x
        assert coords.size(0) == feats.size(0), f"Coords and feats not aligned! coords's batchsize is {coords.size(0)} while feats' is {feats.size(0)}. "

        out_feats = self.encoder(x)
        return out_feats