from typing import List
import torch
from torch import nn
import sparseconvnet as scn

from utils.registry import MODEL_REGISTRY

class SparseConvBase_(nn.Module):
    def getEncoder(self, *args, **kwarg):
        return None

    def __init__(self, *args, **kwarg):
        super().__init__()
        self.encoder = self.getEncoder(*args, **kwarg)
    
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

@MODEL_REGISTRY.register(embed_length=lambda m: m)
class SparseConvUNet(SparseConvBase_):
    def getEncoder(self, m, dimension, full_scale, block_reps, residual_blocks):
        return scn.Sequential(
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

@MODEL_REGISTRY.register(embed_length=lambda m: 7 * m)
class SparseConvFCNetEncoder(SparseConvBase_):
    def FullyConvolutionalNetEncoder(self, dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2], MP_freq=3):
        """
        Fully-convolutional style network without concatenation with VGG or ResNet-style blocks.
        """
        def block(m, a, b):
            if residual_blocks: #ResNet style blocks
                m.add(scn.ConcatTable()
                    .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                    .add(scn.Sequential()
                        .add(scn.BatchNormReLU(a))
                        .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                        .add(scn.BatchNormReLU(b))
                        .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
                ).add(scn.AddTable())
            else: #VGG style blocks
                m.add(scn.Sequential()
                    .add(scn.BatchNormReLU(a))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))

        def BatchNormWithMaybeMP(channels, MP_count):
            if MP_count:
                return scn.Sequential(
                    scn.BatchNormReLU(channels),
                    scn.MaxPooling(dimension, 2, 2)
                )
            else:
                return scn.BatchNormReLU(channels)

        def U(nPlanes, MP_count:int=0): #Recursive function
            m = scn.Sequential()
            if len(nPlanes) == 1:
                for _ in range(reps):
                    block(m, nPlanes[0], nPlanes[0])
            else:
                m = scn.Sequential()
                for _ in range(reps):
                    block(m, nPlanes[0], nPlanes[0])
                m.add(
                    scn.Sequential(
                        scn.BatchNormReLU(nPlanes[0]),
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1], downsample[0], downsample[1], False),
                        U(nPlanes[1:]),
                        # scn.UnPooling(dimension, downsample[0], downsample[1])
                    )
                )
            return m
        m = U(nPlanes)
        return m

    def getEncoder(self, m, dimension, full_scale, block_reps, residual_blocks, depth:int=7, downsample=[2, 2]):
        return scn.Sequential(
            scn.InputLayer(dimension, full_scale, mode=4),
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False),
            self.FullyConvolutionalNetEncoder(
                dimension, 
                block_reps, 
                [(i+1) * m for i in range(depth)], 
                residual_blocks,
                downsample=downsample
                ),
            scn.UnPooling(dimension, downsample[0] ** (depth-1), downsample[1] ** (depth - 1)),
            scn.BatchNormReLU(depth * m),
            scn.OutputLayer(dimension)
        )

@MODEL_REGISTRY.register(embed_length=lambda m: 7 * (7+1) * m // 2)
class SparseConvFCNet(SparseConvBase_):
    def getEncoder(self, m, dimension, full_scale, block_reps, residual_blocks, depth:int=7, downsample=[2, 2]):
        return scn.Sequential(
            scn.InputLayer(dimension, full_scale, mode=4),
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False),
            scn.FullyConvolutionalNet(
                dimension, 
                block_reps, 
                [(i+1) * m for i in range(depth)], 
                residual_blocks,
                downsample=downsample
                ),
            scn.BatchNormReLU(depth * (depth+1) * m // 2),
            scn.OutputLayer(dimension)
        )