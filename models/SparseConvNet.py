from typing import List
import torch
from torch import nn
import sparseconvnet as scn

from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register(embed_length=lambda m: m)
class SparseConvUNet(nn.Module):
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

def FullyConvolutionalNetEncoder(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2], MP_freq=3):
    """
    Fully-convolutional style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.FullyConvolutionalNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(256, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
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
            m.add(scn.Sequential(
                        BatchNormWithMaybeMP(nPlanes[0], MP_count=((MP_count+1) % MP_freq == 0)),
                        scn.BatchNormReLU(nPlanes[0]),
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1], downsample[0], downsample[1], False), 
                        U(nPlanes[1:], MP_count=MP_count+1)
                    ))
        return m
    m = U(nPlanes)
    return m

@MODEL_REGISTRY.register(embed_length=lambda m: 7 * (7+1) * m // 2)
class SparseConvFCNet(nn.Module):
    def __init__(self, m, dimension, full_scale, block_reps, residual_blocks, depth:int=7):
        super().__init__()
        self.encoder = scn.Sequential(
            scn.InputLayer(dimension, full_scale, mode=4),
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False),
            scn.FullyConvolutionalNet(
                dimension, 
                block_reps, 
                [(i+1) * m for i in range(depth)], 
                residual_blocks,
                ),
            scn.BatchNormReLU(depth * (depth+1) * m // 2),
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
        out_feats: torch.Tensor, (B * N, m)
        """
        coords, feats = x
        assert coords.size(0) == feats.size(0), f"Coords and feats not aligned! coords's batchsize is {coords.size(0)} while feats' is {feats.size(0)}. "

        out_feats = self.encoder(x)
        return out_feats