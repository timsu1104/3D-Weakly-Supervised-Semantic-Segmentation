from typing import List
from numpy import isin
import torch
from torch import nn
import sparseconvnet as scn
from easydict import EasyDict as edict

from utils.registry import MODEL_REGISTRY

class SparseConvBase_(nn.Module):
    """
    To inherit this class, getEncoder must be redefined, while encode and postProcessing is optional. 
    """
    def getEncoder(self, *args, **kwarg):
        return None

    def encode(self, input: List[torch.Tensor]):
        return self.encoder(input)
    
    def postProcessing(self, out_feats: torch.Tensor, batch_offsets: list):
        B = len(batch_offsets) - 1
        global_feats = []
        for idx in range(B):
            global_feats.append(torch.mean(out_feats[batch_offsets[idx] : batch_offsets[idx+1]], dim=0))
        global_feats = torch.stack(global_feats)
        return global_feats

    def __init__(self, name:str, *args, **kwarg):
        super().__init__()
        assert name == self.__class__.__name__
        self.encoder = self.getEncoder(*args, **kwarg)
    
    
    def forward(self, x: edict, istrain=False):
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
        assert isinstance(x, edict), f"batch data type unsupported. Expected EasyDict, got {type(x)}. "
        coords = x.coords
        feats = x.feature
        assert coords.size(0) == feats.size(0), f"Coords and feats not aligned! coords's batchsize is {coords.size(0)} while feats' is {feats.size(0)}. "

        out_feats = self.encode([coords, feats])
        if istrain:
            out_feats = self.postProcessing(out_feats, x.batch_offsets)
        
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

@MODEL_REGISTRY.register(embed_length=lambda m: sum([m, 64, 128, 192, 256]))
class SparseConvFCNetNarrow(SparseConvBase_):
    def getEncoder(self, m, dimension, full_scale, block_reps, residual_blocks, nPlanes:List[int] = [64, 128, 192, 256], downsample=[2, 2]):
        return scn.Sequential(
            scn.InputLayer(dimension, full_scale, mode=4),
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False),
            scn.FullyConvolutionalNet(
                dimension, 
                block_reps, 
                [m] + nPlanes, 
                residual_blocks,
                downsample=downsample
                ),
            scn.BatchNormReLU(m + sum(nPlanes)),
            scn.OutputLayer(dimension)
        )
        
@MODEL_REGISTRY.register(embed_length=lambda m: 256)
class SparseConvFCNetDirectUpPool(SparseConvBase_):

    def FCNEncoder(self, dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2]):
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
        def U(nPlanes): #Recursive function
            m = scn.Sequential()
            if len(nPlanes) == 1:
                for _ in range(reps):
                    block(m, nPlanes[0], nPlanes[0])
            else:
                m = scn.Sequential()
                for _ in range(reps):
                    block(m, nPlanes[0], nPlanes[0])
                m.add(
                    scn.Sequential().add(
                        scn.BatchNormReLU(nPlanes[0])).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        U(nPlanes[1:])).add(
                        scn.UnPooling(dimension, downsample[0], downsample[1])))
            return m
        m = U(nPlanes)
        return m

    def getEncoder(self, m, dimension, full_scale, block_reps, residual_blocks, nPlanes:List[int] = [64, 128, 192, 256], downsample=[2, 2]):
        return scn.Sequential(
                    scn.InputLayer(dimension, full_scale, mode=4),
                    scn.SubmanifoldConvolution(dimension, 3, m, 3, False),
                    self.FCNEncoder(
                        dimension, 
                        block_reps, 
                        [m] + nPlanes, 
                        residual_blocks,
                        downsample=downsample
                        ),
                    scn.BatchNormReLU(nPlanes[-1]),
                    scn.OutputLayer(dimension)
                )