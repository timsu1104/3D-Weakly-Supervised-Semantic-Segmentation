from typing import List
import torch
from torch import nn
import sparseconvnet as scn
import warnings
warnings.filterwarnings("ignore")

def show_size(name: str, x: scn.SparseConvNetTensor) -> None:
    print(name)
    print("Feature size:", x.features.size())
    print("Space size:", x.spatial_size)

# Sample data, 2 point clouds with 1000 points in each in 3d space
sample_data = torch.load('dataset/ScanNet/train_processed/scene0694_00_vh_clean_2.pth')
coords, color, _ = sample_data
coords = torch.from_numpy(coords)
color = torch.from_numpy(color)

# normalize
coords -= torch.min(coords, dim=0)[0]
coords /= torch.max(coords, dim=0)[0] - torch.min(coords, dim=0)[0]
color -= torch.min(color, dim=0)[0]
color /= torch.max(color, dim=0)[0] - torch.min(color, dim=0)[0]
coords *= 4096 # [0, 4096]
color = 2 * color - 1 # [-1, 1]

# batch_inds = torch.tile(torch.tensor([[0], [1]]), (100000, 1))
batch_inds = torch.zeros((coords.size(0), 1))
coords = torch.cat([coords, batch_inds], dim=-1)
# coords = torch.repeat_interleave(coords[None, :], 3, dim=0)
# color = torch.repeat_interleave(color[None, :], 3, dim=0)
# print(coords.shape, color.shape)

### input
"""
Input Layer.

scn.InputLayer(dimension, spatial_size, mode=3)

dimension: dimensions of the space
spatial_size: the length of the space (is a cube actually)
mode: how to deal with duplicates 
1 means last-occured, 2 means first occured, 3 means sum, 4 means average. 
"""
input_layer = scn.InputLayer(3, 4096, mode=4)
input_tensor = input_layer([coords, color])
show_size('input', input_tensor)

### convolution
"""
Convolution Layer. Two types.

scn.SubmanifoldConvolution(dimension, nIn, nOut, filter_size, bias, groups=1) - same size
scn.Convolution(dimension, nIn, nOut, filter_size, filter_stride, bias, groups=1)

dimension: dimensions of the space.
nIn, nOut: input and output channels.
filter_size: kernel
filter_stride: stride
bias: whether contains bias term
"""
width = 16
SubMConv_layer = scn.SubmanifoldConvolution(3, 3, width, 3, False)
Middle_SubMConv_layer = scn.SubmanifoldConvolution(3, width, width, 3, False)
Conv_layer = scn.Convolution(3, width, width, 2, 2, False)
submconv = SubMConv_layer(input_tensor)
submconv = Middle_SubMConv_layer(submconv)
submconv = Middle_SubMConv_layer(submconv)
show_size('submconv', submconv)
conv = Conv_layer(submconv)
show_size('conv', conv)

### Pooling Layer
"""
scn.MaxPooling(dimension, pool_size, pool_stride)
"""
MaxPool_layer = scn.MaxPooling(3, 4, 4)
maxpool = MaxPool_layer(conv)
show_size('maxpool', maxpool)

dimension = 3
full_scale = 4096
reps = 2 #Conv block repetition factor
m = 32 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m] #UNet number of features per level
sparse_fcn = scn.Sequential().add(
           scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
           scn.FullyConvolutionalNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2,2])).add(
           scn.BatchNormReLU(sum(nPlanes)))

sparse_unet = scn.Sequential().add(
           scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
               scn.UNet(dimension, 1, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks=False)).add(
           scn.BatchNormReLU(m))

# FCN_feats = sparse_fcn(input_tensor)
# show_size('FCN', FCN_feats)

# UNET_feats = sparse_unet(input_tensor)
# show_size('UNET', UNET_feats)


def UNetEncoder(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2], leakiness=0, n_input_planes=-1):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
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
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
    def Uencoder(nPlanes,n_input_planes=-1): #Recursive function
        m = scn.Sequential()
        for i in range(reps):
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
            n_input_planes=-1
        if len(nPlanes) > 1:
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        Uencoder(nPlanes[1:]))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
        return m
    m = Uencoder(nPlanes,n_input_planes)
    return m

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
            m.add(
                scn.Sequential().add(
                    scn.BatchNormReLU(nPlanes[0])).add(
                    scn.Convolution(dimension, nPlanes[0], nPlanes[1], downsample[0], downsample[1], False)).add(
                    U(nPlanes[1:])).add(
                    scn.UnPooling(dimension, downsample[0], downsample[1]))
            )
        return m
    m = U(nPlanes)
    return m

sparse_fcn_encoder = scn.Sequential().add(
           scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
           FullyConvolutionalNetEncoder(dimension, reps, nPlanes, residual_blocks=False, downsample=[2,2], MP_freq=4)).add(
            scn.BatchNormReLU(nPlanes[-1])
            )

FCN_feats = sparse_fcn_encoder(input_tensor)
show_size('FCN3D', FCN_feats)

from models.SparseConvNet import SparseConvFCNetDirectUpPool
from easydict import EasyDict as edict
x = edict({
    'coords': coords,
    'feature': color
})
test_module = SparseConvFCNetDirectUpPool('SparseConvFCNetDirectUpPool', m, dimension, full_scale, 1, False, downsample=[4, 4])
print(test_module(x).size())