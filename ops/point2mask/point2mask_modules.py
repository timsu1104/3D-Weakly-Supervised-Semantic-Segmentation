# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Point2Mask layers.
Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
Extended with the following:
1. Uniform sampling in each local region (sample_uniformly)
2. Return sampled points indices to support votenet.
'''
import torch
import torch.nn as nn
import math

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import point2mask_utils
from typing import Dict, Tuple, Union

def projection(pc, theta, phi, r=1):
    """
    Project a point cloud onto a plane.

    Parameters
    --------------
    pc: (B, N, 3)
    theta: torch.Tensor, (M, )
    phi: torch.Tensor, (M, )
    r: torch.Tensor, (M, ) or broadcastable to (M, )

    Return
    --------------
    proj: torch.Tensor, (B, M, N, 2)
    """
    sint, cost = torch.sin(theta), torch.cos(theta)
    sinp, cosp = torch.sin(phi), torch.cos(phi)
    U = torch.stack([- sint, cost, torch.zeros_like(theta)], dim=-1) # (M, 3)
    V = torch.stack([cost * sinp, sint * sinp, cosp], dim=-1) # (M, 3)
    basis = torch.stack([U, V], dim=-1) # (M, 3, 2)
    Center = torch.stack([cost * cosp, sint * cosp, sinp], dim=-1) * r # (M, 3)
    coords = torch.squeeze((pc[:, None, :] - Center[None, :, None]).unsqueeze(3) @ basis[None, :, None], dim=-2)
    return coords

class _Point2MaskModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None

    def forward(
        self, 
        coords: torch.Tensor, 
        features: torch.Tensor, 
        res: Union[int, Tuple[int, int]],
        points_num: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor
            (B, N, 2) tensor of the projected coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the segmentation scores
        res : int or Tuple[int, int]
            resolution (H, W) of the mask
        points_num : torch.Tensor
            (B, ) tensor of points number of each instance (paddings MUST be mean value)

        Returns
        -------
        mask : torch.Tensor
            (B, H, W, 2) tensor of the generated mask
        """

        device = coords.device

        # To enable box supervision
        # c = features.argmax(-1)
        # for cl, ptnum in zip(c, points_num): assert (cl[:ptnum] == cl[0]).all(), "Pointcloud should belong to same instance"

        if type(res) == tuple:
            H, W = res
        else:
            H = W = res
        p = torch.tensor([[[H, W]]], device=device) # (1, 1, 2)

        # scale to [0, H] x [0, W]
        center = (coords.max(dim=-2, keepdim=True)[0] + coords.min(dim=-2, keepdim=True)[0]).detach() / 2
        scale = torch.clamp(coords.max(dim=-2, keepdim=True)[0] - coords.min(dim=-2, keepdim=True)[0], min=1e-5).detach() / 2
        coords = ((coords - center) / scale + 1) * 0.8 * p / 2 + 0.1 * p # (B, N, 2)
        B, N, _ = coords.size()
        coords = coords.view(-1, N, 2) # (B, N, 2)

        x_grid, y_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        samples = torch.repeat_interleave(torch.stack([x_grid, y_grid], dim=-1).view(-1, 2).unsqueeze(0), B, dim=0).float().to(device) # (B, H*W, 2)

        # reformulate channels
        pts_feats = torch.sort(features)[0][..., -2:]# (B, N, 2)

        new_features = self.groupers(
            coords, 
            samples, 
            pts_feats.transpose(1, 2).contiguous(), 
            points_num # (B, )
        )  # (B, 2, H*W, nsample)

        occupation = torch.sum(new_features != 0., -1)
        empty_inds = occupation == 0.
        occupation[empty_inds] = 1.
        new_features = torch.sum(new_features, -1) # (B, 2, H*W)
        new_features = (new_features / occupation).transpose(1, 2).contiguous() # (B, H*W, 2)
        new_features = torch.softmax(new_features, -1)
        empty_inds = (new_features[..., 0] == new_features[..., 1]).unsqueeze(-1)
        empty_pad = torch.stack([torch.ones_like(occupation, device=device), torch.zeros_like(occupation, device=device)], dim=-1).float()
        mask = torch.masked_scatter(new_features, empty_inds, empty_pad)
        mask = mask.view(B, H, W, 2)

        return mask

class Point2MaskModule(_Point2MaskModuleBase):
    """2D Point set abstraction layer (Multi-view)

    Parameters
    ----------
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    """

    def __init__(
            self,
            *,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = False, 
            sample_uniformly: bool = False
    ):
        super().__init__()
        self.groupers = point2mask_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly)
    
class Point2Mask(nn.Module):
    """3D Point cloud to mask conversion (Multi-view and multi-label, no parameters)

    Parameters
    ----------
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    """
    def __init__(
            self,
            *,
            radius: float,
            nsample: int
    ):
        super().__init__()
        self.mask_generator = Point2MaskModule(radius=radius, nsample=nsample)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, proposals : torch.Tensor, res: Union[int, Tuple[int, int]], theta: torch.Tensor, phi: torch.Tensor, r=1) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the projected coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the semantic segmentation scores
        proposals : torch.Tensor
            (B, N) tensor of the instance label
        res : int or Tuple[int, int]
            resolution (H, W) of the mask
        theta: torch.Tensor
            (M, ) tensor of azimuth
        phi: torch.Tensor
            (M, ) tensor of elevation

        Returns
        -------
        masks : torch.Tensor
            (NumMasks, 2, H, W) tensor of the generated mask
        """

        device = xyz.device
        pc_coords = projection(xyz, theta, phi, r=r) # (B, M, N, 2)
        labels = proposals # (B, N)
        M = theta.size(0)
        C = features.size(-1)

        masks = []

        for coords, label, feats in zip(pc_coords, labels, features):
            """
            coords : (M, N, 2)
            label : (N, )
            feats : (N, C)

            mask : (M * inst_num, H, W, 2)
            """
            labelpool = label.unique()

            input_coords_list = []
            input_features_list = []
            pointnums_list = []

            for l in labelpool:
                cropped_inds = label == l
                input_coords_list.append(coords[:, cropped_inds])
                input_features_list.append(feats[cropped_inds])
                pointnums_list.append(torch.sum(cropped_inds))

            pointnums = torch.stack(pointnums_list, 0)
            max_pt_num = pointnums.max()
            for i, (new_coords, new_features, ptnum) in enumerate(zip(input_coords_list, input_features_list, pointnums)):
                if ptnum == max_pt_num:
                    continue
                assert new_coords.size(1) == new_features.size(0) == ptnum
                
                pad_c = torch.ones((M, max_pt_num - ptnum, 2), device=device) * torch.mean(new_coords, dim=1, keepdim=True)
                pad_f = torch.ones((max_pt_num - ptnum, C), device=device) * torch.mean(new_features, dim=0, keepdim=True)
                input_coords_list[i] = torch.cat([new_coords, pad_c], dim=1)
                input_features_list[i] = torch.cat([new_features, pad_f], dim=0)

            input_coords = torch.stack(input_coords_list).view(-1, max_pt_num, 2) # (NumInst*M, N', 2)
            input_features = torch.stack(input_features_list).repeat_interleave(M, 0) # (NumInst*M, N', C)
            pointnums = pointnums.repeat_interleave(M, 0)

            mask = self.mask_generator(input_coords, input_features, res, pointnums.int()) # (NumInst*M, H, W, 2)
            mask = torch.repeat_interleave(mask[..., 1:], 3, dim=-1) * 255

            masks.append(mask)
        
        masks = torch.cat(masks)
        return masks.permute((0, 3, 1, 2))

class Pixel2Mask(nn.Module):
    """2D image alpha to mask conversion (no parameters)

    Parameters
    ----------
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    """
    def __init__(
            self,
            *,
            radius: float,
            nsample: int
    ):
        super().__init__()
        self.mask_generator = Point2MaskModule(radius=radius, nsample=nsample)
    
    def forward(self, image: torch.IntTensor, res: Union[int, Tuple[int, int]]) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.IntTensor
            (B, H, W, 3) tensor of the alphas in rgb format, should be (255, 255, 255) for foreground or (0, 0, 0) for background
        res : int or Tuple[int, int]
            resolution (H, W) of the mask

        Returns
        -------
        masks : torch.Tensor
            (B, H, W, 2) tensor of the generated mask
        """

        device = image.device
        B = image.size(0)

        assert (torch.unique(image) == torch.tensor([0, 255], device=device)).all() and (image[..., 0] == image[..., 1]).all() and (image[..., 0] == image[..., 2]).all(), "Image format is incorrect."

        input_coords_list = []
        pointnums_list = []

        for img in image:
            coords = torch.nonzero(img[..., 0] == 255).float() + 0.5 # (N, 2)
            input_coords_list.append(coords)
            pointnums_list.append(coords.size(0))

        pointnums = torch.tensor(pointnums_list, device=device) # (B, )
        max_pt_num = pointnums.max().item()
        # print("Padding to", max_pt_num)
        for i, (new_coords, ptnum) in enumerate(zip(input_coords_list, pointnums)):
            if ptnum < max_pt_num:
                pad_c = torch.ones((max_pt_num - ptnum, 2), device=device) * torch.mean(new_coords, dim=0, keepdim=True)
                input_coords_list[i] = torch.cat([new_coords, pad_c], dim=0)

        input_coords = torch.stack(input_coords_list).view(B, max_pt_num, 2) # (B, N', 2)
        input_features = torch.stack([torch.zeros((B, max_pt_num), device=device), torch.ones((B, max_pt_num), device=device)], dim=-1) # (B, N', 2)

        # print(input_coords.size(1), pointnums.int())
        mask = self.mask_generator(input_coords, input_features, res, pointnums.int()) # (B, H, W, 2)
        mask = mask + mask / math.e

        mask = torch.repeat_interleave(mask[..., 1:], 3, dim=-1) * 255

        return mask
    
class Box2Mask(nn.Module):
    """3D Point cloud to mask conversion (Multi-view and multi-label, no parameters)

    Parameters
    ----------
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    """
    def __init__(
            self,
            *,
            radius: float,
            nsample: int
    ):
        super().__init__()
        self.mask_generator = Point2MaskModule(radius=radius, nsample=nsample)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, boxes : torch.Tensor, res: Union[int, Tuple[int, int]], theta: torch.Tensor, phi: torch.Tensor, r=1, box_filter_bar : Dict = None) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the projected coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the semantic segmentation scores
        boxes : torch.Tensor
            (B, M, 6) tensor of the detected boxes
        res : int or Tuple[int, int]
            resolution (H, W) of the mask
        theta: torch.Tensor
            (M, ) tensor of azimuth
        phi: torch.Tensor
            (M, ) tensor of elevation
        box_filter_bar : Dict[String, Float]
            purity_lwbnd : minimum purity
            density_lwbnd : minimum density

        Returns
        -------
        masks : torch.Tensor
            (NumMasks, 3, H, W) tensor of the generated mask
        """

        device = xyz.device
        pc_coords = projection(xyz, theta, phi, r=r) # (B, M, N, 2)
        label = features.argmax(-1)
        M = theta.size(0)
        C = features.size(-1)

        purity_lwbnd = box_filter_bar.get('purity_lwbnd', 0)
        density_lwbnd = box_filter_bar.get('density_lwbnd', 0)

        masks = []

        for coords, boxpool, feats, pc in zip(pc_coords, boxes, features, xyz):
            """
            coords : (M, N, 2)
            boxpool : (NumBoxes, 6) xmin, ymin, zmin, xmax, ymax, zmax
            feats : (N, C)
            pc : (N, 3)

            mask : (M * NumBoxes, H, W, 2)
            """

            input_coords_list = []
            input_features_list = []
            pointnums_list = []

            for box in boxpool:
                cropped_inds = torch.prod(pc <= box[3:], -1) * torch.prod(pc >= box[:3], -1)  # binary mask
                cropped_pc = pc[cropped_inds] # (N', 3)
                cropped_label = label[cropped_inds]
                box_label = torch.mode(cropped_label)[0]
                NumPts = cropped_pc.size(0)
                Volume = torch.prod(box[3:] - box[:3])
                SelPts = torch.sum(cropped_label == box_label)
                purity = NumPts / SelPts
                density = NumPts / Volume
                if purity >= purity_lwbnd and density >= density_lwbnd:
                    input_coords_list.append(coords[:, cropped_inds])
                    input_features_list.append(feats[cropped_inds])
                    pointnums_list.append(torch.sum(cropped_inds))
            
            if len(input_coords_list) == 0:
                continue

            pointnums = torch.stack(pointnums_list, 0)
            max_pt_num = pointnums.max()
            for i, (new_coords, new_features, ptnum) in enumerate(zip(input_coords_list, input_features_list, pointnums)):
                if ptnum == max_pt_num:
                    continue
                assert new_coords.size(1) == new_features.size(0) == ptnum
                
                pad_c = torch.ones((M, max_pt_num - ptnum, 2), device=device) * torch.mean(new_coords, dim=1, keepdim=True)
                pad_f = torch.ones((max_pt_num - ptnum, C), device=device) * torch.mean(new_features, dim=0, keepdim=True)
                input_coords_list[i] = torch.cat([new_coords, pad_c], dim=1)
                input_features_list[i] = torch.cat([new_features, pad_f], dim=0)

            input_coords = torch.stack(input_coords_list).view(-1, max_pt_num, 2) # (NumBoxes*M, N', 2)
            input_features = torch.stack(input_features_list).repeat_interleave(M, 0) # (NumBoxes*M, N', C)
            pointnums = pointnums.repeat_interleave(M, 0)

            mask = self.mask_generator(input_coords, input_features, res, pointnums.int()) # (NumBoxes*M, H, W, 2)
            mask = torch.repeat_interleave(mask[..., 1:], 3, dim=-1) * 255

            masks.append(mask)
        
        masks = torch.cat(masks)
        return masks.permute((0, 3, 1, 2))

if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from torch.autograd import Variable
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    xyz = Variable(torch.randn(4, 200000, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(4, 200000, 20).cuda(), requires_grad=True)

    test_module = Point2Mask(
        radius=4, nsample=3
    )
    test_module.cuda()
    input_ = xyz, xyz_feats, xyz_feats.argmax(-1), 256, torch.zeros(3).cuda(), torch.zeros(3).cuda()
    # print(test_module(*input_))

    for _ in range(1):
        from time import time
        start = time()
        new_features = test_module(*input_)
        end1 = time()
        print("Forward time {}s".format(end1-start))
        new_features[..., 1].backward(
            torch.cuda.FloatTensor(*new_features[..., 1].size()).fill_(1)
        )
        end2 = time()
        print("Backward time {}s".format(end2-end1))
        print(new_features.size())
        print((xyz_feats.grad != 0).any().item())

    test_module = Pixel2Mask(
        radius=4, nsample=3
    )
    images = torch.zeros(4, 256, 256, 3).cuda()
    images[:, 64:-64, 64:-64] = 255
    start = time()
    mask = test_module(images, 256)
    print("new_image_size", mask.size())
    print("TIME {}s".format(time() - start))