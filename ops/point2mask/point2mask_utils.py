# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn

import builtins

try:
    import point2mask_ext as _ext
except ImportError:
    if not getattr(builtins, "__POINT2MASK_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

# from torch.utils.cpp_extension import load
# import glob, os.path as osp

# _this_dir = osp.dirname(osp.abspath(__file__))
# _ext_src_root = "_ext_src"
# _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
#     "{}/src/*.cu".format(_ext_src_root)
# )
# _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
# _ext = load(name='point2mask_ext', sources=_ext_sources, extra_cflags=["-O3"],
#             extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
#             extra_include_paths=[osp.join(_this_dir, _ext_src_root, "include")])

from typing import *

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indices of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz, pointsnum):
        # type: (Any, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 2) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 2) centers of the ball query
        pointsnum : torch.Tensor
            (B, )Number of points of each instance

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indices of the features that form the query balls
        """
        # print(xyz.shape, xyz.min(dim=1)[0], xyz.max(dim=1)[0])
        # print(new_xyz.shape, new_xyz.min(dim=1)[0], new_xyz.max(dim=1)[0])
        inds = _ext.ball_query(new_xyz, xyz, pointsnum, radius, nsample)
        # print(inds)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool, bool, bool, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None, pointsnum=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 2)
        new_xyz : torch.Tensor
            centroids (B, npoint, 2)
        features : torch.Tensor
            Descriptors of the features (B, C, N)
        pointsnum : torch.Tensor
            Number of points of each instance

        Returns
        -------
        new_features : torch.Tensor
            (B, 2 + C, npoint, nsample) tensor
        """
        if pointsnum is None:
            pointsnum = xyz.size(1) * torch.ones(xyz.size(0), device=xyz.device)
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz, pointsnum.contiguous())

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 2, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 2, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)