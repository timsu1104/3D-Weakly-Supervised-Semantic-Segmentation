import torch
import torch.nn as nn
from wypr_backbone.backbone_pointnet2 import Pointnet2Backbone
from det_net import Pointnet2DetHead
from seg_net import Pointnet2SegHead
from wypr_utils.pseudo_util import *

class SegDetNet(nn.Module):
    def __init__(self, R, C, input_feature_dim=0, num_class=0, suffix="") -> None:
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.seg_head1 = Pointnet2SegHead(input_feature_dim, num_class, suffix)
        self.seg_head2 = Pointnet2SegHead(input_feature_dim, num_class, suffix)
        self.det_head = Pointnet2DetHead(R, C, suffix)

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None, proposals=None, scene_label=None, p1=0.0, p2=0.0,tao=0.0):
        end_points = self.backbone(pointcloud)
        end_points = self.seg_head1(end_points, classification=True)
        U_seg = end_points['sem_seg_pred'+self.suffix]
        # proposals = self.get_proposals()
        U_det = self.det_head(proposals, end_points)

        # get pseudo labels
        seg_labels = get_pseudo_labels_seg(scene_label, U_seg, proposals, tao, p2)
        det_labels = get_pseudo_labels_det(scene_label, U_seg, p1)

        # seg and det logits without self-training
        # self.end_points = end_points

        # scene_label: torch.Tensor, U_det: torch.Tensor, proposals: torch.Tensor, tao: float, p2: float
        S_seg = self.seg_head2(self.end_points, classification=True)
        RoI_feats = self.detHead.RoI_Pool(proposals, end_points)
        S_det = self.det_head.linear_det(RoI_feats)

        return S_seg, S_det, seg_labels, det_labels

    # def get_pseudo_seg(self):
    #     return
    
    # def get_pseudo_det(self):
    #     return

    # def pseudo_supervised(self):
    #     proposals = self.get_proposals()
    #     S_seg = self.seg_head2(self.end_points, classification=True)
    #     S_det = self.det_head.supervised_det(proposals)
    #     return S_seg, S_det

    # def get_proposals(self):
    #     torch.load()
    #     return