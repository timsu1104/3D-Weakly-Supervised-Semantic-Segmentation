import torch
import torch.nn as nn

class Pointnet2DetHead(nn.Module):
    """ Segmentation head for pointnet++ """
    def __init__(self, R, C, suffix='') -> None:
        super().__init__()
        self.suffix = suffix
        self.linear_cls = nn.Linear(R, C + 1)
        self.linear_obj = nn.Linear(R, C + 1)
        self.linear_det = nn.Linear(R, C + 1)
        
    def RoI_Pool(self, proposals, end_points=None):
        seg_feats = end_points['sem_seg_feat'+self.suffix]
        feats_x, feats_y, feats_z = end_points['input_xyz'].T
        RoI_feats = []
        for p in proposals:
            x, y, z, w, h, l = p
            x_min = x - w * 0.5
            x_max = x + w * 0.5
            y_min = y - h * 0.5
            y_max = y + h * 0.5
            z_min = z - l * 0.5
            z_max = z + l * 0.5
            inds = torch.where((feats_x>=x_min)*(feats_x<=x_max)*(feats_y>=y_min)*(feats_y<=y_max)*(feats_z>=z_min)*(feats_z<=z_max), 1, 0)
            seg_feats_within_RoI = seg_feats[inds]
            RoI_feats.append(torch.mean(seg_feats_within_RoI, dim=0))

        RoI_feats = torch.vstack(RoI_feats)
        return RoI_feats
    
    def forward(self, proposals, end_points=None):
        RoI_feats = self.RoI_Pool(proposals, end_points)
        cls_logits = self.linear_cls(RoI_feats)
        obj_logits = self.linear_obj(RoI_feats)
        cls_p = torch.softmax(cls_logits, dim=1)
        obj_p = torch.sofrmax(obj_logits, dim=0)
        U_det = cls_p * obj_p
        return U_det

    def supervised_det(self, proposals, end_points=None):
        RoI_feats = self.RoI_Pool(proposals, end_points)
        S_det = self.linear_det(RoI_feats)
        return S_det