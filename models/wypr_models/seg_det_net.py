import torch
import torch.nn as nn
from det_net import Pointnet2DetHead
from seg_net import Pointnet2SegHead

class SegDetNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seg_head = Pointnet2SegHead()
        self.det_head = Pointnet2DetHead()