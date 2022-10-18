import torch
import torch.nn as nn
from models import utils
from dataset.data import NUM_CLASSES
from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class PointSeg(nn.Module):#only use feature and a small fcn
    def __init__(self):
        super().__init__()
        self.label_size = label_size = NUM_CLASS

        self.fc1=LinDropAct(NUM_CLASS,256,activation=nn.LeakyReLU(0.2))
        self.fc2=nn.Linear(256,NUM_CLASS)

    def forward(self, loc, feats):
        x=self.fc1(feats)
        mask=self.fc2(x)
        return loc,mask
        


