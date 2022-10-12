import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.data import NUM_CLASSES
from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class NaiiveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_size = label_size = NUM_CLASS

        self.conv=nn.Sequential(
            nn.Conv2d(1,32,3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            ConvBNAct(32,32,3,stride=2,padding=1,activation=nn.LeakyReLU(0.2)),#14
            nn.Dropout2d(0.3),
            ConvBNAct(32,64,3,stride=2,padding=1,activation=nn.LeakyReLU(0.2)),#7
            nn.Dropout2d(0.3),
            ConvBNAct(64,128,3,stride=2,padding=1,activation=nn.LeakyReLU(0.2)),#4
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1=LinDropAct(128+self.label_size,256,activation=nn.LeakyReLU(0.2))
        self.fc2=nn.Linear(256,1)


    def forward(self, img, label):

        l2=F.one_hot(label,num_classes=self.label_size).float().to(device)
        x1=self.conv(img)
        x1=x1.view(label.shape[0],128)
        x1=torch.cat((x1,l2),dim=1)
        x1=self.fc1(x1)
        x1=self.fc2(x1)
        x1=nn.Sigmoid()(x1)
        return x1


