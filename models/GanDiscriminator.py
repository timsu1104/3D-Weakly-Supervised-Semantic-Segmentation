import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.data import NUM_CLASSES
from utils.registry import MODEL_REGISTRY
import time
class LinDropAct(nn.Module):
    def __init__(self,in_put,out_put,activation=nn.SiLU,drop_rate=0.25):
        super(LinDropAct, self).__init__()
        self.lin= nn.Linear(in_put,out_put)
        self.act = activation
        self.dropout=nn.Dropout(drop_rate)

    def forward(self, x):
        result = self.lin(x)
        result = self.dropout(result)
        result = self.act(result)

        return result
class ConvBNAct(nn.Module):
    def __init__(self,in_put,out_put,kernel_size=3,stride=1,padding=1,activation=nn.SiLU,momentum=0.1):
        super(ConvBNAct, self).__init__()
        self.conv= nn.Conv2d(in_put,out_put,kernel_size,stride=stride,padding=padding)
        self.act = activation
        self.bn = nn.BatchNorm2d(out_put,momentum=momentum)
    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result

@MODEL_REGISTRY.register()
class NaiiveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_size = label_size = NUM_CLASSES

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
        #self.fc1=LinDropAct(128,256,activation=nn.LeakyReLU(0.2))
        self.fc2=nn.Linear(256,1)


    def forward(self, img, label):
        #print('label')
        #print(img.shape)
        #print(label.shape)
        l2=F.one_hot(label,num_classes=self.label_size).float().to('cuda')
        #s_iter = time.time()
        x1=self.conv(img)
        x1=x1.view(img.shape[0],128)
        x1=torch.cat((x1,l2),dim=1)
        x1=self.fc1(x1)
        x1=self.fc2(x1)
        x1=nn.Sigmoid()(x1)
        #print(time.time()-s_iter)
        return x1


