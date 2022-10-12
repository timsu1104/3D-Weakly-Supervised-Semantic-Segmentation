from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Pseudo_Images(Dataset):
    def __init__(self, root_dir, cls_lst,cls_valid,img_type='mask',transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            cls_lst (list of string): The name of all classes appears (in order).
            cls_valid (list of bool): Whether the class actually used for training. 
            img_type(string): type of image style
        Output:
            Pseudo_Images[i]: (Transformed image,Integer of class)
        """
        #general routine: get image names when initialize, read and transform when require
        self.cls_lst=cls_lst
        self.valid_cls=valid_cls
        self.root_dir = root_dir
        self.transform = transform
        self.img_lst=[]
        for ind,cls_name in cls_list:
            if valid_cls[ind]:
                cls_pth=os.path.join(root_dir,cls_name)
                for file in os.listdir(cls_pth):
                    self.img_lst.append((os.path.join(cls_pth,file),ind))
    def __len__(self):
        return len(self.img_lst)
    def __getitem__(self,idx):
        image=io.imread(self.img_lst[idx][0])
        if self.transform:
            image=self.transform(image)
        return (image,self.img_lst[idx][1])

                 
                