from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
class Pseudo_Images(Dataset):
    def __init__(self, root_dir, cls_lst,valid_cls,img_type='mask',transform=None):
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
        for ind,cls_name in enumerate(cls_lst):
            if valid_cls[ind]:
                cls_pth=os.path.join(root_dir,cls_name,img_type)
                for file in os.listdir(cls_pth):
                    self.img_lst.append((os.path.join(cls_pth,file),ind))
    def __len__(self):
        return len(self.img_lst)
    def __getitem__(self,idx):
        image=Image.open(self.img_lst[idx][0]).convert('L')
        if self.transform:
            #print(type(image))
            image=self.transform(image)
        return (image,self.img_lst[idx][1])
if __name__=='__main__':
    cls_lst=['wall','floor','cabinet','bed','chairs','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
    valid_cls=[False,False,  False    ,False,True ,False,   False,  False, False,   False,      False,    False,    False,  False,   False,          False,          False,    False,   False,  False]
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),#TODO:decide a size to cut
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    #TODO: I think some tricky normalize needs to be applied, like 1->0.999, 0->0.0001, or other?
    ])
    pseudo_dataset=Pseudo_Images('./pseudo_images/',cls_lst,valid_cls,img_type='mask',transform=transforms.ToTensor())
    print(pseudo_dataset[0])

                 
                