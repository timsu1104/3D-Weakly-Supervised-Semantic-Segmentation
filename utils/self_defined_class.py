from torch.utils.data import DataLoader, Dataset
import torch

class PointCloudDataset(Dataset):
   def __init__(self, data_list):
       self.data_list = data_list

   def __getitem__(self, index):
       return self.data_list[index]

   def __len__(self):
       return len(self.data_list)