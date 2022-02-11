from cmath import isnan
from torch.utils.data import Dataset
from os import listdir
import torch
import numpy as np

class OpportunityDS(Dataset):
    def __init__(self,df,lenght=600):
        self.dict_loco = {0:0,101:1,102:2,104:3,105:4}
        self.dict_gesture = {0:0,506616:1,506617:2,504616:3,504617:4,
                            506620:5,504620:6,506605:7,504605:8,
                            506619:9,504619:10,506611:11,504611:12,
                            506608:13,504608:14,508612:15,507621:16,
                            505606:17}
        self.x = torch.tensor(df.values[:,1:114])
        self.x = torch.nan_to_num(self.x.float())
        self.yloco = torch.tensor([self.dict_loco[int(y)] for y in df.values[:,114]])
        self.ygesture = torch.tensor([self.dict_gesture[int(y)] for y in df.values[:,115]])
        self.lenght = lenght
        
    def __len__(self):
        return int(self.x.size(0)/self.lenght)
    
    def __getitem__(self,i):
        return self.x[i*self.lenght:(i+1)*self.lenght],self.yloco[i*self.lenght:(i+1)*self.lenght] ,self.ygesture[i*self.lenght:(i+1)*self.lenght]