from torch.utils.data import Dataset
from os import listdir
from pandas import read_csv
import torch

class OpportunityDS(Dataset):
    def __init__(self,dirPath):
        self.filename = [f for f in listdir(dirPath) if f[-4:]=='.dat']
        self.dict_loco = {0:0,101:1,102:2,104:3,105:4}
        self.dict_gesture = {0:0,506616:1,506617:2,504616:3,504617:4,
                             506620:5,504620:6,506605:7,504605:8,
                             506619:9,504619:10,506611:11,504611:12,
                             506608:13,504608:14,508612:15,507621:16,
                             505606:17}
        
    def __len__(self):
        return 1#len(self.filename)
    
    def __getitem__(self,i):
        df = read_csv('data/Opportunity/'+self.filename[i],delimiter=' +',engine='python')
        x = torch.tensor(df.values[:,1:114])
        yloco = torch.tensor([self.dict_loco[int(y)] for y in df.values[:,114]])
        ygesture = torch.tensor([self.dict_gesture[int(y)] for y in df.values[:,115]])
        return torch.nan_to_num(x.float()),yloco,ygesture