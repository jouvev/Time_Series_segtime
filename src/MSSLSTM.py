import torch
from torch import nn
from src.SkipLSTM import SkipLSTM

class MSSLSTM(nn.Module):
    def __init__(self,input_size,k_list,h_size_list):
        super(MSSLSTM, self).__init__()
        self.skiplstms = nn.ModuleList([SkipLSTM(input_size,hidden_size=h_size_list[i],k=k_list[i]) for i in range(len(k_list))])
        
    def forward(self,x):
        res = []
        for lstm in self.skiplstms:
            res.append(lstm(x))
        return torch.cat(res,dim=2)
        