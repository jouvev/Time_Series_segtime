from torch import nn
import torch
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self,in_x,num_classed,in_lowf,out_lowf):
        super(Decoder,self).__init__()
        self.conv_for_lowf = nn.Conv1d(in_lowf,out_lowf,1,bias=False)
        self.batchnorm_lowf = nn.BatchNorm1d(out_lowf)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_x+out_lowf, in_x, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_x),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_x, in_x, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_x),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_x, num_classed, kernel_size=1, stride=1))
        
        
    def forward(self,lowf,x):
        low_level = self.conv_for_lowf(lowf)
        low_level = F.relu(self.batchnorm_lowf(low_level))
        x = F.interpolate(x, size=low_level.size()[2:], mode='linear', align_corners=True)
        x = torch.cat((x, low_level), dim=1)
        x = self.conv(x)
        
        return x