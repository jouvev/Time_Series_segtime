from src.MSSLSTM import MSSLSTM
import torch
from src.stepwise import Stepwise
from encoder import Encoder
from decoder import Decoder
from torch import dropout, nn


class Segtime(nn.Module):
    def __init__(self,input_size,k_list,h_size_list,num_classes,latent_size,resoultion=1,p_drop=0.2):
        super(Segtime,self).__init__()
        self.mss = MSSLSTM(input_size,k_list,h_size_list)
        self.dropout = nn.Dropout(p_drop)
        self.encoder = Encoder()
        self.decoder = Decoder(input_size,num_classes,latent_size*2,int(latent_size*1.5))
        self.stepwise = Stepwise(sum(h_size_list)+num_classes,num_classes,resoultion)
        
    def forward(self,x):
        out_mss = self.mss(x).permute(0,2,1)# car conv1d B*C*L
        out_enc, low_f = self.encoder(x.permute(0,2,1))
        out_dec = self.decoder(low_f,out_enc)
        out = torch.cat((out_mss,out_dec),dim=1)
        out = self.dropout(out)
        out = self.stepwise(out)
        return out
        