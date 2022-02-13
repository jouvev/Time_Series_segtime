from src.MSSLSTM import MSSLSTM
import torch
from src.stepwise import Stepwise
from src.encoder import Encoder
from src.decoder import Decoder
from torch import dropout, nn
from torch.nn import functional as F


class Segtime(nn.Module):
    def __init__(self,input_size,k_list,h_size_list,num_classes,latent_size,amsp_channel,output_stride=8,resoultion=1,p_drop=0.2):
        """Module segtime

        Args:
            input_size (int): nombre de chanel en input
            k_list (list[int]): list des pas pour mss-lstm
            h_size_list (list[int]): list des tailles des h pour mss-lstm
            num_classes (int): nombre de classe
            latent_size (int): nombre de filtre pour encoder
            amsp_channel (int): nombre de filtre pour aspm
            output_stride (int, optional): [description]. Defaults to 8.
            resoultion (int, optional): [description]. Defaults to 1.
            p_drop (float, optional): [description]. Defaults to 0.2.
        """
        super(Segtime,self).__init__()
        self.mss = MSSLSTM(input_size,k_list,h_size_list)
        self.dropout = nn.Dropout(p_drop)
        self.encoder = Encoder(input_size,latent_size,amsp_channel,output_stride)
        self.decoder = Decoder(amsp_channel,num_classes,latent_size*2,int(latent_size*1.5))
        self.stepwise = Stepwise(sum(h_size_list)+num_classes,num_classes,resoultion)
        
    def forward(self,x):
        out_mss = self.mss(x).permute(0,2,1)# car conv1d B*C*L
        out_enc, low_f = self.encoder(x.permute(0,2,1))
        out_dec = self.decoder(low_f,out_enc)
        out_dec = F.interpolate(out_dec, size=x.size()[1], mode='linear', align_corners=True)
        out = torch.cat((out_mss,out_dec),dim=1)
        out = self.dropout(out)
        out = self.stepwise(out)
        return out
        