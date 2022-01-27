from src.MSSLSTM import MSSLSTM
from src.stepwise import Stepwise
from torch import dropout, nn

class Segtime(nn.Module):
    def __init__(self,input_size,k_list,h_size_list,num_classes,resoultion=1,p_drop=0.2):
        super(Segtime,self).__init__()
        self.mss = MSSLSTM(input_size,k_list,h_size_list)
        self.dropout = nn.Dropout(p_drop)
        self.stepwise = Stepwise(sum(h_size_list),num_classes,resoultion)
        
    def forward(self,x):
        out = self.mss(x)
        out = self.dropout(out).permute(0,2,1) # car conv1d B*C*L
        out = self.stepwise(out)
        return out
        