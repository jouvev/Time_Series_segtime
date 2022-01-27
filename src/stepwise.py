from torch import nn
from torch.nn.functional import interpolate

class Stepwise(nn.Module):
    def __init__(self,n_in,num_classes,resoultion=1):
        super(Stepwise,self).__init__()
        self.conv = nn.Conv1d(n_in,num_classes,kernel_size=1,stride=1)
        self.averagepool = nn.AvgPool1d(kernel_size=(resoultion))
        
    def forward(self,x):
        out = self.conv(x)
        out = self.averagepool(out)#depend de la frequence de changement
        out = interpolate(out,size=x.size(-1))#up sampling pour remettre à la longuer de la chaine de base pour la segmentation
        return out
        