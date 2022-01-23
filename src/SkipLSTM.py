from torch import nn
import torch.nn.functional as F

class SkipLSTM(nn.Module):
    
    def __init__(self,input_size, hidden_size, k=1):
        """[summary]

        Args:
            input_size (int): size of the input
            hidden_size (int): size of the hidden state
            k (int, optional): skipping factor. Defaults to 1.
        """
        super(SkipLSTM, self).__init__()
        self.k = k
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True)
        
    def forward(self,x):
        """

        Args:
            x (3D tensor): input sequence, B x L x input_size
        """
        
        skip_x = x[:,::self.k,:]
        out,_ = self.lstm(skip_x)
        # up sampling
        out = F.interpolate(out.permute(0,2,1), size=x.size()[1], mode='linear', align_corners=True)
        
        return out.permute(0,2,1)