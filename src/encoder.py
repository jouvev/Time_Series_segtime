from torch import nn
from torch.nn import functional as F

class Bottleneck1D(nn.Module):
    expansion = 2
    def __init__(self, in_size,out_size,stride=1,dilation=1,downsample=None):
        super(Bottleneck1D,self).__init__()
        self.conv1 = nn.Conv1d(in_size,out_size,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm1d(out_size)
        self.conv2 = nn.Conv1d(out_size, out_size, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_size)
        self.conv3 = nn.Conv1d(out_size, out_size * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_size * 2)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        
    def forward(self,x):
        res = x 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            res = self.downsample(x)
            
        out += res
        out = F.relu(out)
        
        return out
    
class Res1DNet(nn.Module):
    def __init__(self,in_size,latent_size,nb_blocks,output_stride):
        super().__init__()
        self.conv1 = nn.Conv1d(in_size, latent_size, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm1d(latent_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inplanes = latent_size
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        
        self.layer1 = self._layer(in_size, nb_blocks[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._layer(in_size*2, nb_blocks[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._layer(in_size*4, nb_blocks[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._layer(in_size*8, nb_blocks[3], stride=strides[3], dilation=dilations[3])
        
    def _layer(self,latent_size,nb_block,stride=1,dilation=1):
        layers = []
        layers.append(Bottleneck1D(self.inplanes, Bottleneck1D.expansion*latent_size, stride, dilation))
        for i in range(1, nb_block):
            layers.append(Bottleneck1D(Bottleneck1D.expansion*latent_size, latent_size, dilation=dilation))
            
        return nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat