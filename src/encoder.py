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
    

class AMSP(nn.Module):
    def __init__(self, input_channel, output_stride, BatchNorm):
        super(AMSP, self).__init__()
        
        inplanes = min(input_channel*32, 1024)
        amsp_channel = min(256, input_channel*8)
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
            
        self.amsp1 = _AMSPModule(inplanes, amsp_channel, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.amsp2 = _AMSPModule(inplanes, amsp_channel, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.amsp3 = _AMSPModule(inplanes, amsp_channel, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.amsp4 = _AMSPModule(inplanes, amsp_channel, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                             nn.Conv1d(inplanes, amsp_channel, 1, stride=1, bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv1d(amsp_channel*5, amsp_channel, 1, bias=False)
        self.bn1 = BatchNorm(amsp_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.amsp1(x)
        x2 = self.amsp2(x)
        x3 = self.amsp3(x)
        x4 = self.amsp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='linear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
class _AMSPModule(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding, dilation, BatchNorm):
        super(_AMSPModule, self).__init__()
        self.atrous_conv = nn.Conv1d(in_size, out_size, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(out_size)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            else:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
                
class Encoder(nn.Module):
    def __init__(self, input_channel, output_stride=8, channel_last_data=True): #sync_bn=True,
        super(Encoder, self).__init__()
        BatchNorm = nn.BatchNorm1d
        
        self.resnet1d = Res1DNet(input_channel, Bottleneck1D, [2, 2, 4, 2], output_stride, nn.BatchNorm1d, pretrained=False)
        
        self.amsp = AMSP(input_channel, output_stride, BatchNorm)
        self.channel_last_data = channel_last_data

    def forward(self, input):
        if self.channel_last_data:# swap data axis
            input = input.permute(0,2,1)

        x, low_level_feat = self.resnet1d(input)
        x = self.amsp(x) 

        return x, low_level_feat