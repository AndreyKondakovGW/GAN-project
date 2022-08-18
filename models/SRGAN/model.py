import torch.nn as nn
import torch

class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True)
        
        self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
        self.bias.data = torch.zeros((out_channels))
        
        for p in self.parameters():
            p.requires_grad = True

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN = False, act = None, stride = 1, bias = True):
        super(conv, self).__init__()
        m = []
        m.append(_conv(in_channels = in_channel, out_channels = out_channel, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True))
        
        if BN:
            m.append(nn.BatchNorm2d(num_features = out_channel))
        
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act = nn.ReLU(inplace = True), bias = True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN = True, act = act))
        m.append(conv(channels, channels, kernel_size, BN = True, act = None))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act = nn.ReLU(inplace = True)):
        super(Upsampler, self).__init__()
        m = []
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale))
    
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        out = self.body(x)
        return out

class ResGenerator(nn.Module):
    
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16, act = nn.PReLU(), scale=4):
        super(ResGenerator, self).__init__()
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        
        resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        
        if(scale == 4):
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]

        self.tail = nn.Sequential(*upsample_blocks)
        
        self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        
    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat