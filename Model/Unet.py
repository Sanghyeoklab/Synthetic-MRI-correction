import torch
import torch.nn as nn
import math

class Conv_layers(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_layers, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel)
        )
        torch.nn.init.normal_(self.layer[0].weight, mean = 0.0, std = 1 / math.sqrt(in_channel * 3 * 3))
        torch.nn.init.normal_(self.layer[3].weight, mean = 0.0, std = 1 / math.sqrt(out_channel * 0.5 * 3 * 3))
    def forward(self, x):
        return self.layer(x) 
    
class Upsample(nn.Module):
    def __init__(self, in_channel1, out_channel1, in_channel2, out_channel2):
        super(Upsample, self).__init__()
        self.pool = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(in_channel1, out_channel1, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel1)
        )
        torch.nn.init.normal_(self.pool[1].weight, mean = 0.0, std = 1 / math.sqrt(in_channel1 * 3 * 3))
        self.Conv_layers = Conv_layers(in_channel2, out_channel2) 
    def forward(self, x, featuremap):
        output = self.pool(x)
        output = torch.cat((output, featuremap), 1)
        return self.Conv_layers(output) 

class Downpool(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downpool, self).__init__()
        self.Conv_layers = Conv_layers(in_channel, out_channel)
        self.pool = nn.Conv2d(out_channel, out_channel, 2, 2, 0)
        torch.nn.init.normal_(self.pool.weight, mean = 0.0, std = 1 / math.sqrt(out_channel * 2 * 2))
        
    def forward(self, x):
        featuremap = self.Conv_layers(x) 
        return self.pool(featuremap), featuremap
    
    
class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        self.encoder1 = Downpool(in_channel, 64) 
        self.encoder2 = Downpool(64, 128) 
        self.encoder3 = Downpool(128, 256) 
        self.encoder4 = Downpool(256, 512)
        
        self.latent = Conv_layers(512, 1024)
        
        self.decoder1 = Upsample(1024, 512, 1024, 512) 
        self.decoder2 = Upsample(512, 256, 512, 256) 
        self.decoder3 = Upsample(256, 128, 256, 128) 
        self.decoder4 = Upsample(128, 64, 128, 64) 
        
        self.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, out_channel, 3, 1, 1, padding_mode='reflect')
        )
        torch.nn.init.normal_(self.last_conv[1].weight, mean = 0.0, std = 1 / math.sqrt(64 * 0.5 * 3 * 3))
    def forward(self, x):
        output, featuremap1 = self.encoder1(x)
        output, featuremap2 = self.encoder2(output)
        output, featuremap3 = self.encoder3(output)
        output, featuremap4 = self.encoder4(output)
        output = self.latent(output)
        output = self.decoder1(output, featuremap4)
        output = self.decoder2(output, featuremap3)
        output = self.decoder3(output, featuremap2)
        output = self.decoder4(output, featuremap1)
        return self.last_conv(output)