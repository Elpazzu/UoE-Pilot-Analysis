import torch
import torch.nn as nn
import torch.nn.functional as F
from net.resnet import ResNet
from net.attention import AttentionGate

class UpSampBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.2, attention=True):
        super(UpSampBlock, self).__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(2*out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(),
        )
        
        self.reduceLayers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1)
        )
        
        self.attention = attention
        if self.attention:
            self.att_gate = AttentionGate(F_g=out_channel, F_l=out_channel, F_int=out_channel//2)

    def forward(self, input, feature): 
        interpolated = F.interpolate(input, scale_factor=2, mode='nearest') 
        interpolated = self.reduceLayers(interpolated)
        if self.attention:
            feature = self.att_gate(interpolated, feature)
        output = torch.cat((interpolated, feature), dim = 1)
        return self.convLayers(output)
    
class AttUNet(nn.Module):
    def __init__(self, backbone_type: str, attention: bool = True, is_train: bool = True, drop_rate: float = 0.3):
        super(AttUNet, self).__init__()
        
        if is_train:
            self.dropRate = drop_rate
            self.is_onehot = False
        else:
            self.dropRate = 0
            self.is_onehot = True
                 
        if backbone_type == "ResNet50":
            self.backbone = ResNet()
            self.downList = [64, 256, 512, 1024, 2048]
        else:
            raise("Invalid type {}".format(self.backbone))
        
        self.upSamp1 = UpSampBlock(self.downList[-1], self.downList[-2], self.dropRate, attention)
        self.upSamp2 = UpSampBlock(self.downList[-2], self.downList[-3], self.dropRate, attention)
        self.upSamp3 = UpSampBlock(self.downList[-3], self.downList[-4], self.dropRate, attention)
        self.upSamp4 = UpSampBlock(self.downList[-4], self.downList[-5], self.dropRate, attention)
        
        self.outBlock = nn.Sequential(
            nn.Conv2d(self.downList[0], 1, 3, 1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        f1, f2, f3, f4, f5 = self.backbone(input)
        out = self.upSamp1(f5, f4)
        out = self.upSamp2(out, f3)
        out = self.upSamp3(out, f2)
        out = self.upSamp4(out, f1)
        out = self.outBlock(out)
        
        if self.is_onehot:
            out = out > 0.5
            out = out.float()
        
        return out
    
if __name__ == "__main__":
    samp = torch.randn(2, 3, 192, 192)
    net = AttUNet("ResNet50", attention=True)
    out = net(samp)
    print(out.shape)
