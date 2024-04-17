import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, stride = 1, dSampScheme=None):  
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channel, mid_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channel, mid_channel * 4, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel * 4),
            nn.ReLU(inplace=False)
        ) 
        self.stride = stride
        self.dSampScheme = dSampScheme
        self.out_ReLU = nn.ReLU(inplace=False)

    def forward(self, input):
        identity = input
        out = self.convs(input)
        
        if self.dSampScheme is not None:  # if downsampling scheme provided
            identity = self.dSampScheme(identity)  # apply it to identity tensor
        out += identity  # add output tensor to identity tensor to form residual connection
        
        return self.out_ReLU(out)
    
class ResNet(nn.Module):
    def __init__(self, block_num_list=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        
        self.block_num_list = block_num_list
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.conv2_x = self.makeLayer(64, 64, block_num_list[0])
        self.conv3_x = self.makeLayer(256, 128, block_num_list[0], 2)
        self.conv4_x = self.makeLayer(512, 256, block_num_list[0], 2)
        self.conv5_x = self.makeLayer(1024, 512, block_num_list[0], 2)
    
    def makeLayer(self, prev_channel, mid_channel, block_num, stride=1):
        dSampScheme = None
        outChannel = 4 * mid_channel
        blockList = []
        if stride != 1 or prev_channel != mid_channel * 4:
            dSampScheme = nn.Sequential(
                nn.Conv2d(prev_channel, mid_channel*4, 1, stride, bias=False),
                nn.BatchNorm2d(outChannel)
            )
        
        convBlock = ConvBlock(prev_channel, mid_channel, stride, dSampScheme)
        blockList.append(convBlock)
        for i in range(1, block_num):
            identityBlock = ConvBlock(outChannel, mid_channel)
            blockList.append(identityBlock)

        return nn.Sequential(*blockList)
    
    def forward(self, input):
        f1 = self.conv1(input)
        f2 = self.conv2_x(self.maxpool(f1))
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        return f1, f2, f3, f4, f5
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 240, 240)
    net = ResNet()
    f1, f2, f3, f4, f5 = net(x)
    print(f1.shape)
    print(f2.shape)
    print(f3.shape)
    print(f4.shape)
    print(f5.shape)
