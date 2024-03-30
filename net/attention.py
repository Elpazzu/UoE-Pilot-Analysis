from audioop import bias
import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):  # F_g: input channels gate g; F_l: input channels featuremap x_l; F_int: intermediate channels
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(  # weight transformation layers for gate g 
            nn.Conv2d(F_g, F_int, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(  # weight transformation layers for feature map x_l
            nn.Conv2d(F_l, F_int, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.relu = nn.ReLU(inplace=False)
        self.psi = nn.Sequential(  # final layer to generate attention map
            nn.Conv2d(F_int, 1, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x_l):
        id_x = x_l
        alpha = self.relu(self.W_g(g) + self.W_x(x_l))
        alpha = self.psi(alpha)
        return alpha * id_x