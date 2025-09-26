#根据PDU魔改的大气光子网：效果不佳
#不采用 可用作消融实验

import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            #nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.ka1 = nn.Sequential(
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.ka2 = nn.Sequential(
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.ka3 = nn.Sequential(
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        #print("avg",a.size())
        a = self.ka1(a)
        #print("ka1", a.size())
        a = self.ka2(a)
        #print("ka2", a.size())
        a = self.ka3(a)
        #print("ka3", a.size())
        return(a)
        #j = torch.mul((1 - t), a) + torch.mul(t, x)
        #print("clean", j.size())
        #return j

if __name__ == "__main__":
    # net = C2PNet(gps=3, blocks=19)
    # print(net)
    images = torch.rand(1, 3, 224, 224)
    model = PDU(3)
    print(model(images).size())