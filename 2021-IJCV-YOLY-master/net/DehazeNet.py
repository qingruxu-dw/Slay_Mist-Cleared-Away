from net.HelpNet import *

class TNet(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            #torch.nn.Conv2d(3, out_channel, 5, 1, 2),
            torch.nn.Conv2d(3,out_channel,3,1,1),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        #print(data.size())
        data = self.conv1(data)
        #print(data.size())
        data = self.conv2(data)
        #print(data.size())
        data = self.conv3(data)
        #print(data.size())
        data = self.conv4(data)
        #print(data.size())
        data = self.final(data)
        #print(data.size())
        return data

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class JNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            #torch.nn.Conv2d(3, out_channel, 5, 1, 2),
            torch.nn.Conv2d(3,64,3,1,1),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True)
        )

        self.ca = CALayer(64)

        self.attn = Edge_Attention_Layer(64)

        self.conv2 = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            #torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 3, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self,img, out):

        #print(data.size())
        data = self.conv1(img)
        data = self.ca(data)
        data = self.attn(data,out[3])

        #print(data.size())
        #data = self.pa(data)
        data = self.conv2(data)
        data = self.ca(data)
        data = self.attn(data,out[1])

        #data = self.pa(data)
        data = self.conv3(data)
        data = self.ca(data)
        data = self.attn(data,out[2])

        #data = self.pa(data)
        #print(data.size())
        data = self.conv4(data)
        data = self.ca(data)
        data = self.attn(data,out[4])

        #data = self.pa(data)
        #print(data.size())
        data = self.final(data)
        #print(data.size())
        return data


class DehazeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.ftp = FeatureProcess()

        self.sca1 = nn.Sequential(
            nn.Conv2d(64, 3, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.tnet = TNet(1)

        self.jnet = JNet()

    def forward(self, out):

        img = out[0] + self.sca1(out[1])

        out = self.ftp(out)

        j = self.jnet(img,out)

        t = self.tnet(img)

        return j,t


if __name__ == '__main__':

    a = [torch.randn(1, 3, 640, 640), torch.randn(1, 64, 320, 320), torch.randn(1, 256, 160, 160), torch.randn(1, 512, 80, 80), torch.randn(1, 1024, 40, 40)]

    import torch
    from thop import profile
    import time
    model = DehazeNet()
    start_time = time.time()
    flops, params = profile(model, inputs=(a,))
    end_time = time.time()
    print(f"macs = {2*flops/1e9}G")
    print(f"params = {params/1e6}M")
    print("flops:{}G".format(flops/1e9))
    print("time_cost:{}s".format((end_time-start_time)))







