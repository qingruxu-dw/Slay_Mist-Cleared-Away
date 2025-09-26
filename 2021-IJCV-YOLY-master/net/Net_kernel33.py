#改成两个3×3卷积为了扩大感受野 可采用
import torch

class Net_kernel33(torch.nn.Module):
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

if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224)
    model = Net_kernel33(out_channel=3)
    print(model(images).size())