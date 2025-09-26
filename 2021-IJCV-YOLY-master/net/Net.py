#原版作者的CNN 效果好
# YOLY 用作对比实验 可兜底
import torch

class Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.final(data)
        return data
if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224)
    model = Net(out_channel=3)
    print(model(images).size())
