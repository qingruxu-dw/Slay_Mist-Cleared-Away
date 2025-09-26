import torch
from torch import nn

class JNet(nn.Module):
    def __init__(self):
        super(JNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 2, 2, 0),
            nn.LeakyReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, out):
        E0 = out[0]
        help1 = out[1]

        E1 = self.conv1(E0)
        E2 = self.conv2(E1 + help1)
        S2 = self.conv3(E2)
        D1 = self.deconv3(S2) + self.conv4(E1)
        D0 = self.deconv2(D1) + self.conv5(E0)
        out = D0
        return out


if __name__ == '__main__':
    T = JNet()
    torch = [torch.randn(1, 3, 464, 544), torch.randn(1, 64, 232, 272), torch.randn(1, 256, 116, 136), torch.randn(1, 512, 58, 68), torch.randn(1, 1024, 29, 34)]
    print(T(torch).shape)


