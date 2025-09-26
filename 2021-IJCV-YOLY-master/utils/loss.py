import torch
import torch.nn.functional as F


def dark_channel(image, window_size=15):
    # 计算暗通道图像
    padded_image = F.pad(image, (window_size // 2, window_size // 2, window_size // 2, window_size // 2),
                         mode='reflect')
    batch_size, channels, height, width = padded_image.size()

    dark_channel_map = torch.zeros((batch_size, 1, height, width), dtype=image.dtype, device=image.device)

    for i in range(height):
        for j in range(width):
            patch = padded_image[:, :, i:i + window_size, j:j + window_size]
            min_value, _ = torch.min(patch.view(batch_size, channels, -1), dim=2)
            dark_channel_map[:, :, i, j] = torch.min(min_value, dim=1)[0]

    return dark_channel_map


class DarkChannelLoss(torch.nn.Module):
    def __init__(self, window_size=15):
        super(DarkChannelLoss, self).__init__()
        self.window_size = window_size

    def forward(self, generated_image):
        # 计算生成图像的暗通道
        generated_dark_channel = dark_channel(generated_image, self.window_size)

        # 暗通道损失：鼓励生成图像的暗通道与自然图像的暗通道相似
        loss = -torch.mean(generated_dark_channel)  # 通过最大化暗通道的最小值来最小化暗通道损失

        return loss