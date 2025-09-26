import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random
from net.CLIPDenoising_arch import ModifiedResNet_RemoveFinalLayer


def compute_cosine_similarity(feature_map1, feature_map2):
    """计算余弦相似度"""
    feature_map1 = feature_map1.flatten(1)
    feature_map2 = feature_map2.flatten(1)
    cosine_sim = F.cosine_similarity(feature_map1, feature_map2, dim=1)
    return cosine_sim.mean().item()


def evaluate_image_similarity(haze_dir, clean_dir, model_path, num_samples=300):
    """评估模糊图像与清晰图像之间的特征相似度"""
    # 加载预训练模型
    encoder = ModifiedResNet_RemoveFinalLayer([3, 4, 6, 3], 3, width=64)
    encoder.load_pretrain_model(model_path)
    for params in encoder.parameters():
        params.requires_grad = False
    encoder.eval()

    # 用于存储不同尺度的相似度
    scales = [[] for _ in range(5)]

    # 随机选择图像
    img_paths = [os.path.join(haze_dir, f) for f in os.listdir(haze_dir)]
    random_selected_images = random.sample(img_paths, num_samples)

    for img in random_selected_images:
        img_name = os.path.basename(img)

        # 加载模糊图像
        hazy_img = Image.open(img).convert('RGB')
        hazy_img = transforms.ToTensor()(hazy_img).unsqueeze(0)

        # 加载清晰图像
        gt_img = Image.open(os.path.join(clean_dir, img_name)).convert('RGB')
        gt_img = transforms.ToTensor()(gt_img).unsqueeze(0)

        with torch.no_grad():
            hazy_out = encoder(hazy_img)
            gt_out = encoder(gt_img)

        for i in range(5):
            scales[i].append(compute_cosine_similarity(hazy_out[i + 1], gt_out[i + 1]))

    # 计算每个尺度的平均相似度
    average_scales = [sum(scale) / len(scale) for scale in scales]

    return average_scales



'''
encoder输入一张图片，可以返回多个不同尺度的特征图。
如果是自己的模型，需要自己写提取不同尺度的特征，然后返回不同尺度的特征：


class ModifiedResNet_RemoveFinalLayer(ModifiedResNet):
   
    def __init__(self, layers, in_chn=3, width=64):
        super().__init__(layers, in_chn, width)

    def forward(self, x):
        out = []  # store multi-scale dense features

        x = x.type(self.conv1.weight.dtype); out.append(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); out.append(x)  #scale-1 F1
        x = self.avgpool(x)
        x = self.layer1(x); out.append(x)   #scale-2 F2
        x = self.layer2(x); out.append(x)   #scale-3 F3
        x = self.layer3(x); out.append(x)   #scale-4 F4
        x = self.layer4(x); out.append(x)         #scale-4 F5

        return out
'''

# 使用示例
haze_dir = 'C:/Users/86150/Desktop/定性比较/SOTS_outdoor/hazy/'
clean_dir = 'C:/Users/86150/Desktop/定性比较/SOTS_outdoor/gt/'

#预训练好的模型
model_path = './clip_model/RN50.pt'

average_similarities = evaluate_image_similarity(haze_dir, clean_dir, model_path)
for i, sim in enumerate(average_similarities, 1):
    print(f'scale{i}:', sim)