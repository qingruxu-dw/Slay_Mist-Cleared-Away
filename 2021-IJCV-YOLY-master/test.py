import os
import random

import cv2
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity
from torchvision.transforms import Resize


def pil_to_np(img_PIL, with_transpose=True):
    """
    入参：img_PIL PIL格式的图像， with_transpose指定是否进行数组维度转置

    From W x H x C [0...255] to C x W x H [0..1]
    """
    #将输入的PIL格式的图像转换为np数组  HWC
    ar = np.array(img_PIL)
    #这一条件代表是RGBA的图像格式，最后一个通道代表透明度
    #转为正常的RGB格式
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
    #转置为三通道的C x W x H格式
    if with_transpose:
        if len(ar.shape) == 3:
        #HWC
            ar = ar.transpose(1,2,0)
        else:
            ar = ar[None, ...]
    #将数据类型转换为np.float32，并将数值范围从[0,255]缩放为[0,1]
    return ar.astype(np.float32) / 255.

def load_img_keys(image_out, key_clean_path, n):
    # 获取key_clean_path下的所有文件名（假设这些文件都是图片）
    img_files = os.listdir(key_clean_path)
    torch_resize = Resize([256, 256])  # 定义Resize类对象
    image_out = torch_resize(image_out)
    image_out = image_out.detach().cpu().numpy()[0].transpose(1, 2, 0)
    # 存储每张图片与image_out的SSIM总和的字典
    ssim_sum_dict = {}

    # 计算每张图片与image_out的SSIM总和
    for img_file in img_files:
        img_path = os.path.join(key_clean_path, img_file)

        # 读取样本图片

        sample_img = Image.open(img_path).convert('RGB')

        sample_img = sample_img.resize((256, 256), Image.BILINEAR)

        sample_img = pil_to_np(sample_img).transpose(2, 0, 1)
        # 计算结构相似性指数（SSIM）
        ssim = structural_similarity(image_out, sample_img, multichannel=True, channel_axis=2,data_range=256)
        # 将SSIM总和存入字典
        ssim_sum_dict[img_path] = ssim

    # 根据SSIM总和对字典进行排序，取前n个SSIM最高的图片路径

    sorted_keys = sorted(ssim_sum_dict, key=ssim_sum_dict.get, reverse=True)[:n]


    # 返回图片集合
    return sorted_keys
#
# img = Image.open('./data/HSTS/synthetic/0586.jpg')
# img = TF.to_tensor(img)
# img = img.unsqueeze(0)
#
# print(load_img_keys(img, './data/key_clean', 5))

print(os.listdir('./data/HSTS/synthetic/'))



