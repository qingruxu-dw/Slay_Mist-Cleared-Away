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

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        #vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])  #第 1 层
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) #第 3 层
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])#第 5 层
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])#第 9 层
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])#第 13 层
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self):

        super(ContrastLoss, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vgg = Vgg19().to(self.device)
        self.vgg.eval()
        self.l1 = nn.L1Loss().to(self.device)
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]


    def forward(self, a, clean_keys, hazy_keys):
        #把a,p,n放进vgg中得到相应的特征值
        #print("a",a.size())
        torch_resize = Resize([256, 256])#定义Resize类对象
        a = torch_resize(a)
        #print("a", a.size())
        a_vgg, p_vgg, n_vgg = self.vgg(a.to(self.device)), self.vgg(clean_keys.to(self.device)), \
            self.vgg(hazy_keys.to(self.device))
        loss = 0
        #对于每一层的a进行计算，此处用到vgg19的5层，所以循环五次
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            #计算positive中的每一个 暂定为5
            for j in range(p_vgg[i].size()[0]):  #遍历正样本batch中的每一个
                # print('a_vgg[i][0].size', a_vgg[i][0].size())
                # print('p_vgg[i][j].size', p_vgg[i][j].detach().size())
                d_ap = d_ap + self.l1(a_vgg[i][0], p_vgg[i][j].detach())  #计算第i层，输出的无雾图和每一个(j)个正样本的L1loss
            d_ap = d_ap / p_vgg[i].size()[0]

            #计算negtive中的每一个暂定为5+
            for k in range(n_vgg[i].size()[0]): #遍历负样本batch中的每一个
                d_an = d_an + self.l1(a_vgg[i][0], n_vgg[i][k].detach())  #计算第i层，输出的无雾图和每一个(k)个负样本的L1loss
            d_an = d_an / n_vgg[i].size()[0]
            contrastive = d_ap / d_an
            loss = loss + self.weights[i] * contrastive
        return loss


def load_clean_keys(image_out, key_clean_path, n):
    # 获取原始图片的尺寸
    #_, _, h, w = image_out.size()

    # 从key_clean文件夹中随机读取n张图片
    clean_files = os.listdir(key_clean_path)
    random.shuffle(clean_files)
    clean_files = clean_files[:n]

    # 读取n张图片并裁剪为和原始图片同样的尺寸
    clean_keys = []
    for clean_file in clean_files:
        clean_path = os.path.join(key_clean_path, clean_file)
        clean_img = Image.open(clean_path).convert('RGB')
        #clean_img = clean_img.crop((0, 0, h, w))
        clean_img = clean_img.resize((256,256),Image.BILINEAR)
        # 将图片转为torch变量
        clean_tensor = torch.from_numpy(np.array(clean_img)).float().permute(2, 1, 0) / 255.0
        clean_keys.append(clean_tensor.unsqueeze(0))
        #print(clean_keys.size())

    return torch.cat(clean_keys, dim=0)

def load_hazy_keys(image_out, key_hazy_path, n):
    # 获取原始图片的尺寸
    #_, _, h, w = image_out.size()
    # h, w = 200, 200
    # 从key_clean文件夹中随机读取n张图片
    hazy_files = os.listdir(key_hazy_path)
    random.shuffle(hazy_files)

    hazy_files = hazy_files[:n]

    # 读取n张图片并裁剪为和原始图片同样的尺寸
    hazy_keys = []
    for hazy_file in hazy_files:
        hazy_path = os.path.join(key_hazy_path, hazy_file)
        hazy_img = Image.open(hazy_path).convert('RGB')
        #hazy_img = hazy_img.crop((0, 0, h, w))
        hazy_img = hazy_img.resize((256,256),Image.BILINEAR)

        # 将图片转为torch变量
        hazy_tensor = torch.from_numpy(np.array(hazy_img)).float().permute(2, 1, 0) / 255.0
        hazy_keys.append(hazy_tensor.unsqueeze(0))
        #print(hazy_keys.size())
    return torch.cat(hazy_keys, dim=0)


# 使用RANSAC随机抽取图片
def load_img_keys(image_out, key_clean_path, n, ransac_times = 5):
    # 获取原始图片的尺寸
    #_, _, h, w = image_out.size()
    #image_out = image_out.detach().cpu().numpy()[0].transpose(1, 2, 0)  # to HWC
    torch_resize = Resize([256, 256])  #定义Resize类对象
    image_out = torch_resize(image_out)

    image_out = image_out.detach().cpu().numpy()[0].transpose(1, 2, 0)
    #cv2.imwrite('img.jpg',image_out)
    #print(image_out.shape)

    # 从key_clean文件夹中随机读取n张图片
    clean_paths = os.listdir(key_clean_path)
    # 存储每张图片与image_out的SSIM总和的字典

    ssim_sum_dict = {}
    # 计算每张图片与image_out的SSIM总和
    for img_file in clean_paths:
        img_path = os.path.join(key_clean_path, img_file)

        # 读取样本图片

        sample_img = Image.open(img_path).convert('RGB')

        sample_img = sample_img.resize((256, 256), Image.BILINEAR)

        sample_img = pil_to_np(sample_img).transpose(2, 0, 1)
        # 计算结构相似性指数（SSIM）
        ssim = structural_similarity(image_out, sample_img, multichannel=True)
        # 将SSIM总和存入字典
        ssim_sum_dict[img_file] = ssim

    # 根据SSIM总和对字典进行排序，取前n个SSIM最高的图片路径
    clean_files = sorted(ssim_sum_dict, key=ssim_sum_dict.get, reverse=True)
        # print(ransac_ssim, clean_files)、
    top_files = clean_files[:2 * n]
    # Randomly select n files from the top 2*n files
    clean_files = random.sample(top_files, n)
    # 读取n张图片并裁剪为和原始图片同样的尺寸
    clean_keys = []
    for clean_file in clean_files:
        clean_path = os.path.join(key_clean_path, clean_file)
        clean_img = Image.open(clean_path).convert('RGB')
        #clean_img = clean_img.crop((0, 0, h, w))
        clean_img = clean_img.resize((256,256),Image.BILINEAR)

        # 将图片转为torch变量
        clean_tensor = torch.from_numpy(np.array(clean_img)).float().permute(2, 0, 1) / 255.0
        #print(clean_tensor.size())
        clean_keys.append(clean_tensor.unsqueeze(0))

    return torch.cat(clean_keys, dim=0)

# def load_hazy_keys(image_out, key_hazy_path, n, ransac_times = 3):
#     # 获取原始图片的尺寸
#     _, _, h, w = image_out.size()
#     # h, w = 200, 200
#     # 从key_clean文件夹中随机读取n张图片
#     hazy_paths = os.listdir(key_hazy_path)
#     max_ssim = 0
#     hazy_files = []
#
#     for i in range(ransac_times):
#         random.shuffle(hazy_paths)
#         ransac_files = hazy_paths[:n]
#         ransac_ssim = 0
#         for ransac_file in ransac_files:
#             ransac_path = os.path.join(key_hazy_path, ransac_file)
#             ransac_img = Image.open(ransac_path).convert('RGB')
#             ransac_img = ransac_img.crop((0, 0, h, w))
#
#             ransac_ssim += structural_similarity(image_out.transpose(1, 2, 0), ransac_img.transpose(1, 2, 0), multichannel=True)
#
#         if ransac_ssim > max_ssim:
#             hazy_files = ransac_files
#
#     # 读取n张图片并裁剪为和原始图片同样的尺寸
#     hazy_keys = []
#     for hazy_file in hazy_files:
#         hazy_path = os.path.join(key_hazy_path, hazy_file)
#         hazy_img = Image.open(hazy_path).convert('RGB')
#         hazy_img = hazy_img.crop((0, 0, h, w))
#
#         # 将图片转为torch变量
#         hazy_tensor = torch.from_numpy(np.array(hazy_img)).float().permute(2, 1, 0) / 255.0
#         hazy_keys.append(hazy_tensor.unsqueeze(0))
#
#     return torch.cat(hazy_keys, dim=0)

#def toTensor(path):
     #im_input = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # 读取图片，将 BGR 转化为 RGB
     #im_input = im_input / 255.0             # 归一化
     #im_input = np.transpose(im_input, (2, 0, 1))    # HWC -> CHW
     #im_input = im_input[np.newaxis, ...]        # CHW -> NCHW
     #return torch.from_numpy(im_input).float()   # 转化为torch变量

#pil格式转化为np数组格式
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


if __name__ == "__main__":
    images_out = torch.rand(1, 3, 1200, 1600)
    c = ContrastLoss()
    c.forward(images_out,
              load_img_keys(images_out,r'D:\ABye\2021-IJCV-YOLY-master\data\key_clean',5),
              load_img_keys(images_out,r'D:\ABye\2021-IJCV-YOLY-master\data\key_haze',5))
    # c.forward(toTensor(r'D:\做科研\code\AECR-Net-main\AECR-Net-main\data_utils\NW_Google_837.jpeg'),
    #           toTensor(r'D:\做科研\code\AECR-Net-main\AECR-Net-main\data_utils\09_GT.png'),
    #           toTensor(r'D:\做科研\code\AECR-Net-main\AECR-Net-main\data_utils\33_hazy.png'))
