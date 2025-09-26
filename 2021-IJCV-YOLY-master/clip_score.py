#CLIP 特征计算与损失函数，计算图像与文本的 CLIP 特征相似度，用于指导去雾过程
import torch
import torch.nn as nn

from CLIP.clip import clip_feature_surgery

from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import torchvision.transforms.functional as TF

# 兼容性处理：处理不同版本的 torchvision 和 PIL
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    try:
        from PIL import Image
        BICUBIC = Image.BICUBIC
    except ImportError:
        # 如果都没有，使用默认插值
        BICUBIC = None

# 创建兼容的 resize 函数
def create_resize_transform(size, interpolation=None):
    if interpolation is not None and BICUBIC is not None:
        try:
            return Resize(size, interpolation=BICUBIC)
        except:
            return Resize(size)  # fallback
    else:
        return Resize(size)

# 修改 img_resize 以兼容旧版本
if BICUBIC is not None:
    try:
        img_resize = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            # 移除 ToTensor，因为输入已经是 tensor
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # 移除 antialias 参数，因为旧版本不支持
        # 如果需要 antialias 功能，可以手动实现或忽略
    except:
        # 如果上面的方法失败，使用简化版本
        img_resize = Compose([
            Resize((224, 224)),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
else:
    img_resize = Compose([
        Resize((224, 224)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# 移除 antialias=True 参数，因为旧版本不支持

#计算去雾后图像与预设文本提示（如"清晰图像"）的 CLIP 特征相似度，作为优化目标
def get_clip_score_from_feature(model, image, text_features, temp=100.):
    # size of image: [b, 3, 224, 224]  ,text_features: [5, 512]
    # 直接处理 tensor，而不是转换为 PIL Image 再转回 tensor
    transformed_images = []
    for img in image:
        # 如果输入已经是 tensor，则直接进行 resize 和 normalize
        # 将 tensor 转换到 [0,1] 范围（如果需要）
        img_normalized = torch.clamp(img, 0, 1)
        # Resize 操作需要先转换为 PIL Image 或使用 interpolate
        resized_img = F.interpolate(img_normalized.unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False)
        # 应用 CLIP 的 normalize
        normalized_img = TF.normalize(
            resized_img.squeeze(0),
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        transformed_images.append(normalized_img)
        
    image = torch.cat(transformed_images, dim=0)
    
     # 确保 image 具有正确的批次维度
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # 添加批次维度
        
    image_features = model.encode_image(image)  # 提取图像特征
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    probs = temp * clip_feature_surgery(image_features, text_features)[:, 1:, :]
    similarity = torch.mean(probs.softmax(dim=-1), dim=1, keepdim=False) # 计算图像-文本相似度
    loss = 1. - similarity[:, 0]      # 损失函数：最大化相似度
    loss = torch.sum(loss) / len(loss)
    return loss

#封装为 PyTorch 损失函数模块，用于反向传播
class L_clip_from_feature(nn.Module):
    def __init__(self, temp=100.):
        super(L_clip_from_feature, self).__init__()
        self.temp = temp
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, model, x, text_features):
        k1 = get_clip_score_from_feature(model, x, text_features, self.temp)
        return k1

#计算去雾结果与输入图像的 CLIP 特征均方误差（MSE），保留内容一致性
def get_clip_score_MSE(res_model, pred, inp, weight):
    # 手动处理图像变换
    pred_transformed = []
    inp_transformed = []
    
    for i in range(pred.shape[0]):
        pred_img = pred[i:i+1]  # 保持批次维度
        inp_img = inp[i:i+1]    # 保持批次维度
        
        pred_transformed.append(img_resize(pred_img))
        inp_transformed.append(img_resize(inp_img))
    
    pred_stacked = torch.cat(pred_transformed, dim=0)
    inp_stacked = torch.cat(inp_transformed, dim=0)
    
    stack = torch.cat([pred_stacked, inp_stacked], dim=1)
    
    pred_image_features = res_model.encode_image(stack[:, :3, :, :])
    inp_image_features = res_model.encode_image(stack[:, 3:, :, :])

    MSE_loss = 0
    for feature_index in range(len(weight)):
        MSE_loss = MSE_loss + weight[feature_index] * F.mse_loss(pred_image_features[1][feature_index], inp_image_features[1][feature_index])

    return MSE_loss


class L_clip_MSE(nn.Module):
    def __init__(self):
        super(L_clip_MSE, self).__init__()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, model, pred, inp, weight=None):
        if weight is None:
            weight = [1.0, 1.0, 1.0, 1.0, 0.5]
        res = get_clip_score_MSE(model, pred, inp, weight)
        return res