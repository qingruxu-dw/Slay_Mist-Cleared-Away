#主去雾流程
#实现从雾图输入到清晰图像输出的完整流程，包括模型初始化、优化和结果保存
from collections import namedtuple
from skimage.color import rgb2hsv
import torch.cuda
from cv2.ximgproc import guidedFilter
from prompts import *
from utils.imresize import np_imresize
from utils.image_io import *
from utils.file_io import write_niqelog
from utils.niqe import *
from options import options, get_output_path, get_log_path
import os
from torch import nn
from net.CLIPDenoising_arch import ModifiedResNet_RemoveFinalLayer
from net.DehazeNet import DehazeNet
from CLIP import *
from utils.DCP import *
from clip_score import L_clip_from_feature
DehazeResult_niqe = namedtuple("DehazeResult", ['learned', 't', 'a', 'niqe'])

class Dehaze(object):
#核心去雾逻辑
    def __init__(self, image_name, image, opt):
        self.image_name = image_name
        self.image = image
        self.num_iter = opt.num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = opt.learning_rate
        self.parameters = None
        #结果存储路径
        self.output_path = get_output_path(opt.datasets, opt.name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_type = torch.cuda.FloatTensor
        # self.device = torch.device("cpu")  # 将代码中的 cuda:0 改为 cpu
        # self.data_type = torch.FloatTensor  # 将 cuda.FloatTensor 改为 FloatTensor
        self.clip = opt.clip
        self.blur_loss = None
        self.best_result = None
        self.current_result_niqe = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.niqe_history = []
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.image_torch = np_to_torch(self.image).type(self.data_type)
        self.enimg = np_to_torch(self.image)

    def _init_nets(self):
        #初始化去雾网络 (DehazeNet) 和 CLIP 模型
        self.dehazeNet = DehazeNet().type(self.data_type)
        clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="/root/2021-YOLY/2021-IJCV-YOLY-master/clip_model/")  # ViT-B/32
        self.clip_model = clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def _init_ambient(self):

        atmosphere = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)

    def _init_parameters(self):

        parameters = [p for p in self.dehazeNet.parameters()]
        self.parameters = parameters

    def _init_loss(self):

        self.mse_loss = torch.nn.MSELoss().type(self.data_type)

        self.clip_loss_fn = L_clip_from_feature().type(self.data_type)

    def _init_inputs(self):

        encoder = ModifiedResNet_RemoveFinalLayer([3, 4, 6, 3], 3, width=64)
        encoder.load_pretrain_model('./clip_model/RN50.pt')
        for params in encoder.parameters():
            params.requires_grad = False
        encoder.eval()

        out = encoder(self.enimg)

        self.out = []
        for i in range(5):
            self.out.append(out[i].type(self.data_type))
        self.sample = enhance_prompts

        self.enhance_features = clip.encode_text(self.clip_model, self.sample, self.device)

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()
            
        # 添加 ↓↓↓
        torch.cuda.empty_cache()  # 迭代结束后清理显存

    def _optimization_closure(self,step):   # 联合优化以下损失函数

        self.image_out, self.mask_out = self.dehazeNet(self.out)

        self.haze_out = self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_val


        hsv = np_to_torch(rgb2hsv(torch_to_np(self.image_out).transpose(1, 2, 0)))
        cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
        self.cap_loss = self.mse_loss(cap_prior, torch.zeros_like(cap_prior))

        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_val,
                                     self.image_torch)

        self.enhance_loss = self.clip_loss_fn(self.clip_model, self.image_out, self.enhance_features)

        self.total_loss = self.mseloss + 0.0001 * self.enhance_loss + 0.01 * self.cap_loss
        # mse_loss：输入与输出的像素级一致性
        # clip_loss：CLIP 特征对齐（使输出接近"清晰图像"的文本描述）
        # cap_loss：基于 HSV 颜色空间的先验约束（暗通道先验变种）

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        #计算当前去雾结果的 NIQE（无参考图像质量指标），动态调整文本提示（RANSAC 策略）

        image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
        mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
        ambient_out_np = np.clip(torch_to_np(self.ambient_val), 0, 1)
        mask_out_np = self.t_matting(mask_out_np)

        post = np.clip((self.image - ((1 - mask_out_np) * ambient_out_np)) / mask_out_np, 0, 1)
        post1 = (post * 255).astype(np.uint8)
        post1 = np.transpose(post1, (1, 2, 0))
        niqe = calculate_niqe(post1,crop_border=0, input_order='HWC', convert_to='y')

        self.niqe_history.append(niqe)

        if len(self.niqe_history) > 50 and all(self.niqe_history[-i] > self.niqe_history[-i - 1] for i in range(1, 51)):
            self.psnr_history = []
            sample = RANSAC(self.sample)
            self.sample = sample
            self.enhance_features = clip.encode_text(self.clip_model, self.sample, self.device)

        self.current_result_niqe = DehazeResult_niqe(learned=image_out_np, t=mask_out_np, a=ambient_out_np,
                                                     niqe=niqe)


        if self.best_result is None or self.best_result.niqe > self.current_result_niqe.niqe:
            self.best_result = self.current_result_niqe

    def _plot_closure(self, step):
        print('Iteration %05d cur_niqe %f max_niqe %f\n' % (
            step,self.current_result_niqe.niqe,self.best_result.niqe), '\r', end='', flush=True)
    def finalize(self):
        psnr_a = np_imresize(self.best_result.a, output_shape=self.image.shape[1:])
        psnr_t = np_imresize(self.best_result.t, output_shape=self.image.shape[1:])
        psnr_img = np.clip((self.image - ((1 - psnr_t) * psnr_a)) / psnr_t, 0, 1)

        save_image(self.image_name + "_niqe", psnr_img, self.output_path)

        final_a = np_imresize(self.current_result_niqe.a, output_shape=self.image.shape[1:])
        final_t = np_imresize(self.current_result_niqe.t, output_shape=self.image.shape[1:])
        post = np.clip((self.image - ((1 - final_t) * final_a)) / final_t, 0, 1)

        save_image(self.image_name + "_final", post, self.output_path)
        
        # 清理显存
        torch.cuda.empty_cache()

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehazing(opt):
    #处理数据集（SOTS、HSTS 等），遍历图像并调用 Dehaze 类去雾，保存结果
    torch.cuda.set_device(opt.cuda)
    # device = torch.device("cpu")
    #日志文件路径
    file_name = get_log_path(opt.datasets, opt.name)
    
#数据集路径 - 支持data/目录下的各个子目录
    if opt.datasets == 'SOTS':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.jpg'
        img_num = len(glob.glob('data/' + opt.datasets + '/synthetic/*.jpg'))  # 动态计算图像数量
    elif opt.datasets == 'HSTS':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.jpg'
        img_num = len(glob.glob('data/' + opt.datasets + '/synthetic/*.jpg'))  # 动态计算图像数量
    elif opt.datasets == 'Fattal':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.png'
        img_num = len(glob.glob('data/' + opt.datasets + '/synthetic/*.png'))  # 动态计算图像数量
    elif opt.datasets == 'NHHAZE':
        hazy_add = 'data/' + opt.datasets + '/haze/*.png'
        img_num = len(glob.glob('data/' + opt.datasets + '/haze/*.png'))  # 动态计算图像数量
    elif opt.datasets == 'school':
        hazy_add = 'data/school/hazy_school/*.jpg'
        img_num = len(glob.glob('data/school/hazy_school/*.jpg'))  # 动态计算图像数量
    elif opt.datasets == 'road':
        hazy_add = 'data/road/hazy_road/*.jpg'
        img_num = len(glob.glob('data/road/hazy_road/*.jpg'))  # 动态计算图像数量
    else:
        # 通用处理：检查data/{datasets}/目录下是否有图像文件
        possible_patterns = [
            f'data/{opt.datasets}/hazy_{opt.datasets}/*.jpg',
            f'data/{opt.datasets}/hazy_{opt.datasets}/*.png',
            f'data/{opt.datasets}/*.jpg',
            f'data/{opt.datasets}/*.png'
        ]
        hazy_add = None
        for pattern in possible_patterns:
            files = glob.glob(pattern)
            if files:
                hazy_add = pattern
                img_num = len(files)
                break
        
        if hazy_add is None:
            print(f'No images found in data/{opt.datasets}/ directory')
            return

    print(hazy_add, img_num)

    rec_niqe = 0

    for item in sorted(glob.glob(hazy_add)):
        print(item)
        # 通用的图像名称提取逻辑
        name = item.split('/')[-1].split('.')[0]  # 提取文件名（不含扩展名）
        print(name)

        hazy_img = prepare_image(item)

        dh = Dehaze(name, hazy_img, opt)
        dh.optimize()
        dh.finalize()
        niqe = dh.best_result.niqe
        rec_niqe += niqe
        
        # 添加
        torch.cuda.empty_cache()  # 单张图像处理完成后清理显存

    rec_niqe = rec_niqe / img_num

    write_niqelog(file_name, 'Average', rec_niqe)


if __name__ == "__main__":
    dehazing(options)
    print('Zero-Shot Image Dehaze Done!')
