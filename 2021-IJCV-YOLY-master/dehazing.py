from collections import namedtuple
from skimage.color import rgb2hsv
import torch.cuda
from cv2.ximgproc import guidedFilter
from prompts import *
from utils.imresize import np_imresize
from utils.image_io import *
from utils.file_io import write_log, write_process
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from options import options, get_output_path, get_log_path, get_process_log_path
import os
from torch import nn
from net.CLIPDenoising_arch import ModifiedResNet_RemoveFinalLayer
from net.DehazeNet import DehazeNet
from CLIP import *
from utils.DCP import *
from clip_score import L_clip_from_feature
DehazeResult_psnr = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])
DehazeResult_ssim = namedtuple("DehazeResult", ['learned', 't', 'a', 'ssim'])

class Dehaze(object):

    def __init__(self, image_name, image, gt_img, opt):
        self.image_name = image_name
        self.image = image
        self.gt_img = gt_img
        self.num_iter = opt.num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = opt.learning_rate
        self.parameters = None
        self.current_result_psnr = None
        self.output_path = get_output_path(opt.datasets, opt.name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_type = torch.cuda.FloatTensor
        self.clip = opt.clip
        self.blur_loss = None
        self.best_result = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.psnr_history = []
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.image_torch = np_to_torch(self.image).type(self.data_type)
        self.enimg = np_to_torch(self.image)

    def _init_nets(self):

        self.dehazeNet = DehazeNet().type(self.data_type)
        clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="clip_model/")  # ViT-B/32
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
        encoder.load_pretrain_model('clip_model/RN50.pt')
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

    def _optimization_closure(self,step):

        self.image_out, self.mask_out = self.dehazeNet(self.out)

        self.haze_out = self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_val


        hsv = np_to_torch(rgb2hsv(torch_to_np(self.image_out).transpose(1, 2, 0)))
        cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
        self.cap_loss = self.mse_loss(cap_prior, torch.zeros_like(cap_prior))

        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_val,
                                     self.image_torch)

        self.enhance_loss = self.clip_loss_fn(self.clip_model, self.image_out, self.enhance_features)

        self.total_loss = self.mseloss + 0.0001 * self.enhance_loss + 0.01 * self.cap_loss

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):

        image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
        mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
        ambient_out_np = np.clip(torch_to_np(self.ambient_val), 0, 1)
        mask_out_np = self.t_matting(mask_out_np)

        post = np.clip((self.image - ((1 - mask_out_np) * ambient_out_np)) / mask_out_np, 0, 1)

        psnr = peak_signal_noise_ratio(self.gt_img, post)

        ssim = structural_similarity(self.gt_img.transpose(1, 2, 0), post.transpose(1, 2, 0), multichannel=True)

        self.psnr_history.append(psnr)

        self.current_result_psnr = DehazeResult_psnr(learned=image_out_np, t=mask_out_np, a=ambient_out_np,
                                                     psnr=psnr)

        self.current_result_ssim = DehazeResult_ssim(learned=image_out_np, t=mask_out_np, a=ambient_out_np,
                                                     ssim=ssim)

        if self.best_result is None or self.best_result.psnr < self.current_result_psnr.psnr:
            self.best_result = self.current_result_psnr

        if self.best_result_ssim is None or self.best_result_ssim.ssim < self.current_result_ssim.ssim:
            self.best_result_ssim = self.current_result_ssim

    def _plot_closure(self, step):
        print('Iteration %05d    Loss  %f %f %0.4f%% cur_ssim %f max_ssim: %f cur_psnr %f max_psnr %f\n' % (
            step, self.total_loss.item(),
            self.enhance_loss,
            # self.cr_loss_value,
            self.enhance_loss / self.total_loss.item(),
            self.current_result_ssim.ssim,
            self.best_result_ssim.ssim,
            self.current_result_psnr.psnr,
            self.best_result.psnr), '\r', end='',flush=True)
        #处理过程日志
        write_process(get_process_log_path(),'Iteration %05d    Loss  %f %f %0.4f%% cur_ssim %f max_ssim: %f cur_psnr %f max_psnr %f\n' % (
            step, self.total_loss.item(),
            self.enhance_loss,
            # self.cr_loss_value,
            self.enhance_loss / self.total_loss.item(),
            self.current_result_ssim.ssim,
            self.best_result_ssim.ssim,
            self.current_result_psnr.psnr,
            self.best_result.psnr))


    def finalize(self):
        psnr_a = np_imresize(self.best_result.a, output_shape=self.image.shape[1:])
        psnr_t = np_imresize(self.best_result.t, output_shape=self.image.shape[1:])
        psnr_img = np.clip((self.image - ((1 - psnr_t) * psnr_a)) / psnr_t, 0, 1)

        save_image(self.image_name + "_PSNR", psnr_img, self.output_path)

        ssim_a = np_imresize(self.best_result_ssim.a, output_shape=self.image.shape[1:])
        ssim_t = np_imresize(self.best_result_ssim.t, output_shape=self.image.shape[1:])
        ssim_img = np.clip((self.image - ((1 - ssim_t) * ssim_a)) / ssim_t, 0, 1)

        save_image(self.image_name + "_SSIM", ssim_img, self.output_path)

        final_a = np_imresize(self.current_result_psnr.a, output_shape=self.image.shape[1:])
        final_t = np_imresize(self.current_result_psnr.t, output_shape=self.image.shape[1:])
        post = np.clip((self.image - ((1 - final_t) * final_a)) / final_t, 0, 1)

        save_image(self.image_name + "_final", post, self.output_path)

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehazing(opt):
      # torch.cuda.set_device(opt.cuda)
    device = torch.device("cpu")
    #日志文件路径
    file_name = get_log_path(opt.datasets, opt.name)
#数据集路径
    if opt.datasets == 'SOTS_indoor':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.png'
        img_num = 500
    elif opt.datasets == 'SOTS_outdoor':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.jpg'
        img_num = 500
    elif opt.datasets == 'HSTS':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.jpg'
        img_num = 10
    elif opt.datasets == 'RICE':
        hazy_add = 'data/' + opt.datasets + '/haze/*.png'
        img_num = 248
    elif opt.datasets == 'SateHaze_thin':
        hazy_add = 'data/' + opt.datasets + '/input/*.png'
        img_num = 45
    elif opt.datasets == 'SateHaze_thick':
        hazy_add = 'data/' + opt.datasets + '/input/*.png'
        img_num = 45
    elif opt.datasets == 'SateHaze_moderate':
        hazy_add = 'data/' + opt.datasets + '/input/*.png'
        img_num = 45
    elif opt.datasets == 'school':
        hazy_add = 'data/' + opt.datasets + '/hazy_school/*.jpg'
        img_num = 100  # 可根据实际图像数量调整
    else:
        print('There are no proper datasets')
        return

    print(hazy_add, img_num)

    rec_psnr = 0
    rec_ssim = 0

    for item in sorted(glob.glob(hazy_add)):
        print(item)
        if opt.datasets == 'HSTS' or opt.datasets == 'SOTS_outdoor':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'SOTS_indoor':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'real-world':
            name = item.split('.')[0].split('/')[2]
        elif opt.datasets == 'RICE':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'SateHaze_thin':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'SateHaze_thick':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'SateHaze_moderate':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'school':
            name = item.split('.')[0].split('/')[3]

        print(name)
#Ground truth 图像路径：
        if opt.datasets == 'SOTS_indoor':
            gt_add = 'data/' + opt.datasets + '/original/' + name + '.png'
        elif opt.datasets == 'SOTS_outdoor':
            gt_add = 'data/' + opt.datasets + '/original/' + name + '.jpg'
        elif opt.datasets == 'HSTS':
            gt_add = 'data/' + opt.datasets + '/original/' + name + '.jpg'
        elif opt.datasets == 'RICE':
            gt_add = 'data/' + opt.datasets + '/clear/' + name + '.png'
        elif opt.datasets == 'SateHaze_thin':
            gt_add = 'data/' + opt.datasets + '/target/' + name.split('-')[0] + '-targets.png'
        elif opt.datasets == 'SateHaze_thick':
            gt_add = 'data/' + opt.datasets + '/target/' + name.split('-')[0] + '-targets.png'
        elif opt.datasets == 'SateHaze_moderate':
            gt_add = 'data/' + opt.datasets + '/target/' + name.split('-')[0] + '-targets.png'
        elif opt.datasets == 'school':
            gt_add = 'data/' + opt.datasets + '/clear_school/' + name + '.jpg'

        hazy_img = prepare_image(item)
        gt_img = prepare_gt(gt_add, dataset=opt.datasets)

        dh = Dehaze(name, hazy_img, gt_img, opt)
        dh.optimize()
        dh.finalize()
        psnr = dh.best_result.psnr
        ssim = dh.best_result_ssim.ssim
        write_log(file_name, name, psnr, ssim)
        rec_psnr += psnr
        rec_ssim += ssim

    rec_psnr = rec_psnr / img_num
    rec_ssim = rec_ssim / img_num
    write_log(file_name, 'Average', rec_psnr, rec_ssim)


if __name__ == "__main__":
    dehazing(options)
    print('Zero-Shot Image Dehaze Done!')
