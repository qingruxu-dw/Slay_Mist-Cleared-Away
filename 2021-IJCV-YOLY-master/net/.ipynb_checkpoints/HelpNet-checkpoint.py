import torch
from torch import nn
from einops import rearrange
import numbers
import torch.nn.functional as F
#创建自定义GELU类,将所有 nn.GELU 替换为 GELU
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Attention(nn.Module):
    def __init__(self, dim, num_heads, is_prompt=False, bias=True):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.is_prompt = is_prompt
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.prompt = nn.Parameter(torch.ones(num_heads, dim//num_heads, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def with_prompt(self, tensor, prompt):
        return tensor if prompt is None else tensor + prompt

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        if self.is_prompt:
            prompt = self.prompt
            q = self.with_prompt(q, prompt)
            k = self.with_prompt(k, prompt)
            v = self.with_prompt(v, prompt)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_Key_Value(nn.Module,):
    def __init__(self, dim, num_heads, bias, is_prompt=False):
        super(Attention_Key_Value, self).__init__()
        self.num_heads = num_heads
        self.is_prompt = is_prompt
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.prompt = nn.Parameter(torch.ones(num_heads, dim // num_heads, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.v_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def with_prompt(self, tensor, prompt):
        return tensor if prompt is None else tensor + prompt

    def forward(self, x, feature1, feature2):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.k_dwconv(torch.cat([k, feature1], dim=1))
        v = self.v_dwconv(torch.cat([v, feature2], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        if self.is_prompt:
            prompt = self.prompt
            q = self.with_prompt(q, prompt)
            k = self.with_prompt(k, prompt)
            v = self.with_prompt(v, prompt)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention_test1(nn.Module):
    def __init__(self, dim, dim_pre, num_heads, is_prompt=False, bias=True):
        super(Attention_test1, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.is_prompt = is_prompt
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.prompt = nn.Parameter(torch.ones(num_heads, dim//num_heads, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.layernorm = LayerNorm(dim_pre)
        self.conv_y = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))
    def with_prompt(self, tensor, prompt):
        return tensor if prompt is None else tensor + prompt

    def forward(self, x, y):

        x = self.layernorm(x)
        y = self.layernorm(y)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        y_q = self.conv_y(y)
        q_concat = self.q_dwconv(torch.cat([q, y_q], dim=1))
        q = rearrange(q_concat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        if self.is_prompt:
            prompt = self.prompt
            q = self.with_prompt(q, prompt)
            k = self.with_prompt(k, prompt)
            v = self.with_prompt(v, prompt)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



class Attention_Qeury(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Qeury, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = self.q_dwconv(torch.cat([q, feature], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock_Query(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_Query, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Qeury = Attention_Qeury(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, feature):
        # print('dim',self.dim)
        x = x + self.attn_Qeury(self.norm1(x), feature)
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock_Key_Value(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock_Key_Value, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Key_Value = Attention_Key_Value(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, feature1, feature2):
        # print('dim',self.dim)
        x = x + self.attn_Key_Value(self.norm1(x), feature1, feature2)
        x = x + self.ffn(self.norm2(x))

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        # print('dim', dim)
        # print('hidden_features', hidden_features)

        self.dwconv1 = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=dim, bias=bias)

        self.project_middle = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x = self.project_in(x)
        x1 = self.dwconv1(x)
        x1 = self.project_middle(x1)
        x1 = F.gelu(x1)
        x2 = self.dwconv2(x1)
        x = self.project_out(x2)
        return x

class TransformerBlock_QKV(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_QKV, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Qeury = Attention_Qeury(dim, num_heads, bias)
        self.attn_Key_Value = Attention_Key_Value(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature1, feature2, feature3):
        # print('dim',self.dim)
        attn_Qeury = self.attn_Qeury(self.norm1(x), feature1)
        attn_KV = self.attn_Key_Value(self.norm1(x), feature2, feature3)

        x = x + self.fusion(torch.cat([attn_Qeury, attn_KV], dim=1))
        x = x + self.ffn(self.norm2(x))

        return x

class Global_Perception(nn.Module):
    def __init__(self, dim, dim_pre, num_heads=8, depth=2, res=(128, 128), pooling_r=4,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True, bias=True,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        # self.transformation = Perception_transformation(dim=dim, pooling_r=pooling_r, soft_perception=soft_perception,
        #                                              hard_perception=hard_perception, bias=bias)

        #######################################################################################################
        self.global_degregation_aware_restore_aware = global_degregation_aware_restore_aware
        self.global_degregation_aware = global_degregation_aware
        self.global_restore_aware = global_restore_aware

        if self.global_degregation_aware_restore_aware is True:
            self.Attention_qkv = TransformerBlock_QKV(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 3, kernel_size=1, stride=1),
                                            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim,
                                                      bias=True))
        elif self.global_degregation_aware is True:
            self.Attention_q = TransformerBlock_Query(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.q_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 2, kernel_size=1, stride=1),
                                          nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim,
                                                    bias=True))
        elif self.global_restore_aware is True:
            self.Attention_kv = TransformerBlock_Key_Value(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 2, kernel_size=1, stride=1),
                                            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim,
                                                      bias=True))

    def forward(self, x, feature_perception=None):
        b, c, h, w = x.size()
        if feature_perception is not None:
            if self.global_degregation_aware_restore_aware is True:
                qkv = self.qkv_dwconv(self.layernorm(feature_perception))
                q_dwconv, k_dwconv, v_dwconv = qkv.chunk(3, dim=1)
                Attention_qkv = self.Attention_qkv(x, feature1=q_dwconv, feature2=k_dwconv,
                                                   feature3=v_dwconv)
                return Attention_qkv
            elif self.global_degregation_aware is True:
                q = self.q_dwconv(self.layernorm(feature_perception))
                Attention_q = self.Attention_q(x, feature=q)
                return Attention_q
            elif self.global_restore_aware is True:
                kv = self.qkv_dwconv(self.layernorm(feature_perception))
                k_dwconv, v_dwconv = kv.chunk(2, dim=1)
                Attention_kv = self.Attention_kv(x, feature1=k_dwconv, feature2=v_dwconv)
                return Attention_kv
        else:
            return x

class FeatureProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.sca1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64)
        )
        self.sca2 = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.BatchNorm2d(64)
        )
        self.sca3 = nn.Sequential(
            nn.Conv2d(512, 64, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=8, mode='nearest'),
            nn.BatchNorm2d(64)
        )
        self.sca4 = nn.Sequential(
            nn.Conv2d(1024, 64, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=16, mode='nearest'),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        out = []
        out.append(x[0])
        scale1 = self.sca1(x[1]); out.append(scale1)
        scale2 = self.sca2(x[2]); out.append(scale2)
        scale3 = self.sca3(x[3]); out.append(scale3)
        scale4 = self.sca4(x[4]); out.append(scale4)
        return out

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class TransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # print('dim',self.dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class GLoba(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(GLoba, self).__init__()
        self.num_heads = num_heads

        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.k_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.v_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.project_out = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)


    def forward(self, x, y):

        b, c, h, w = x.shape
        qkv1 = self.qkv_dwconv1(self.qkv1(x))
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        qkv2 = self.qkv_dwconv2(self.qkv2(y))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q_concat = self.q_dwconv(torch.cat([q1, q2], dim=1))
        k_concat = self.k_dwconv(torch.cat([k1, k2], dim=1))
        v_concat = self.v_dwconv(torch.cat([v1, v2], dim=1))

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_concat = rearrange(q_concat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_concat = rearrange(k_concat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_concat = rearrange(v_concat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        attn1 = (q_concat @ k1.transpose(-2, -1)) * self.temperature1
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v1)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out1 = self.project_out1(out1)

        attn2 = (q1 @ k_concat.transpose(-2, -1)) * self.temperature2
        attn2 = attn2.softmax(dim=-1)

        out2 = (attn2 @ v_concat)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out2 = self.project_out2(out2)

        out = torch.cat([out1, out2], dim=1)

        out = self.project_out(out)

        return out



class Local_Perception(nn.Module):
    def __init__(self, dim, dim_pre, bias=True):
        super(Local_Perception, self).__init__()
        self.degradation = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
        ###############################
        self.input = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )
        self.main_kernel = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), GELU(),
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
        )
        self.degradation_kernel = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), GELU(),
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
        )
        self.fusion1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=bias)
        self.layernorm1 = LayerNorm(dim_pre)
        self.layernorm2 = LayerNorm(dim)

    def forward(self, x, feature_perception=None):
        b, c, h, w = x.shape
        ############################### Soft_Concert ###############################
        degradation = self.degradation(self.layernorm1(feature_perception))
        degradation1 = degradation[:, c:, :, :]  #q
        degradation2 = degradation[:, :c, :, :]  #k
        input = self.input(self.layernorm2(x))
        input1 = input[:, c:, :, :]  # q
        input2 = input[:, :c, :, :]  # k
        ###############################

        main_kernel = self.main_kernel(torch.cat([input1, degradation1], dim=1))
        main_kernel = F.sigmoid(main_kernel)
        main_kernel_mul = torch.mul(main_kernel, degradation2)
        ###############################

        degradation_kernel = self.degradation_kernel(torch.cat([input2, degradation2], dim=1))
        degradation_kernel = F.sigmoid(degradation_kernel)
        degradation_kernel_mul = torch.mul(degradation_kernel, input1)

        out = self.fusion1(torch.cat([degradation_kernel_mul, main_kernel_mul], dim=1)) + x

        return out


class Edge_Attention_Layer(nn.Module):
    def __init__(self, channel):
        super(Edge_Attention_Layer, self).__init__()

        # #结构1
        # self.denoise = nn.Sequential(
        #     nn.Conv2d(channel, channel, 1, bias=True),
        #     nn.Conv2d(channel, channel, 3, padding=1 ,bias=True),
        #     nn.Conv2d(channel, channel, 1, bias=True),
        # )
        # 结构2

        self.denoise = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.Conv2d(channel, channel, 1, bias=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
        )

        self.XConv = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
        )
        self.YConv = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )
        self.InfoConv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.InstanceNorm2d(channel),
            # Relu可以注释掉
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input, feature):

        feature = self.denoise(feature)
        x = self.XConv(feature)
        y = self.YConv(feature)
        ea = x + y
        info = self.InfoConv(input)
        # ckp.saveAttentionMapInBlocks(x, 'x')
        # ckp.saveAttentionMapInBlocks(y, 'y')
        # ckp.saveAttentionMapInBlocks(ea, 'ea')
        return input + info*ea



if __name__ == '__main__':
    x = torch.randn(1, 64, 446, 512)
    y = torch.randn(1, 64, 446, 512)
    model = Edge_Attention_Layer(64)
    out = model(x, y)
    print(out.shape)

