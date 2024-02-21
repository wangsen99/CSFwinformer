import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from timm.models.layers import trunc_normal_
from mmcv.cnn.utils.weight_init import constant_init
from mmseg.ops import resize
import math
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import nchw_to_nlc, nlc_to_nchw
from .dct import dct_layer, reverse_dct_layer, check_and_padding_imgs, remove_image_padding, resize_flow


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = self.q(q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SFA(nn.Module):
    """
    RGB_imgs, (n, c, h, w)

    """
    def __init__(self, dct_kernel=8, window_size=8, input_size=16, num_heads=8, dim=1024, mlp_ratio=2., drop_path=0.):
        super(SFA, self).__init__()
        self.dct_kernel = dct_kernel
        self.window_size = window_size
        self.num_heads = num_heads
        self.output_size = math.ceil(input_size/dct_kernel) * dct_kernel

        self.dct = dct_layer(in_c=dim, h=dct_kernel, w=dct_kernel)
        self.rdct = reverse_dct_layer(out_c=dim, h=dct_kernel, w=dct_kernel)
        self.fold = nn.Fold(output_size=(self.output_size, self.output_size), kernel_size=(dct_kernel, dct_kernel), stride=dct_kernel)
        self.unfold = nn.Unfold(kernel_size=(dct_kernel, dct_kernel), stride=dct_kernel)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.attn1 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None)
        
        self.attn2 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None)

        # self.enhanced = EnhancedAttention(dim=dim, num_heads=self.num_heads)
        self.concat = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, feat, paddings):
        B, C, H, W = feat.shape

        padding_h, padding_w = paddings
        assert padding_h==0 and padding_w==0

        x_spac = feat

        # dct
        x_freq = self.dct(feat) #B C*64 H/8 W/8
        B_f, C_f, H_f, W_f = x_freq.shape
        x_freq = x_freq.flatten(2) # B C*64 H/8*W/8
        x_freq = self.fold(x_freq) # B C H W

        x_freq = self.norm1(x_freq)
        x_spac = self.norm2(x_spac)

        x_freq = x_freq.permute(0, 2, 3, 1)
        x_spac = x_spac.permute(0, 2, 3, 1)

        # partition windows
        x_freq_windows = window_partition(x_freq, self.window_size)  # nW*B, window_size, window_size, C
        x_freq_windows = x_freq_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_spac_windows = window_partition(x_spac, self.window_size)  # nW*B, window_size, window_size, C
        x_spac_windows = x_spac_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # space to frequency
        stf_windows = self.attn1(x_spac_windows, x_freq_windows, x_freq_windows)  # nW*B, window_size*window_size, C

        # merge windows
        stf_windows = stf_windows.view(-1, self.window_size, self.window_size, C)
        stf = window_reverse(stf_windows, self.window_size, H, W)  # B H' W' C
        stf = stf.view(B, H * W, C)

        #rdct
        stf = stf.transpose(1, 2).reshape(B, C, H, W)
        stf = self.unfold(stf) # B C*64 H/8*W/8
        stf = stf.view(B_f, C_f, H_f, W_f)
        stf = self.rdct(stf) #B C H W

        # frequency to space
        fts_windows = self.attn2(x_freq_windows, x_spac_windows, x_spac_windows)

        # merge windows
        fts_windows = fts_windows.view(-1, self.window_size, self.window_size, C)
        fts = window_reverse(fts_windows, self.window_size, H, W)  # B H' W' C
        fts = fts.view(B, H * W, C)
        fts = fts.transpose(1, 2).reshape(B, C, H, W)

        #fusion
        # out = self.enhanced(stf, fts, feat)
        out = torch.cat((stf, fts),dim=1)
        out = self.concat(out)

        # residual
        shortcut = feat
        # shortcut = remove_image_padding(shortcut, padding_h, padding_w)

        out = shortcut + self.drop_path(out)
        out = nchw_to_nlc(out) #B H*W C
        out = out + self.drop_path(self.mlp(self.norm3(out))) # B, H * W, C
        out = nlc_to_nchw(out, [H, W])

        return out

class Contextcontrast(nn.Module):
    def __init__(self, dct_kernel=8, window_size=8, input_size=16, num_heads=8, dim=1024, mlp_ratio=2., drop_path=0.):
        super(Contextcontrast, self).__init__()
        self.dct_kernel = dct_kernel
        self.window_size = window_size
        self.num_heads = num_heads

        self.u1 = nn.Unfold(kernel_size=self.dct_kernel, dilation=input_size//self.dct_kernel, padding=0, stride=1)
        self.f1 = nn.Fold(output_size=input_size, kernel_size=self.dct_kernel, dilation=1, padding=0, stride=self.dct_kernel)
        self.norm1 = LayerNorm2d(dim)
        self.re_u = nn.Unfold(kernel_size=self.dct_kernel, dilation=1, padding=0, stride=self.dct_kernel)
        self.re_f = nn.Fold(output_size=input_size, kernel_size=self.dct_kernel, dilation=input_size//self.dct_kernel, padding=0, stride=1)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.norm3 = LayerNorm2d(dim)
        self.act = nn.GELU()

        self.concat = nn.Sequential(
            nn.Conv2d(dim * 3 , dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
    def forward(self, feat, local):
        B, C, H, W = feat.shape
        dilated = self.f1(self.u1(feat))
        dilated = self.norm1(dilated)
        dilated = dilated.permute(0, 2, 3, 1)
        dilated_window = window_partition(dilated, self.window_size)
        dilated_window = dilated_window.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        context_windows = self.attn(dilated_window, dilated_window, dilated_window)

        context_windows = context_windows.view(-1, self.window_size, self.window_size, C)
        context = window_reverse(context_windows, self.window_size, H, W)  # B H' W' C
        context = context.view(B, H * W, C)
        context = context.transpose(1, 2).reshape(B, C, H, W)

        context = self.re_f(self.re_u(context))

        # residual
        shortcut = feat
        # shortcut = remove_image_padding(shortcut, padding_h, padding_w)

        context = shortcut + self.drop_path(context)
        context = nchw_to_nlc(context) #B H*W C
        context = context + self.drop_path(self.mlp(self.norm2(context))) # B, H * W, C
        context = nlc_to_nchw(context, [H, W])

        cc = self.act(self.norm3(local - context))

        out = self.concat(torch.cat((local, context, cc),dim=1))

        return out


class cascade_fusion(nn.Module):
    def __init__(self, channel, high_channel):
        super(cascade_fusion, self).__init__()

        self.conv_h = nn.Sequential(
            nn.Conv2d(high_channel, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_l = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, high, low):
        hw = low.shape[2:]
        high = self.conv_h(high)
        high = resize(
            high, size=hw, mode='bilinear', align_corners=False
        )
        out = low + high
        out = self.conv_l(out)
        return out

@HEADS.register_module()
class MirrorDecoder(BaseDecodeHead):
    def __init__(self, dct_kernel=(8,8), num_heads=[3, 6, 12, 24], img_size=224, drop_path=0., **kwargs):
        super(MirrorDecoder, self).__init__(input_transform='multiple_select', **kwargs)

        self.img_size = img_size
        self.dct_kernel = dct_kernel

        # compute the relationship between space and frequency
        self.spac_freq_32 = SFA(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//32, num_heads=num_heads[3], dim=self.in_channels[3], drop_path=drop_path)
        self.spac_freq_16 = SFA(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//16, num_heads=num_heads[2], dim=self.in_channels[2], drop_path=drop_path)
        self.spac_freq_8 = SFA(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//8, num_heads=num_heads[1], dim=self.in_channels[1], drop_path=drop_path)
        self.spac_freq_4 = SFA(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//4, num_heads=num_heads[0], dim=self.in_channels[0], drop_path=drop_path)
        
        # context contrast
        self.cc_32 = Contextcontrast(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//32, num_heads=num_heads[3], dim=self.in_channels[3], drop_path=drop_path)
        self.cc_16 = Contextcontrast(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//16, num_heads=num_heads[2], dim=self.in_channels[2], drop_path=drop_path)
        self.cc_8 = Contextcontrast(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//8, num_heads=num_heads[1], dim=self.in_channels[1], drop_path=drop_path)
        self.cc_4 = Contextcontrast(dct_kernel=self.dct_kernel[0], window_size=self.dct_kernel[0], input_size=img_size//4, num_heads=num_heads[0], dim=self.in_channels[0], drop_path=drop_path)


        # self.de_32 = CCL(self.in_channels[3], dilation=2)
        self.de_16 = cascade_fusion(channel=self.in_channels[2], high_channel=self.in_channels[3])
        self.de_8 = cascade_fusion(channel=self.in_channels[1], high_channel=self.in_channels[2])
        self.de_4 = cascade_fusion(channel=self.in_channels[0], high_channel=self.in_channels[1])


        # predict mirror maps
        self.pre_1_32 = nn.Conv2d(self.in_channels[3], self.num_classes, 1)
        self.pre_1_16 = nn.Conv2d(self.in_channels[2], self.num_classes, 1)
        self.pre_1_8 = nn.Conv2d(self.in_channels[1], self.num_classes, 1)
        self.pre_1_4 = nn.Conv2d(self.in_channels[0], self.num_classes, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = []
        for seg_logit in seg_logits:
            losses.append(self.losses(seg_logit, gt_semantic_seg))
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        # 1 C H W
        return self.forward(inputs)[-1]    

    def forward(self, inputs):

        results = []
        feat_4, feat_8, feat_16, feat_32 = inputs[0], inputs[1], inputs[2], inputs[3]

        # space and frequency
        feat_32, padding_h, padding_w = check_and_padding_imgs(feat_32, self.dct_kernel)
        fea_1_32 = self.spac_freq_32(feat_32, [padding_h, padding_w]) # B, h_32+padd*w_32+padd, 768
        
        cc_32 = self.cc_32(feat_32, fea_1_32)


        mask_1_32 = self.pre_1_32(cc_32)# B, 768, h_32, w_32
        results.append(mask_1_32)

        # space and frequency
        feat_16, padding_h, padding_w = check_and_padding_imgs(feat_16, self.dct_kernel)
        fea_1_16 = self.spac_freq_16(feat_16, [padding_h, padding_w]) # B, h_16+padd * w_16+padd, 384

        cc_16 = self.cc_16(feat_16, fea_1_16)

        de_16 = self.de_16(cc_32, cc_16)

        # predict mirror maps
        mask_1_16 = self.pre_1_16(de_16)
        results.append(mask_1_16)

        # space and frequency
        # if width or height is not divisible by dct_kernel, may cause some mistakes during backward
        feat_8, padding_h, padding_w = check_and_padding_imgs(feat_8, self.dct_kernel)
        fea_1_8 = self.spac_freq_8(feat_8, [padding_h, padding_w])

        # ccl
        cc_8 = self.cc_8(feat_8, fea_1_8)

        #cascade
        de_8 = self.de_8(de_16, cc_8)

        # predict mirror maps
        mask_1_8 = self.pre_1_8(de_8)
        results.append(mask_1_8)

        # space and frequency
        feat_4, padding_h, padding_w = check_and_padding_imgs(feat_4, self.dct_kernel)
        fea_1_4 = self.spac_freq_4(feat_4, [padding_h, padding_w]) # B, h_8+padd * w_8+padd, 96

        # ccl
        cc_4 = self.cc_4(feat_4, fea_1_4)

        #cascade
        de_4 = self.de_4(de_8, cc_4)

        # predict mirror maps
        mask_1_4 = self.pre_1_4(de_4)
        results.append(mask_1_4)


        # final_features = torch.cat((mask_1_4, mask_1_8, mask_1_16, mask_1_32),1)
        # final_predict = self.refinement(final_features)
        # results.append(final_predict)

        return tuple(results)

if __name__ == '__main__':
    model = SFA()
    img = torch.randn(4, 1024, 16, 16)
    # x = model(img)
    # print(x.shape)
    # bicubic_imgs, padding_h, padding_w = check_and_padding_imgs(img, [8,8])
    # print(padding_h)
    # t = 10
    # for i in range(t-1, -1, -1):
    #     print(i, '\n')
