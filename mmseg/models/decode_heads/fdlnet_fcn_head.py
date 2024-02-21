# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .frelayer import MultiSpectralAttentionLayer

@HEADS.register_module()
class FDLNetFCNHead(BaseDecodeHead):
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FDLNetFCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.att = MultiSpectralAttentionLayer(self.in_channels, dct_h=8, dct_w=8, frenum=8)

        self.fam =  _FAHead(in_ch=self.in_channels, inter_channels=self.channels, norm_layer=nn.SyncBatchNorm, **kwargs)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        midput = self._forward_feature(inputs)
        freput = inputs[-1]

        fre = self.att(freput)

        output = self.fam(midput, fre) 
        output = self.cls_seg(output)
        return output

class _FAHead(nn.Module):
    def __init__(self, in_ch, inter_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FAHead, self).__init__()
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(in_ch, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.freatt = _FreqAttentionModule()
        # self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
        )

    def forward(self, x, fre):
        feat_x = self.conv_x1(x)
        feat_f = self.conv_f1(fre)

        feat_p = self.freatt(feat_x, feat_f)
        feat_p = self.conv_p2(feat_p)

        return feat_p

class _FreqAttentionModule(nn.Module):
    """ attention module"""

    def __init__(self):
        super(_FreqAttentionModule, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, fre):
        batch_size, _, height, width = x.size()
        fre = fre.expand_as(x)
        feat_a = x.view(batch_size, -1, height * width) #B C H*W
        feat_f_transpose = fre.view(batch_size, -1, height * width).permute(0, 2, 1) #B H*W C
        attention = torch.bmm(feat_a, feat_f_transpose)  # B C C
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new) # B C C

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width) # B C H*W
        # out = feat_e
        out = self.alpha*feat_e + x
        # print(self.alpha)
        return out