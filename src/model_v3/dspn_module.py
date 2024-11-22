import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.ops import deform_conv2d
import math

class Dyspnv6(nn.Module):
    # def __init__(self, args, ch_g, ch_f, k_g, k_f):
    def __init__(self, prop_time=3):
        super(Dyspnv6, self).__init__()
        self.prop_time = prop_time
        self.ch_g = 9
        self.ch_f = 1
        self.k_g = 3
        self.k_f = 3
        pad_g = int((self.k_g - 1) / 2)
        pad_f = int((self.k_f - 1) / 2)
        self.num = self.k_f * self.k_f
        self.conv_offset_aff = nn.Conv2d(
            self.ch_g * self.prop_time, 3 * self.ch_g * self.prop_time, kernel_size=3, stride=1,
            padding=pad_g, bias=True,
            # groups=self.prop_time
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        self.alpha = nn.Parameter(torch.ones(1))

    def _get_offset_affinity(self, guidance):
        B, _, H, W = guidance.shape
        offset_aff = self.conv_offset_aff(guidance).view(B, self.prop_time, self.num * 3, H, W)
        section = [self.num * 2, self.num]
        offset_1, aff_1 = torch.split(offset_aff, section, dim=2)
        aff_1 = torch.softmax(aff_1, dim=2)
        return torch.unbind(offset_1.half(), dim=1), torch.unbind(aff_1.half(), dim=1)

    def _propagate_once(self, feat, offset, aff):
        return deform_conv2d(feat, offset, self.w, self.b, (self.stride, self.stride), (self.padding, self.padding),
                             (self.dilation, self.dilation), mask=aff)  # torch>=1.8

    def forward(self, feat_init, guidance, confidence, feat_fix):
        assert self.ch_f == feat_init.shape[1]
        confidence = feat_fix.sign() * torch.sigmoid(confidence)
        feat_result = feat_init.float().contiguous()

        offset1, aff1 = self._get_offset_affinity(guidance)

        for k in range(self.prop_time):  # 0.002
            feat_result = (1 - confidence) * self._propagate_once(feat_result, offset1[k].float(),
                                                                  aff1[k].float()) + confidence * feat_fix

        return feat_result

