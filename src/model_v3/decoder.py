import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile
from torch.autograd import Variable
import torch.nn.functional as F

from .layer_factory import conv1x1, conv3x3, convbnrelu, CRPBlock, Conv3x3
from .coordatt import CoordAtt
from .resnet_cbam import BasicBlock

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(
            convbnrelu(in_planes, intermed_planes, 1),
            convbnrelu(
                intermed_planes,
                intermed_planes,
                3,
                stride=stride,
                groups=intermed_planes,
            ),
            convbnrelu(intermed_planes, out_planes, 1, act=False),
        )

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return out + residual
        else:
            return out

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class DepthDecoderCF(nn.Module):
    def __init__(self):
        super(DepthDecoderCF, self).__init__()
        self.num_neighbors = 8
        channels = [64, 96, 96, 128, 160]
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_bn_relu(channels[4], 96, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(96, 96, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_bn_relu(96+channels[3], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=8),
        )

        # 1/4
        self.dec4 = nn.Sequential(
            convt_bn_relu(64 + channels[2], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/2
        self.dec3 = nn.Sequential(
            convt_bn_relu(64 + channels[1], 48, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(48, 48, stride=1, downsample=None, ratio=4),
        )

        # 1/1
        self.dec2 = nn.Sequential(
            convt_bn_relu(48 + channels[0], 48, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(48, 48, stride=1, downsample=None, ratio=4),
        )

        # Init Depth Branch
        # 1/1
        self.dep_dec1 = conv_bn_relu(48 + 64, 64, kernel=3, stride=1,
                                     padding=1)
        self.dep_dec0 = conv_bn_relu(64, 1, kernel=3, stride=1,
                                     padding=1, bn=False, relu=True)

        self.conv_disp1 = Conv3x3(64, 1)
        self.conv_disp2 = Conv3x3(64, 1)
        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(48 + 64, 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64, 27, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        self.cf_dec1 = conv_bn_relu(48 + 64, 32, kernel=3, stride=1,
                                        padding=1)
        self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Sigmoid()
            )

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f



    def forward(self, input_features):
        fe3 = input_features[0]
        fe4 = input_features[1]
        fe5 = input_features[2]
        fe6 = input_features[3]
        fe7 = input_features[4]

        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))

        # Init Depth Decoding
        dep_fd1 = self.dep_dec1(torch.concat((fd3, fe3), dim=1))
        init_depth = self.dep_dec0(dep_fd1)
        output_2 = self.conv_disp1(fd4)
        output_4 = self.conv_disp2(fd5)

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(torch.concat((fd3, fe3), dim=1))
        guide = self.gd_dec0(gd_fd1)

        cf_fd1 = self.cf_dec1(torch.concat((fd3, fe3), dim=1))
        confidence = self.cf_dec0(cf_fd1)

        return init_depth, output_2, output_4, guide, confidence
