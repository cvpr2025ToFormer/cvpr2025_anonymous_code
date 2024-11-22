import torch.nn as nn

def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
        )

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)
    padding = int(kernel / 2.0)
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv3x3(
                    in_planes if (i == 0) else out_planes,
                    out_planes,
                    stride=1,
                    bias=False,
                ),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x
        return x
