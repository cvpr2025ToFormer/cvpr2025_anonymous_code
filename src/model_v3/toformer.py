import torch
import torch.nn as nn
import numpy
import pdb
from torchinfo import summary
from thop import profile
from torch.autograd import Variable

from .encoder import DepthEncoder
from .decoder import DepthDecoderCF
from .dspn_module import Dyspnv6

class ToFormer(nn.Module):
    def __init__(self):
        super(ToFormer, self).__init__()
        self.encoder = DepthEncoder(model='toformer-v2-s')
        self.decoder = DepthDecoderCF()
        self.dyspn = Dyspnv6(prop_time=3)

    def forward(self, sample, flag=False):
        rgb = sample['rgb']
        dep = sample['dep']
        input_pc = sample['pc']

        features = self.encoder(rgb, dep, input_pc)

        pred_init, pred_2, pred_4, guide, conf = self.decoder(features)
        pred_final = self.dyspn(pred_init, guide, conf, dep)

        guide = torch.zeros_like(pred_init)
        y_inter = [pred_init, ]
        y = pred_final
        y = torch.clamp(y, min=0.0)
        offset, aff, aff_const = torch.zeros_like(y), torch.zeros_like(y), torch.zeros_like(y).mean()

        output = {'pred': y, 'pred_init': pred_init, 'pred_2': pred_2, 'pred_4': pred_4,
                  'pred_inter': y_inter, 'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': None}

        return output
