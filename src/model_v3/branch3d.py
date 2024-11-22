import os
import pdb
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchinfo import summary
import DCNv4

from .coordatt import CoordAtt

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
                                     # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class Branch3d_stage0(nn.Module):
    def __init__(self, k, in_chans, feat_dim, feat_h, feat_w):
        super(Branch3d_stage0, self).__init__()
        self.k = k
        self.in_chans = in_chans
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.feat_dim = feat_dim

        self.coordatt = CoordAtt(inp=24, oup=24)
        self.preConv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_chans, out_channels=24, kernel_size=3, stride=1, padding=1),
            self.coordatt
        )

        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear4 = nn.Sequential(nn.Linear(256, 256, bias=False),
                                     nn.Linear(256, 64, bias=False))

        self.dcn5 = nn.Sequential(DCNv4.DCNv4(channels=64, kernel_size=3),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.dcn6 = nn.Sequential(DCNv4.DCNv4(channels=64, kernel_size=3),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.softmax = nn.Softmax(dim=1)

        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.Conv2d(64, self.feat_dim, kernel_size=1, bias=False))

    def forward(self, pc_xyzrgb, feat_s00):
        batch_size = pc_xyzrgb.shape[0]
        num_points = pc_xyzrgb.shape[2]
        v = pc_xyzrgb[:, 0:1, :].clone().detach()
        u = pc_xyzrgb[:, 1:2, :].clone().detach()
        pc_t = torch.transpose(pc_xyzrgb, 1, 2)

        batch_list = list(range(batch_size))
        batch_index = torch.LongTensor(batch_list).unsqueeze(dim=1)
        c = torch.zeros(1, num_points).long()
        batch_index = batch_index + c
        batch_index = batch_index.view(-1).detach()

        v = torch.squeeze(v, dim=1)
        u = torch.squeeze(u, dim=1)
        v = (torch.floor(v.add(240))).long()
        u = (torch.floor(u.add(320))).long()
        v = v.view(-1)
        u = u.view(-1)
        index = (batch_index,
                 v,
                 u)

        feat_s1 = self.preConv(feat_s00)

        feat_s1 = feat_s1.permute(0, 2, 3, 1)
        feat_2d = feat_s1[index[0], index[1], index[2]]

        feat_2d = feat_2d.reshape((batch_size, num_points, 24))
        feat_3d = torch.concat((pc_t, feat_2d), dim=2)
        feat_3d = torch.transpose(feat_3d, 1, 2)

        feat_3d = get_graph_feature(feat_3d, k=self.k)
        feat_3d = self.conv1(feat_3d)
        x1 = feat_3d.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = torch.transpose(x, 1, 2)
        x = self.linear4(x)

        feat_map = torch.zeros(batch_size, self.feat_h, self.feat_w, 64).cuda()  # (B, 1/2, 128)

        index1 = (index[0],
                  torch.div(index[1], 2, rounding_mode='floor'),
                  torch.div(index[2], 2, rounding_mode='floor'))
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        feat_map.index_put_(index1, x, accumulate=True)
        feat_map = feat_map.permute(0, 3, 1, 2)  # ->(B,C,H,W)

        feat_map = self.softmax(feat_map)  # B C H W
        feat_map = self.dcn5(feat_map)
        feat_map = self.dcn6(feat_map)
        feat_map = self.conv7(feat_map)

        return feat_map, index1
