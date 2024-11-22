import os
import warnings
import pdb
import random
import math
from matplotlib import pyplot as plt

import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

'''
LASER-ToF dataset preparation
'''
class Custom(BaseDataset):
    def __init__(self, args, mode):
        super(Custom, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        height, width = (480, 640)
        crop_size = (480, 640)

        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.max_num_pc = 2000

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            536.2517,
            536.3375,
            323.6360,
            250.8662
        ])
        self.fx = 536.2517
        self.fy = 536.3375
        self.cx = 323.6360
        self.cy = 250.8662

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[self.mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_rgb = os.path.join(self.args.dir_data,
                                 self.sample_list[idx]['rgb_name'])
        path_gt = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['gt_name'])
        path_guide = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['guide_name'])

        rgb = Image.open(path_rgb).convert('RGB')

        dep = np.array(Image.open(path_gt))
        dep = dep.astype(np.float32) / 1000.0  # raw depth's unit is (mm)
        dep = Image.fromarray(dep.astype('float32'), mode='F')

        dep_raw_sp = np.array(Image.open(path_guide))
        dep_raw_sp = dep_raw_sp.astype(np.float32) / 1000.0
        dep_raw_sp = Image.fromarray(dep_raw_sp.astype('float32'), mode='F')

        _scale = 1.0

        if self.augment and self.mode == 'train':
            _scale = 1.0  # np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)
            sample_noise = np.random.uniform(0.0, 1.0)
            mask = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                dep_raw_sp = TF.hflip(dep_raw_sp)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)
            dep_raw_sp = TF.rotate(dep_raw_sp, angle=degree, resample=Image.NEAREST)

            t_rgb = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.4, hue=0.1),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
            ])
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            t_dep = T.Compose([
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            dep_raw_sp = t_dep(dep_raw_sp)
            dep_to_train = torch.clamp(dep_raw_sp, min=0, max=self.args.max_depth)

            if sample_noise > 0.85:
                noise = torch.normal(mean=torch.ones_like(dep_to_train), std=0.05)
                dep_to_train = torch.mul(dep_to_train, noise)
            if mask > 0.80:
                hole_min, hole_max = 0.0, 1.0
                method_sel = np.random.uniform(0.0, 1.0)
                if method_sel <= 0.33:
                    hole_min, hole_max = 0.0, 1.0
                elif method_sel > 0.33 and method_sel <= 0.66:
                    hole_min, hole_max = 0.2, 0.4
                elif method_sel > 0.66:
                    hole_min, hole_max = 0.0, 0.5
                rand_mask = self.RandomMask(iw=self.width, ih=self.height, hole_range=[hole_min, hole_max])
                mask_tensor = torch.from_numpy(rand_mask.astype(np.float32))
                dep_to_train = torch.mul(dep_to_train, mask_tensor)

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
        else:
            t_rgb = T.Compose([
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
            ])
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            t_dep = T.Compose([
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            dep_raw_sp = t_dep(dep_raw_sp)
            dep_to_train = torch.clamp(dep_raw_sp, min=0, max=self.args.max_depth)

            K = self.K.clone()

        with torch.no_grad():
            dep_2 = F.max_pool2d(dep, kernel_size=2, stride=2, padding=0)
            dep_4 = F.max_pool2d(dep_2, kernel_size=2, stride=2, padding=0)

        with torch.no_grad():
            if torch.any(torch.isnan(dep_to_train)):
                print(path_guide)
                print("max: {}".format(dep_to_train[0, :, :].max()))
                print("min: {}".format(dep_to_train[0, :, :].min()))
                print("mean: {}".format(dep_to_train[0, :, :].mean()))
                print("norm: {}".format(dep_to_train[0, :, :].norm(1)))

        point_cloud = self.get_pointcloud(dep_to_train, rgb, _scale)

        output = {'rgb': rgb, 'dep': dep_to_train, 'gt': dep, 'K': K, 'gt_2': dep_2, 'gt_4': dep_4, 'pc': point_cloud}

        return output

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel * height * width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))
        # print(mask.nonzero())

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

    def get_pointcloud(self, dep_sp, rgb, scale):
        image_points = dep_sp.nonzero()
        point_list = []
        point_num = image_points.shape[0]
        # torch.set_printoptions(threshold=float('inf'))
        for i in range(point_num):
            u = image_points[i][2].item()
            v = image_points[i][1].item()

            r = rgb[0][v][u].item()
            g = rgb[1][v][u].item()
            b = rgb[2][v][u].item()
            
            depth = dep_sp[0][v][u].item()

            x = depth * (u - self.cx) / (self.fx * scale)
            y = depth * (v - self.cy) / (self.fy * scale)
            z = 1.0 * depth

            v_r = v - 240  # 480/2
            u_r = u - 320  # 640/2

            point_list.append([v_r, u_r, x, y, z, r, g, b])
        if len(point_list) < self.max_num_pc:
            while len(point_list) < self.max_num_pc:
                point_list.append([0, 0, 0, 0, 0, 0, 0, 0])
        elif len(point_list) > self.max_num_pc:
            random.shuffle(point_list)
            while len(point_list) > self.max_num_pc:
                point_list.pop()
        point_tensor = torch.FloatTensor(point_list)
        point_tensor = torch.transpose(point_tensor, 0, 1)

        return point_tensor

    def RandomBrush(
            self,
            max_tries,
            iw,
            ih,
            min_num_vertex=4,
            max_num_vertex=18,
            mean_angle=2 * math.pi / 5,
            angle_range=2 * math.pi / 15,
            min_width=12,
            max_width=48):
        H, W = ih, iw
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 0)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask

    def RandomMask(self, iw, ih, hole_range=None):
        if hole_range is None:
            hole_range = [0.0, 1.0]
        coef = min(hole_range[0] + hole_range[1], 1.0)
        while True:
            mask = np.ones((ih, iw), np.uint8)

            def Fill(max_w, max_h):
                w, h = np.random.randint(max_w), np.random.randint(max_h)
                ww, hh = w // 2, h // 2
                x, y = np.random.randint(-ww, iw - w + ww), np.random.randint(-hh, ih - h + hh)
                mask[max(y, 0): min(y + h, ih), max(x, 0): min(x + w, iw)] = 0

            def MultiFill(max_tries, max_w, max_h):
                for _ in range(np.random.randint(max_tries)):
                    Fill(max_w, max_h)

            MultiFill(int(4 * coef), iw // 2, ih // 2)
            MultiFill(int(2 * coef), iw, ih)
            mask = np.logical_and(mask, 1 - self.RandomBrush(int(8 * coef), iw, ih))  # hole denoted as 0, reserved as 1
            hole_ratio = 1 - np.mean(mask)
            if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
                continue
            return mask[np.newaxis, ...].astype(np.float32)
