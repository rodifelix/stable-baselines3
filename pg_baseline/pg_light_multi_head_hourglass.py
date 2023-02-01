################################################################################
################################################################################
#Replace this paragraph by a short description of your program.
#
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Angel Martinez-Gonzalez <angel.martinez@idiap.ch>,
#Modified by Marco Ewerton
#
#This file is part of ResidualPose.
#
#ResidualPose is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 3 as
#published by the Free Software Foundation.
#
#ResidualPose is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with ResidualPose. If not, see <http://www.gnu.org/licenses/>.
################################################################################


import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import flatten
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn import (
    ELU,
    ReLU,
    Module,
    Linear,
    Conv2d,
    Sigmoid,
    Upsample,
    MaxPool2d,
    InstanceNorm2d,
)

class Push_Into_Box_Net(Module):
    def __init__(
        self,
        device,
        img_shape = [224, 224],
        num_out_channels = 8,
        num_inp_feat_channels = 64,
        num_out_feat_channels = 32,
        positional_encoding = False,
    ):

        super(Push_Into_Box_Net, self).__init__()

        self.device = device
        self.img_shape = img_shape
        self.pos_encoding = positional_encoding
        self.num_out_channels = num_out_channels
        self.num_inp_feat_channels = num_inp_feat_channels
        self.num_out_feat_channels = num_out_feat_channels

        # Number of input channels
        self.num_input_channels = 1  # Depth image
        if self.pos_encoding:
            self.num_input_channels = 3 # Depth image + x,y coordinates

        # Network: small hourglass network
        self.network = Network_hg(
            self.num_inp_feat_channels,
            self.num_out_feat_channels,
        )

        # Preprocessing layers
        self.preprocess = Sequential(
            Conv2d(self.num_input_channels, 32, 5, stride=1, padding=2),
            ReLU(),
            InstanceNorm2d(32, affine=True),
            Conv2d(32, self.num_inp_feat_channels, 3, stride=1, padding=1),
            ReLU(),
            InstanceNorm2d(self.num_inp_feat_channels, affine=True),
        )

        # Mask head
        self.head_mask = Sequential(
            Conv2d(self.num_out_feat_channels, 24, 3, stride=1, padding=1),
            ReLU(),
            InstanceNorm2d(24, affine=True),
            Conv2d(24, self.num_out_channels, 3, stride=1, padding=1),
            Sigmoid()
        )

        # Reward head
        self.head_reward = Sequential(
            Conv2d(self.num_out_feat_channels, 24, 3, stride=1, padding=1),
            ReLU(),
            InstanceNorm2d(24, affine=True),
            Conv2d(24, self.num_out_channels, 3, stride=1, padding=1)
        )

        # Spatial/positional encoding
        if self.pos_encoding:
            indices = np.indices((img_shape[0], img_shape[1]))
            u_coord = indices[0] / (img_shape[0] - 1)
            v_coord = indices[1] / (img_shape[1] - 1)
            u_coord = torch.from_numpy(u_coord)
            v_coord = torch.from_numpy(v_coord)
            u_coord = u_coord.view(1, 1, img_shape[0], img_shape[1])
            v_coord = v_coord.view(1, 1, img_shape[0], img_shape[1])
            self.uv_coords = torch.cat((u_coord, v_coord), 1)

    def forward(self, x):

        # Batch size
        bs = x.size()[0]

        # Add spatial/positional encoding
        if self.pos_encoding:
            uv_encoding = self.uv_coords.repeat(bs, 1, 1, 1).float().to(self.device)
            x = torch.cat((x, uv_encoding), 1)

        # Preprocessing layers
        x = self.preprocess(x)

        # Network
        out_net = self.network(x)

        # Multi-heads outputs
        out_mask = self.head_mask(out_net)
        out_reward = self.head_reward(out_net)

        mask = out_mask >= 0.14
        mask = mask.float() - 1.
        mask = mask * torch.finfo(torch.float).max

        return [out_reward+mask, out_reward, out_mask]

class Network_hg(Module):

    def __init__(
        self,
        num_inp_feat_channels,
        num_out_feat_channels,
    ):

        super(Network_hg, self).__init__()

        # Number of input/output feature channels
        self.num_inp_feat_channels = num_inp_feat_channels
        self.num_out_feat_channels = num_out_feat_channels

        # Top-down convolutional layers
        self.B1 = Sequential(
            Conv2d(self.num_inp_feat_channels, 64, 3, stride=1, padding=1),
            ReLU(True),
            InstanceNorm2d(64, affine=True)
        )
        self.B2 = Sequential(
            Conv2d(64, 64, 3, stride=1, padding=1),
            ReLU(True),
            InstanceNorm2d(64, affine=True)
        )
        self.B3 = Sequential(
            Conv2d(64, 64, 3, stride=1, padding=1),
            ReLU(True),
            InstanceNorm2d(64, affine=True)
        )
        self.B4 = Sequential(
            Conv2d(64, 128, 3, stride=1, padding=1),
            ReLU(True),
            InstanceNorm2d(128, affine=True)
        )
        self.B5 = Sequential(
            Conv2d(128, 128, 3, stride=1, padding=1),
            ReLU(True),
            InstanceNorm2d(128, affine=True)
        )

        # Max pooling layers
        self.P1 = MaxPool2d(kernel_size=2, stride=2)
        self.P2 = MaxPool2d(kernel_size=2, stride=2)
        self.P3 = MaxPool2d(kernel_size=2, stride=2)
        self.P4 = MaxPool2d(kernel_size=2, stride=2)

        # Upsampling layers
        self.U4 = Upsample(scale_factor=2, mode='bilinear')
        self.U3 = Upsample(scale_factor=2, mode='bilinear')
        self.U2 = Upsample(scale_factor=2, mode='bilinear')
        self.U1 = Upsample(scale_factor=2, mode='bilinear')

        # Skip layers
        self.S4 = Sequential(
            Conv2d(128, 64, 1, 1, padding=0),
            ReLU(),
            InstanceNorm2d(64, affine=True)
        )
        self.S3 = Sequential(
            Conv2d(64, 64, 1, 1, padding=0),
            ReLU(),
            InstanceNorm2d(64, affine=True)
        )
        self.S2 = Sequential(
            Conv2d(64, 32, 1, 1, padding=0),
            ReLU(),
            InstanceNorm2d(32, affine=True)
        )

        # Bottom-up convolutional layers
        self.R4 = Sequential(
            Conv2d(192, 64, 3, 1, padding=1),
            ReLU(),
            InstanceNorm2d(64, affine=True)
        )
        self.R3 = Sequential(
            Conv2d(128, 64, 3, 1, padding=1),
            ReLU(),
            InstanceNorm2d(64, affine=True)
        )
        self.R2 = Sequential(
            Conv2d(96, 64, 3, 1, padding=1),
            ReLU(),
            InstanceNorm2d(64, affine=True)
        )
        self.R1 = Sequential(
            Conv2d(64, self.num_out_feat_channels, 3, 1, padding=1),
            ReLU(),
            InstanceNorm2d(self.num_out_feat_channels, affine=True)
        )

    def forward(self, x):

        # Top-down layers
        # 224x224 - 64
        b1 = self.B1(x)
        p1 = self.P1(b1)
        # 112x112 - 64
        b2 = self.B2(p1)
        p2 = self.P2(b2)
        # 56x56 - 64
        b3 = self.B3(p2)
        p3 = self.P3(b3)
        # 28x28 - 128
        b4 = self.B4(p3)
        p4 = self.P4(b4)
        # 14x14 - 128
        b5 = self.B5(p4)

        # Skip layers
        # 28x28 - 64
        s4 = self.S4(b4)
        # 56x56 - 64
        s3 = self.S3(b3)
        # 112x112 - 32
        s2 = self.S2(b2)

        # Bottom-up layers
        # 14x14 - 128
        u4 = self.U4(b5)
        # 28x28 - 128
        c4 = torch.cat((u4, s4), 1)
        # 28x28 - 192
        r4 = self.R4(c4)

        # 28x28 - 64
        u3 = self.U3(r4)
        # 56x56 - 64
        c3 = torch.cat((u3, s3), 1)
        # 56x56 - 128
        r3 = self.R3(c3)

        # 56x56 - 64
        u2 = self.U2(r3)
        # 112x112 - 64
        c2 = torch.cat((u2, s2), 1)
        # 112x112 - 96
        r2 = self.R2(c2)
        # 112x112 - 64
        u1 = self.U1(r2)
        # 224x224 - 64
        r1 = self.R1(u1)
        # 224x224 - num_out_feat_channels

        return r1
