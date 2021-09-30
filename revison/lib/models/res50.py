# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by lyh
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from .adaptive_conv import AdaptiveConv2d

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_bn=True, has_relu=True, efficient=False, groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(
            self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        x = func(x)

        return x


class PRM(nn.Module):

    def __init__(self, output_chl_num):
        super(PRM, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_prm_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                                               stride=1, padding=1, has_bn=True, has_relu=True)
        self.conv_bn_relu_prm_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True)
        self.conv_bn_relu_prm_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True)
        self.sigmoid2 = nn.Sigmoid()
        self.conv_bn_relu_prm_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True)
        self.conv_bn_relu_prm_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                                                 stride=1, padding=4, has_bn=True, has_relu=True,
                                                 groups=self.output_chl_num)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out
        out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1, 1))
        out_2 = self.conv_bn_relu_prm_2_1(out_2)
        out_2 = self.conv_bn_relu_prm_2_2(out_2)
        out_2 = self.sigmoid2(out_2)
        out_3 = self.conv_bn_relu_prm_3_1(out_1)
        out_3 = self.conv_bn_relu_prm_3_2(out_3)
        out_3 = self.sigmoid3(out_3)
        out = out_1.mul(1 + out_2.mul(out_3))
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        fc = 2 * extra.NUM_DECONV_FILTERS[-1]  # gai cheng 128 256 dou hui bao cuo zhen shi qi le guai le

        self.point_layer_1 = conv_bn_relu(extra.NUM_DECONV_FILTERS[-1], fc, kernel_size=1,
                                          stride=1, padding=0, has_bn=True, has_relu=True)
        # 基本的卷积都是卷 bn relu 如果有残差 会在relu前+残差？ 32-256
        self.point_layer_2 = nn.Conv2d(
            in_channels=fc,
            out_channels=extra.NUM_DECONV_FILTERS[-1],  # 测试
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.point_layer_bn = nn.BatchNorm2d(extra.NUM_DECONV_FILTERS[-1], momentum=BN_MOMENTUM)
        # 后面接一个relu
        self.point_layer_3 = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=16,  # 测试
            kernel_size=1,
            stride=1,
            padding=0
        )
        pn = 15
        self.part_layer_1 = conv_bn_relu(extra.NUM_DECONV_FILTERS[-1], fc, kernel_size=1,
                                         stride=1, padding=0, has_bn=True, has_relu=True)
        # 基本的卷积都是卷 bn relu 如果有残差 会在relu前+残差？
        self.part_layer_2 = nn.Conv2d(
            in_channels=fc,
            out_channels=extra.NUM_DECONV_FILTERS[-1],  # 测试
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.part_layer_bn = nn.BatchNorm2d(extra.NUM_DECONV_FILTERS[-1], momentum=BN_MOMENTUM)
        self.part_layer_3 = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=pn,  # 先测试8个part 后面添加入CFG
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 两个 统一维度的特征层不得交叉处理啊

        self.pose_refine_machine = PRM(output_chl_num=cfg['MODEL']['NUM_JOINTS'])

        self.part_param_adapter = nn.Sequential(
            # nn.Conv2d(2, 2, 3, 2),
            # nn.MaxPool2d(kernel_size=2, stride=2), # new
            nn.Conv2d(2, 2, 3, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2, 2, 3, padding=1))
        self.keypoint_param_adapter = nn.Sequential(
            # nn.Conv2d(2, 2, 3, 2),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # new
            nn.Conv2d(2, 2, 3, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2, 2, 3, padding=1))

        self.keypoint_conv1x1 = nn.Conv2d(2, 2, 1, 1)
        self.part_conv1x1 = nn.Conv2d(2, 2, 1, 1)

        self.keypoint_refine_layer = conv_bn_relu(2, 2, kernel_size=1,
                                                  stride=1, padding=0, has_bn=True, has_relu=True)
        self.part_refine_layer = conv_bn_relu(2, 2, kernel_size=1,
                                              stride=1, padding=0, has_bn=True, has_relu=True)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        f = x
        x = self.final_layer(x)

        # 特征图 64 64 256
        # 假定该特征包含了全部信息 也可以搞一个多监督的信息 但是得把前面特征联合起来  感觉实现起来困难
        # 加入动态卷积 自动获得权重？

        # keypoint
        fk = self.point_layer_1(f)
        fk = self.point_layer_2(fk)
        fk = self.point_layer_bn(fk)

        fp = self.part_layer_1(f)
        fp = self.part_layer_2(fp)
        fp = self.part_layer_bn(fp)

        k = self.relu(f + fk)
        # k = self.relu(f + fk + fp)
        p = self.relu(f + fp)
        # p = self.relu(f + fk + fp)
        fk = self.point_layer_3(k)
        fp = self.part_layer_3(p)

        fk1, fk2, fk3, fk4, fk5, fk6, fk7, fk8, fk9, fk10, fk11, fk12, fk13, fk14, fk15, fk16 = fk.split(1, 1)
        fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13, fp14, fp15 = fp.split(1, 1)

        # limbs_links = [[0, 1], [1, 2], [2, 6], [3, 6], [4, 3], [5, 4], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12],
        #                        [12, 8], [13, 8], [14, 13], [15, 14]]

        batchsize = x.size(0)
        adaptive_conv = AdaptiveConv2d(batchsize * 2,
                                       batchsize * 2,
                                       7, padding=3,
                                       groups=batchsize * 2,
                                       bias=False)

        kk1 = torch.cat((fk1, fk2), 1)
        ff1 = torch.cat((fp1, fp1), 1)
        kk1_res = self.keypoint_conv1x1(kk1)
        # meiyou zu jian  xin xi bu yong 1*1 juanjile ?
        kp1_theta = self.part_param_adapter(ff1)

        kk1_res = adaptive_conv(kk1_res, kp1_theta)
        kk1_res = self.keypoint_refine_layer(kk1_res)
        fkr_1 = kk1_res + kk1
        fk1_1, fk2_1 = fkr_1.split(1, 1)

        fp1_res = self.part_conv1x1(ff1)
        fp1_phi = self.keypoint_param_adapter(kk1)
        fp1_res = adaptive_conv(fp1_res, fp1_phi)
        fp1_res = self.part_refine_layer(fp1_res)
        fp1 = fp1_res + fp1
        fp1_1, fp1_2 = fp1.split(1, 1)
        fp1 = fp1_1 + fp1_2

        kk2 = torch.cat((fk2, fk3), 1)
        ff2 = torch.cat((fp2, fp2), 1)
        kk2_res = self.keypoint_conv1x1(kk2)
        # meiyou zu jian  xin xi bu yong 1*1 juanjile ?
        kp2_theta = self.part_param_adapter(ff2)
        kk2_res = adaptive_conv(kk2_res, kp2_theta)
        kk2_res = self.keypoint_refine_layer(kk2_res)
        fkr_2 = kk2_res + kk2
        fk2_2, fk3_1 = fkr_2.split(1, 1)

        fp2_res = self.part_conv1x1(ff2)
        fp2_phi = self.keypoint_param_adapter(kk2)
        fp2_res = adaptive_conv(fp2_res, fp2_phi)
        fp2_res = self.part_refine_layer(fp2_res)
        fp2 = fp2_res + fp2
        fp2_1, fp2_2 = fp2.split(1, 1)
        fp2 = fp2_1 + fp2_2

        kk3 = torch.cat((fk3, fk7), 1)
        ff3 = torch.cat((fp3, fp3), 1)
        kk3_res = self.keypoint_conv1x1(kk3)
        kp3_theta = self.part_param_adapter(ff3)
        kk3_res = adaptive_conv(kk3_res, kp3_theta)
        kk3_res = self.keypoint_refine_layer(kk3_res)
        fkr_3 = kk3_res + kk3
        fk3_2, fk7_1 = fkr_3.split(1, 1)

        fp3_res = self.part_conv1x1(ff3)
        fp3_phi = self.keypoint_param_adapter(kk3)
        fp3_res = adaptive_conv(fp3_res, fp3_phi)
        fp3_res = self.part_refine_layer(fp3_res)
        fp3 = fp3_res + fp3
        fp3_1, fp3_2 = fp3.split(1, 1)
        fp3 = fp3_1 + fp3_2

        kk4 = torch.cat((fk4, fk7), 1)
        ff4 = torch.cat((fp4, fp4), 1)
        kk4_res = self.keypoint_conv1x1(kk4)
        kp4_theta = self.part_param_adapter(ff4)
        kk4_res = adaptive_conv(kk4_res, kp4_theta)
        kk4_res = self.keypoint_refine_layer(kk4_res)
        fkr_4 = kk4_res + kk4
        fk4_1, fk7_2 = fkr_4.split(1, 1)

        fp4_res = self.part_conv1x1(ff4)
        fp4_phi = self.keypoint_param_adapter(kk4)
        fp4_res = adaptive_conv(fp4_res, fp4_phi)
        fp4_res = self.part_refine_layer(fp4_res)
        fp4 = fp4_res + fp4
        fp4_1, fp4_2 = fp4.split(1, 1)
        fp4 = fp4_1 + fp4_2

        kk5 = torch.cat((fk5, fk4), 1)
        ff5 = torch.cat((fp5, fp5), 1)
        kk5_res = self.keypoint_conv1x1(kk5)
        kp5_theta = self.part_param_adapter(ff5)
        kk5_res = adaptive_conv(kk5_res, kp5_theta)
        kk5_res = self.keypoint_refine_layer(kk5_res)
        fkr_5 = kk5_res + kk5
        fk5_1, fk4_2 = fkr_5.split(1, 1)

        fp5_res = self.part_conv1x1(ff5)
        fp5_phi = self.keypoint_param_adapter(kk5)
        fp5_res = adaptive_conv(fp5_res, fp5_phi)
        fp5_res = self.part_refine_layer(fp5_res)
        fp5 = fp5_res + fp5
        fp5_1, fp5_2 = fp5.split(1, 1)
        fp5 = fp5_1 + fp5_2

        kk6 = torch.cat((fk6, fk5), 1)
        ff6 = torch.cat((fp6, fp6), 1)
        kk6_res = self.keypoint_conv1x1(kk6)
        kp6_theta = self.part_param_adapter(ff6)
        kk6_res = adaptive_conv(kk6_res, kp6_theta)
        kk6_res = self.keypoint_refine_layer(kk6_res)
        fkr_6 = kk6_res + kk6
        fk6_1, fk5_2 = fkr_6.split(1, 1)

        fp6_res = self.part_conv1x1(ff6)
        fp6_phi = self.keypoint_param_adapter(kk6)
        fp6_res = adaptive_conv(fp6_res, fp6_phi)
        fp6_res = self.part_refine_layer(fp6_res)
        fp6 = fp6_res + fp6
        fp6_1, fp6_2 = fp6.split(1, 1)
        fp6 = fp6_1 + fp6_2

        kk7 = torch.cat((fk7, fk8), 1)
        ff7 = torch.cat((fp7, fp7), 1)
        kk7_res = self.keypoint_conv1x1(kk7)
        kp7_theta = self.part_param_adapter(ff7)
        kk7_res = adaptive_conv(kk7_res, kp7_theta)
        kk7_res = self.keypoint_refine_layer(kk7_res)
        fkr_7 = kk7_res + kk7
        fk7_3, fk8_1 = fkr_7.split(1, 1)

        fp7_res = self.part_conv1x1(ff7)
        fp7_phi = self.keypoint_param_adapter(kk7)
        fp7_res = adaptive_conv(fp7_res, fp7_phi)
        fp7_res = self.part_refine_layer(fp7_res)
        fp7 = fp7_res + fp7
        fp7_1, fp7_2 = fp7.split(1, 1)
        fp7 = fp7_1 + fp7_2

        kk8 = torch.cat((fk8, fk9), 1)
        ff8 = torch.cat((fp8, fp8), 1)
        kk8_res = self.keypoint_conv1x1(kk8)
        kp8_theta = self.part_param_adapter(ff8)
        kk8_res = adaptive_conv(kk8_res, kp8_theta)
        kk8_res = self.keypoint_refine_layer(kk8_res)
        fkr_8 = kk8_res + kk8
        fk8_2, fk9_1 = fkr_8.split(1, 1)

        fp8_res = self.part_conv1x1(ff8)
        fp8_phi = self.keypoint_param_adapter(kk8)
        fp8_res = adaptive_conv(fp8_res, fp8_phi)
        fp8_res = self.part_refine_layer(fp8_res)
        fp8 = fp8_res + fp8
        fp8_1, fp8_2 = fp8.split(1, 1)
        fp8 = fp8_1 + fp8_2

        kk9 = torch.cat((fk9, fk10), 1)
        ff9 = torch.cat((fp9, fp9), 1)
        kk9_res = self.keypoint_conv1x1(kk9)
        kp9_theta = self.part_param_adapter(ff9)
        kk9_res = adaptive_conv(kk9_res, kp9_theta)
        kk9_res = self.keypoint_refine_layer(kk9_res)
        fkr_9 = kk9_res + kk9
        fk9_2, fk10_1 = fkr_9.split(1, 1)

        fp9_res = self.part_conv1x1(ff9)
        fp9_phi = self.keypoint_param_adapter(kk9)
        fp9_res = adaptive_conv(fp9_res, fp9_phi)
        fp9_res = self.part_refine_layer(fp9_res)
        fp9 = fp9_res + fp9
        fp9_1, fp9_2 = fp9.split(1, 1)
        fp9 = fp9_1 + fp9_2

        kk10 = torch.cat((fk11, fk12), 1)
        ff10 = torch.cat((fp10, fp10), 1)
        kk10_res = self.keypoint_conv1x1(kk10)
        kp10_theta = self.part_param_adapter(ff10)
        kk10_res = adaptive_conv(kk10_res, kp10_theta)
        kk10_res = self.keypoint_refine_layer(kk10_res)
        fkr_10 = kk10_res + kk10
        fk11_1, fk12_1 = fkr_10.split(1, 1)

        fp10_res = self.part_conv1x1(ff10)
        fp10_phi = self.keypoint_param_adapter(kk10)
        fp10_res = adaptive_conv(fp10_res, fp10_phi)
        fp10_res = self.part_refine_layer(fp10_res)
        fp10 = fp10_res + fp10
        fp10_1, fp10_2 = fp10.split(1, 1)
        fp10 = fp10_1 + fp10_2

        kk11 = torch.cat((fk12, fk13), 1)
        ff11 = torch.cat((fp11, fp11), 1)
        kk11_res = self.keypoint_conv1x1(kk11)
        kp11_theta = self.part_param_adapter(ff11)
        kk11_res = adaptive_conv(kk11_res, kp11_theta)
        kk11_res = self.keypoint_refine_layer(kk11_res)
        fkr_11 = kk11_res + kk11
        fk12_2, fk13_1 = fkr_11.split(1, 1)

        fp11_res = self.part_conv1x1(ff11)
        fp11_phi = self.keypoint_param_adapter(kk11)
        fp11_res = adaptive_conv(fp11_res, fp11_phi)
        fp11_res = self.part_refine_layer(fp11_res)
        fp11 = fp11_res + fp11
        fp11_1, fp11_2 = fp11.split(1, 1)
        fp11 = fp11_1 + fp11_2

        kk12 = torch.cat((fk13, fk9), 1)
        ff12 = torch.cat((fp12, fp12), 1)
        kk12_res = self.keypoint_conv1x1(kk12)
        kp12_theta = self.part_param_adapter(ff12)
        kk12_res = adaptive_conv(kk12_res, kp12_theta)
        kk12_res = self.keypoint_refine_layer(kk12_res)
        fkr_12 = kk12_res + kk12
        fk13_2, fk9_3 = fkr_12.split(1, 1)

        fp12_res = self.part_conv1x1(ff12)
        fp12_phi = self.keypoint_param_adapter(kk12)
        fp12_res = adaptive_conv(fp12_res, fp12_phi)
        fp12_res = self.part_refine_layer(fp12_res)
        fp12 = fp12_res + fp12
        fp12_1, fp12_2 = fp12.split(1, 1)
        fp12 = fp12_1 + fp12_2

        kk13 = torch.cat((fk14, fk9), 1)
        ff13 = torch.cat((fp13, fp13), 1)
        kk13_res = self.keypoint_conv1x1(kk13)
        kp13_theta = self.part_param_adapter(ff13)
        kk13_res = adaptive_conv(kk13_res, kp13_theta)
        kk13_res = self.keypoint_refine_layer(kk13_res)
        fkr_13 = kk13_res + kk13
        fk14_1, fk9_4 = fkr_13.split(1, 1)

        fp13_res = self.part_conv1x1(ff13)
        fp13_phi = self.keypoint_param_adapter(kk13)
        fp13_res = adaptive_conv(fp13_res, fp13_phi)
        fp13_res = self.part_refine_layer(fp13_res)
        fp13 = fp13_res + fp13
        fp13_1, fp13_2 = fp13.split(1, 1)
        fp13 = fp13_1 + fp13_2

        kk14 = torch.cat((fk15, fk14), 1)
        ff14 = torch.cat((fp14, fp14), 1)
        kk14_res = self.keypoint_conv1x1(kk14)
        kp14_theta = self.part_param_adapter(ff14)
        kk14_res = adaptive_conv(kk14_res, kp14_theta)
        kk14_res = self.keypoint_refine_layer(kk14_res)
        fkr_14 = kk14_res + kk14
        fk15_1, fk14_2 = fkr_14.split(1, 1)

        fp14_res = self.part_conv1x1(ff14)
        fp14_phi = self.keypoint_param_adapter(kk14)
        fp14_res = adaptive_conv(fp14_res, fp14_phi)
        fp14_res = self.part_refine_layer(fp14_res)
        fp14 = fp14_res + fp14
        fp14_1, fp14_2 = fp14.split(1, 1)
        fp14 = fp14_1 + fp14_2

        kk15 = torch.cat((fk16, fk15), 1)
        ff15 = torch.cat((fp15, fp15), 1)
        kk15_res = self.keypoint_conv1x1(kk15)
        kp15_theta = self.part_param_adapter(ff15)
        kk15_res = adaptive_conv(kk15_res, kp15_theta)
        kk15_res = self.keypoint_refine_layer(kk15_res)
        fkr_15 = kk15_res + kk15
        fk16_1, fk15_2 = fkr_15.split(1, 1)

        fp15_res = self.part_conv1x1(ff15)
        fp15_phi = self.keypoint_param_adapter(kk15)
        fp15_res = adaptive_conv(fp15_res, fp15_phi)
        fp15_res = self.part_refine_layer(fp15_res)
        fp15 = fp15_res + fp15
        fp15_1, fp15_2 = fp15.split(1, 1)
        fp15 = fp15_1 + fp15_2

        fk1 = fk1_1
        fk2 = fk2_1 + fk2_2
        fk3 = fk3_1 + fk3_2
        fk4 = fk4_1 + fk4_2
        fk5 = fk5_1 + fk5_2
        fk6 = fk6_1
        fk7 = fk7_1 + fk7_2 + fk7_3
        fk8 = fk8_1 + fk8_2
        fk9 = fk9_1 + fk9_2 + fk9_3 + fk9_4
        fk10 = fk10_1
        fk11 = fk11_1
        fk12 = fk12_1 + fk12_2
        fk13 = fk13_1 + fk13_2
        fk14 = fk14_1 + fk14_2
        fk15 = fk15_1 + fk15_2
        fk16 = fk16_1

        fk = torch.cat((fk1, fk2, fk3, fk4, fk5, fk6, fk7, fk8, fk9, fk10, fk11, fk12, fk13, fk14, fk15, fk16), 1)
        # fp = torch.cat((fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13, fp14, fp15), 1)

        # x = self.final_layer(y_list[0]) #关键点的检测

        fk = self.pose_refine_machine(x)

        return fk, fp
        # return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
