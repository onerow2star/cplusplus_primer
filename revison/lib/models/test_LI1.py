# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by lyh
# ------------------------------------------------------------------------------

# 关键点分组 + 信息传递 + 后处理


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

class SpatialAttention(nn.Module):
    def __init__(self, output_chl_num):
        super(SpatialAttention, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_sa_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                                               stride=1, padding=1, has_bn=True, has_relu=True)
        self.conv_bn_relu_sa_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True)
        self.conv_bn_relu_sa_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                                                 stride=1, padding=4, has_bn=True, has_relu=True,
                                                 groups=self.output_chl_num)
        self.sigmoid2 = nn.Sigmoid()
    def forward(self, x):
        x = self.conv_bn_relu_sa_1(x)
        out = self.conv_bn_relu_sa_2_1(x)
        out = self.conv_bn_relu_sa_2_2(out)
        out = self.sigmoid2(out)
        x = x.mul(1 + out)
        return x

class ChannelAttention(nn.Module):

    def __init__(self, output_chl_num):
        super(ChannelAttention, self).__init__()
        self.output_chl_num = output_chl_num
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_bn_relu_ca_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                                               stride=1, padding=1, has_bn=True, has_relu=True)
        self.conv_bn_relu_ca_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True)
        self.sigmoid2 = nn.Sigmoid()
    def forward(self, x):
        x = self.conv_bn_relu_ca_1(x)
        x1 = self.avg_pool(x)
        # x2 = self.max_pool(x)
        x1 = self.conv_bn_relu_ca_2_1(x1)
        # x1 = self.conv_bn_relu_ca_2_2(x1)
        # x2 = self.conv_bn_relu_ca_2_1(x2)
        # x2 = self.conv_bn_relu_ca_2_2(x2)
        out = self.sigmoid2(x1)
        x = x.mul(1 + out)
        # x = x.mul(out)
        return x

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


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        fc = 2 * pre_stage_channels[0]  # gai cheng 128 256 dou hui bao cuo zhen shi qi le guai le

        self.point_layer_1 = conv_bn_relu(pre_stage_channels[0], fc, kernel_size=1,
                                          stride=1, padding=0, has_bn=True, has_relu=True)
        # 基本的卷积都是卷 bn relu 如果有残差 会在relu前+残差？ 32-256
        self.point_layer_2 = nn.Conv2d(
            in_channels=fc,
            out_channels=pre_stage_channels[0],  # 测试
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.point_layer_bn = nn.BatchNorm2d(pre_stage_channels[0], momentum=BN_MOMENTUM)
        # 后面接一个relu
        self.point_layer_3 = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=16,  # 测试
            kernel_size=1,
            stride=1,
            padding=0
        )
        pn = 15
        self.part_layer_1 = conv_bn_relu(pre_stage_channels[0], fc, kernel_size=1,
                                         stride=1, padding=0, has_bn=True, has_relu=True)
        # 基本的卷积都是卷 bn relu 如果有残差 会在relu前+残差？
        self.part_layer_2 = nn.Conv2d(
            in_channels=fc,
            out_channels=pre_stage_channels[0],  # 测试
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.part_layer_bn = nn.BatchNorm2d(pre_stage_channels[0], momentum=BN_MOMENTUM)
        self.part_layer_3 = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=pn,  # 先测试8个part 后面添加入CFG
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 两个 统一维度的特征层不得交叉处理啊

        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        # self.pose_refine_machine = PRM(output_chl_num=cfg['MODEL']['NUM_JOINTS'])



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

        self.sa = SpatialAttention(output_chl_num=cfg['MODEL']['NUM_JOINTS'])
        self.sap = SpatialAttention(output_chl_num=15)
        self.ca = ChannelAttention(output_chl_num=2)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        f = y_list[0]  # 特征图 64 64 32
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

        #####################################

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
        kk1_res = self.keypoint_conv1x1(kk1_res)
        kk1_res = self.ca(kk1_res)
        fkr_1 = kk1_res + kk1
        fk1_1, fk2_1 = fkr_1.split(1, 1)

        fp1_res = self.part_conv1x1(ff1)
        fp1_phi = self.keypoint_param_adapter(kk1)
        fp1_res = adaptive_conv(fp1_res, fp1_phi)
        fp1_res = self.ca(fp1_res)
        fp1 = fp1_res + fp1
        fp1_1, fp1_2 = fp1.split(1, 1)
        fp1 = fp1_1 + fp1_2

        kk2 = torch.cat((fk2, fk3), 1)
        ff2 = torch.cat((fp2, fp2), 1)
        kk2_res = self.keypoint_conv1x1(kk2)
        # meiyou zu jian  xin xi bu yong 1*1 juanjile ?
        kp2_theta = self.part_param_adapter(ff2)
        kk2_res = adaptive_conv(kk2_res, kp2_theta)
        kk2_res = self.keypoint_conv1x1(kk2_res)
        kk2_res = self.ca(kk2_res)
        fkr_2 = kk2_res + kk2
        fk2_2, fk3_1 = fkr_2.split(1, 1)

        fp2_res = self.part_conv1x1(ff2)
        fp2_phi = self.keypoint_param_adapter(kk2)
        fp2_res = adaptive_conv(fp2_res, fp2_phi)
        fp2_res = self.ca(fp2_res)
        fp2 = fp2_res + fp2
        fp2_1, fp2_2 = fp2.split(1, 1)
        fp2 = fp2_1 + fp2_2

        kk3 = torch.cat((fk3, fk7), 1)
        ff3 = torch.cat((fp3, fp3), 1)
        kk3_res = self.keypoint_conv1x1(kk3)
        kp3_theta = self.part_param_adapter(ff3)
        kk3_res = adaptive_conv(kk3_res, kp3_theta)
        kk3_res = self.keypoint_conv1x1(kk3_res)
        kk3_res = self.ca(kk3_res)
        fkr_3 = kk3_res + kk3
        fk3_2, fk7_1 = fkr_3.split(1, 1)

        fp3_res = self.part_conv1x1(ff3)
        fp3_phi = self.keypoint_param_adapter(kk3)
        fp3_res = adaptive_conv(fp3_res, fp3_phi)
        fp3_res = self.ca(fp3_res)
        fp3 = fp3_res + fp3
        fp3_1, fp3_2 = fp3.split(1, 1)
        fp3 = fp3_1 + fp3_2

        kk4 = torch.cat((fk4, fk7), 1)
        ff4 = torch.cat((fp4, fp4), 1)
        kk4_res = self.keypoint_conv1x1(kk4)
        kp4_theta = self.part_param_adapter(ff4)
        kk4_res = adaptive_conv(kk4_res, kp4_theta)
        kk4_res = self.keypoint_conv1x1(kk4_res)
        kk4_res = self.ca(kk4_res)
        fkr_4 = kk4_res + kk4
        fk4_1, fk7_2 = fkr_4.split(1, 1)

        fp4_res = self.part_conv1x1(ff4)
        fp4_phi = self.keypoint_param_adapter(kk4)
        fp4_res = adaptive_conv(fp4_res, fp4_phi)
        fp4_res = self.ca(fp4_res)
        fp4 = fp4_res + fp4
        fp4_1, fp4_2 = fp4.split(1, 1)
        fp4 = fp4_1 + fp4_2

        kk5 = torch.cat((fk5, fk4), 1)
        ff5 = torch.cat((fp5, fp5), 1)
        kk5_res = self.keypoint_conv1x1(kk5)
        kp5_theta = self.part_param_adapter(ff5)
        kk5_res = adaptive_conv(kk5_res, kp5_theta)
        kk5_res = self.keypoint_conv1x1(kk5_res)
        kk5_res = self.ca(kk5_res)
        fkr_5 = kk5_res + kk5
        fk5_1, fk4_2 = fkr_5.split(1, 1)

        fp5_res = self.part_conv1x1(ff5)
        fp5_phi = self.keypoint_param_adapter(kk5)
        fp5_res = adaptive_conv(fp5_res, fp5_phi)
        fp5_res = self.ca(fp5_res)
        fp5 = fp5_res + fp5
        fp5_1, fp5_2 = fp5.split(1, 1)
        fp5 = fp5_1 + fp5_2

        kk6 = torch.cat((fk6, fk5), 1)
        ff6 = torch.cat((fp6, fp6), 1)
        kk6_res = self.keypoint_conv1x1(kk6)
        kp6_theta = self.part_param_adapter(ff6)
        kk6_res = adaptive_conv(kk6_res, kp6_theta)
        kk6_res = self.keypoint_conv1x1(kk6_res)
        kk6_res = self.ca(kk6_res)
        fkr_6 = kk6_res + kk6
        fk6_1, fk5_2 = fkr_6.split(1, 1)

        fp6_res = self.part_conv1x1(ff6)
        fp6_phi = self.keypoint_param_adapter(kk6)
        fp6_res = adaptive_conv(fp6_res, fp6_phi)
        fp6_res = self.ca(fp6_res)
        fp6 = fp6_res + fp6
        fp6_1, fp6_2 = fp6.split(1, 1)
        fp6 = fp6_1 + fp6_2

        kk7 = torch.cat((fk7, fk8), 1)
        ff7 = torch.cat((fp7, fp7), 1)
        kk7_res = self.keypoint_conv1x1(kk7)
        kp7_theta = self.part_param_adapter(ff7)
        kk7_res = adaptive_conv(kk7_res, kp7_theta)
        kk7_res = self.keypoint_conv1x1(kk7_res)
        kk7_res = self.ca(kk7_res)
        fkr_7 = kk7_res + kk7
        fk7_3, fk8_1 = fkr_7.split(1, 1)

        fp7_res = self.part_conv1x1(ff7)
        fp7_phi = self.keypoint_param_adapter(kk7)
        fp7_res = adaptive_conv(fp7_res, fp7_phi)
        fp7_res = self.ca(fp7_res)
        fp7 = fp7_res + fp7
        fp7_1, fp7_2 = fp7.split(1, 1)
        fp7 = fp7_1 + fp7_2

        kk8 = torch.cat((fk8, fk9), 1)
        ff8 = torch.cat((fp8, fp8), 1)
        kk8_res = self.keypoint_conv1x1(kk8)
        kp8_theta = self.part_param_adapter(ff8)
        kk8_res = adaptive_conv(kk8_res, kp8_theta)
        kk8_res = self.keypoint_conv1x1(kk8_res)
        kk8_res = self.ca(kk8_res)
        fkr_8 = kk8_res + kk8
        fk8_2, fk9_1 = fkr_8.split(1, 1)

        fp8_res = self.part_conv1x1(ff8)
        fp8_phi = self.keypoint_param_adapter(kk8)
        fp8_res = adaptive_conv(fp8_res, fp8_phi)
        fp8_res = self.ca(fp8_res)
        fp8 = fp8_res + fp8
        fp8_1, fp8_2 = fp8.split(1, 1)
        fp8 = fp8_1 + fp8_2

        kk9 = torch.cat((fk9, fk10), 1)
        ff9 = torch.cat((fp9, fp9), 1)
        kk9_res = self.keypoint_conv1x1(kk9)
        kp9_theta = self.part_param_adapter(ff9)
        kk9_res = adaptive_conv(kk9_res, kp9_theta)
        kk9_res = self.keypoint_conv1x1(kk9_res)
        kk9_res = self.ca(kk9_res)
        fkr_9 = kk9_res + kk9
        fk9_2, fk10_1 = fkr_9.split(1, 1)

        fp9_res = self.part_conv1x1(ff9)
        fp9_phi = self.keypoint_param_adapter(kk9)
        fp9_res = adaptive_conv(fp9_res, fp9_phi)
        fp9_res = self.ca(fp9_res)
        fp9 = fp9_res + fp9
        fp9_1, fp9_2 = fp9.split(1, 1)
        fp9 = fp9_1 + fp9_2

        kk10 = torch.cat((fk11, fk12), 1)
        ff10 = torch.cat((fp10, fp10), 1)
        kk10_res = self.keypoint_conv1x1(kk10)
        kp10_theta = self.part_param_adapter(ff10)
        kk10_res = adaptive_conv(kk10_res, kp10_theta)
        kk10_res = self.keypoint_conv1x1(kk10_res)
        kk10_res = self.ca(kk10_res)
        fkr_10 = kk10_res + kk10
        fk11_1, fk12_1 = fkr_10.split(1, 1)

        fp10_res = self.part_conv1x1(ff10)
        fp10_phi = self.keypoint_param_adapter(kk10)
        fp10_res = adaptive_conv(fp10_res, fp10_phi)
        fp10_res = self.ca(fp10_res)
        fp10 = fp10_res + fp10
        fp10_1, fp10_2 = fp10.split(1, 1)
        fp10 = fp10_1 + fp10_2

        kk11 = torch.cat((fk12, fk13), 1)
        ff11 = torch.cat((fp11, fp11), 1)
        kk11_res = self.keypoint_conv1x1(kk11)
        kp11_theta = self.part_param_adapter(ff11)
        kk11_res = adaptive_conv(kk11_res, kp11_theta)
        kk11_res = self.keypoint_conv1x1(kk11_res)
        kk11_res = self.ca(kk11_res)
        fkr_11 = kk11_res + kk11
        fk12_2, fk13_1 = fkr_11.split(1, 1)

        fp11_res = self.part_conv1x1(ff11)
        fp11_phi = self.keypoint_param_adapter(kk11)
        fp11_res = adaptive_conv(fp11_res, fp11_phi)
        fp11_res = self.ca(fp11_res)
        fp11 = fp11_res + fp11
        fp11_1, fp11_2 = fp11.split(1, 1)
        fp11 = fp11_1 + fp11_2

        kk12 = torch.cat((fk13, fk9), 1)
        ff12 = torch.cat((fp12, fp12), 1)
        kk12_res = self.keypoint_conv1x1(kk12)
        kp12_theta = self.part_param_adapter(ff12)
        kk12_res = adaptive_conv(kk12_res, kp12_theta)
        kk12_res = self.keypoint_conv1x1(kk12_res)
        kk12_res = self.ca(kk12_res)
        fkr_12 = kk12_res + kk12
        fk13_2, fk9_3 = fkr_12.split(1, 1)

        fp12_res = self.part_conv1x1(ff12)
        fp12_phi = self.keypoint_param_adapter(kk12)
        fp12_res = adaptive_conv(fp12_res, fp12_phi)
        fp12_res = self.ca(fp12_res)
        fp12 = fp12_res + fp12
        fp12_1, fp12_2 = fp12.split(1, 1)
        fp12 = fp12_1 + fp12_2

        kk13 = torch.cat((fk14, fk9), 1)
        ff13 = torch.cat((fp13, fp13), 1)
        kk13_res = self.keypoint_conv1x1(kk13)
        kp13_theta = self.part_param_adapter(ff13)
        kk13_res = adaptive_conv(kk13_res, kp13_theta)
        kk13_res = self.keypoint_conv1x1(kk13_res)
        kk13_res = self.ca(kk13_res)
        fkr_13 = kk13_res + kk13
        fk14_1, fk9_4 = fkr_13.split(1, 1)

        fp13_res = self.part_conv1x1(ff13)
        fp13_phi = self.keypoint_param_adapter(kk13)
        fp13_res = adaptive_conv(fp13_res, fp13_phi)

        fp13_res = self.ca(fp13_res)
        fp13 = fp13_res + fp13
        fp13_1, fp13_2 = fp13.split(1, 1)
        fp13 = fp13_1 + fp13_2

        kk14 = torch.cat((fk15, fk14), 1)
        ff14 = torch.cat((fp14, fp14), 1)
        kk14_res = self.keypoint_conv1x1(kk14)
        kp14_theta = self.part_param_adapter(ff14)
        kk14_res = adaptive_conv(kk14_res, kp14_theta)
        kk14_res = self.keypoint_conv1x1(kk14_res)
        kk14_res = self.ca(kk14_res)
        fkr_14 = kk14_res + kk14
        fk15_1, fk14_2 = fkr_14.split(1, 1)

        fp14_res = self.part_conv1x1(ff14)
        fp14_phi = self.keypoint_param_adapter(kk14)
        fp14_res = adaptive_conv(fp14_res, fp14_phi)
        fp14_res = self.ca(fp14_res)
        fp14 = fp14_res + fp14
        fp14_1, fp14_2 = fp14.split(1, 1)
        fp14 = fp14_1 + fp14_2

        kk15 = torch.cat((fk16, fk15), 1)
        ff15 = torch.cat((fp15, fp15), 1)
        kk15_res = self.keypoint_conv1x1(kk15)
        kp15_theta = self.part_param_adapter(ff15)
        kk15_res = adaptive_conv(kk15_res, kp15_theta)
        kk15_res = self.keypoint_conv1x1(kk15_res)
        kk15_res = self.ca(kk15_res)
        fkr_15 = kk15_res + kk15
        fk16_1, fk15_2 = fkr_15.split(1, 1)

        fp15_res = self.part_conv1x1(ff15)
        fp15_phi = self.keypoint_param_adapter(kk15)
        fp15_res = adaptive_conv(fp15_res, fp15_phi)
        fp15_res = self.ca(fp15_res)
        fp15 = fp15_res + fp15
        fp15_1, fp15_2 = fp15.split(1, 1)
        fp15 = fp15_1 + fp15_2

        fk1 = fk1_1 # ra
        fk2 = fk2_1 + fk2_2# rk
        fk3 = fk3_1 + fk3_2 # rh
        fk4 = fk4_1 + fk4_2
        fk5 = fk5_1 + fk5_2
        fk6 = fk6_1
        fk7 = (fk7_1 + fk7_2 + fk7_3)
        fk8 = fk8_1 + fk8_2
        fk9 = (fk9_1 + fk9_2 + fk9_3 + fk9_4)
        fk10 = fk10_1
        fk11 = fk11_1
        fk12 = (fk12_1 + fk12_2)
        fk13 = fk13_1 + fk13_2
        fk14 = fk14_1 + fk14_2
        fk15 = (fk15_1 + fk15_2)
        fk16 = fk16_1

        fk = torch.cat((fk1, fk2, fk3, fk4, fk5, fk6, fk7, fk8, fk9, fk10, fk11, fk12, fk13, fk14, fk15, fk16), 1)
        # fp = torch.cat((fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13, fp14, fp15), 1)

        # x = self.final_layer(y_list[0]) #关键点的检测

        # fk = self.pose_refine_machine(fk)
        fk = self.sa(fk)
        # fp = self.sap(fp) # zhi jie wei 0
        ##############################################

        return fk, fp

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
