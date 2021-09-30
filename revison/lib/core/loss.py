# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by lyh
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

# torch.set_printoptions(profile="full")

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),

                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsFocalLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsFocalLoss, self).__init__()
        # self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)  # 17个关键点组成的元组 (b，1，x)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)  # 17个关键点组成的元组
        loss = 0
        alpha = 0.1
        beta = 0.02
        gamma = 1

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()  # 删除维度为1的 (b，x)
            heatmap_gt = heatmaps_gt[idx].squeeze()  # 删除维度为1的
            # print(target_weight.shape)
            # print(heatmap_gt.shape)
            eps = 1e-8  # 1e-12
            # heatmap_pred = torch.clamp(heatmap_pred, eps, 1. - eps)  # improve the stability of the focal loss
            # -1.3387e-04,
            st = torch.where(torch.ge(heatmap_gt, 0.01), heatmap_pred - alpha, 1 - heatmap_pred - beta)
            factor = torch.abs(1. - st)** gamma # gamma为1 (1. - st) ** gamma  for gamma=2
            # print(factor.shape)
            if self.use_target_weight:  # true 每个点不同的权重值
                # target_weight(1: visible, 0: invisible) 并不是我想象的数值权重
                out = (heatmap_pred - heatmap_gt) ** 2 * factor * target_weight[:, idx]
                loss += out.mean()
                # loss += 0.5 * self.criterion(
                #     heatmap_pred.mul(target_weight[:, idx]),
                #     heatmap_gt.mul(target_weight[:, idx])
                # )  # 除以2是为了抵消平方的微分
            else:
                out = (heatmap_pred - heatmap_gt) ** 2 * factor
                loss += 0.5 * out.mean()
        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
