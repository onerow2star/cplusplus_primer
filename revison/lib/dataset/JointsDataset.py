# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by lyh
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''



        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)



        joints = db_rec['joints_3d']
        gtjoints = copy.deepcopy(db_rec['joints_3d']) # tensor need edit


        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_part, target_weight, target_part_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'gtjoints': gtjoints,
        }

        return input, target, target_part, target_weight, target_part_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected


    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''


        # limbs_links = [[0, 1], [1, 2], [5, 4], [4, 3], [10, 11], [11, 12], [15, 14], [14, 13]]

        limbs_links = [[0, 1], [1, 2], [2, 6], [3, 6], [4, 3], [5, 4], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12],
                       [12, 8], [13, 8], [14, 13], [15, 14]]

        # 16
        # 15


        coco_limbs_links = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        # 17
        # 19

        limbs = np.array(limbs_links)
        # limbs = np.array(coco_limbs_links)-1
        pn= 15
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        target_part_weight = np.ones((pn, 1), dtype=np.float32)
        target_type = 'gaussian'
        assert target_type == 'gaussian', \
            'Only support gaussian map now!'
        if target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            target_part = np.zeros((pn,
                                    self.heatmap_size[1],
                                    self.heatmap_size[0]),
                                   dtype=np.float32)
            sigma = 2
            tmp_size = sigma * 3
            num_joints = self.num_joints
            for index, lb in enumerate(limbs):
                if np.all(target_weight[lb] > 0):
                    # Test whether all array elements along a given axis evaluate to True
                    # 动态的创建一个场 首先 我们要找到场的偏移向量
                    target_part_weight[index, 0] = (target_weight[lb[0]] + target_weight[lb[1]])/2

                    feat_stride = 4  # image_size / heatmap_size
                    mu_x_1 = int(joints[lb[0]][0] / feat_stride + 0.5)
                    mu_y_1 = int(joints[lb[0]][1] / feat_stride + 0.5)
                    mu_x_2 = int(joints[lb[1]][0] / feat_stride + 0.5)
                    mu_y_2 = int(joints[lb[1]][1] / feat_stride + 0.5)

                    joint_1 = np.array([mu_x_1, mu_y_1])
                    joint_2 = np.array([mu_x_2, mu_y_2])

                    offset = joint_2[:2] - joint_1[:2]  # 热图的位置
                    offset_d = np.linalg.norm(offset)  # 求范数 整体平方和开根号

                    num = max(2, int(np.ceil(offset_d)))  # 向上取整

                    # dynamically create s
                    si = max(3, int(offset_d * 0))
                    # s 奇数 有0 整数 偶数 0.5 无0 间隔为一
                    # meshgrid: coordinate matrix 返回两个元素的列表 都是矩阵 x*y的矩阵 第一个元素是横坐标取值 第二个同理
                    xyv = np.stack(np.meshgrid(
                        np.linspace(-0.5 * (si - 1), 0.5 * (si - 1), si),
                        np.linspace(-0.5 * (si - 1), 0.5 * (si - 1), si),
                    ), axis=-1).reshape(-1, 2)
                    # 九宫格的坐标
                    fmargin = (si / 3) / (offset_d + np.spacing(1))
                    # print(fmargin)
                    # np.spacing(1) 可以取一个数值X并返回“从abs(X)到下一个更大的相同精度的浮点数的正距离
                    fmargin = np.clip(fmargin, 0.05, 0.15)
                    # print(fmargin)
                    # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值
                    # fmargin = 0.0
                    frange = np.linspace(fmargin, 1.0 - fmargin, num=num)
                    filled_ij = set()  # 创建一个集合
                    for f in frange:  # 从 frange 开始
                        for xyo in xyv:  # 一堆奇怪的坐标？
                            fij = np.round(joint_1[:2] + f * offset + xyo).astype(np.int)  # + self.config.padding
                            # round() 方法返回浮点数x的四舍五入值。
                            if fij[0] < 0 or fij[0] >= self.heatmap_size[0] or \
                                    fij[1] < 0 or fij[1] >= self.heatmap_size[1]:
                                continue
                            fij_int = (int(fij[1]), int(fij[0]))
                            if fij_int in filled_ij:
                                continue
                            filled_ij.add(fij_int)  # 去重

                            # mask
                            # perpendicular distance computation:
                            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                            # Coordinate systems for this computation is such that
                            # joint1 is at (0, 0).
                            fxy = fij  # - self.config.padding
                            f_offset = fxy - joint_1[:2]  # 每个点到起点的偏置
                            sink_l = np.fabs(
                                offset[1] * f_offset[0]
                                - offset[0] * f_offset[1]
                            ) / (offset_d + 0.01)
                            # 点到线的距离？？？  offset y2 -y1 f_offset[0] x0-x1 某点到两点的距离
                            # 越小越近
                            # print(fij_int)
                            target_part[index][fij_int] = (3 - sink_l) / 3  # 图像时 y x
                            # target_part[0][fij_int] = (3-sink_l)/3 # 图像时 y x
                            # if sink_l > self.fields_reg_l[field_i, fij[1], fij[0]]:
                            #     continue
                            # fields_reg_l[field_i, fij[1], fij[0]] = sink_l
                            #
                            # 记录每个坐标 ij 的场强度 ?12
                            # target_part[index][fij[1], fij[0]] = keypoints[joint1i][:2] - fxy
                            # target_part[index][fij[1], fij[0]] = keypoints[joint2i][:2] - fxy

                    # # Generate gaussian
                    # sigma = 2
                    # tmp_size = sigma * 3

                    # 111111111111111111111111111111111111111111111111111111111111111111111111
                    # sigma_p = 0.5
                    # tmp_size_p = sigma_p * 3
                    # size_p = 2 * tmp_size_p + 1  # 7 # part 小一点
                    # # size = 7
                    # # size = 3
                    # # sigma = 2
                    # # tmp_size = sigma * 3
                    # x = np.arange(0, size_p, 1, np.float32)  # 0 1 2 3 4 5 6
                    # y = x[:, np.newaxis]  # (7,1）
                    # x0 = y0 = size_p // 2  # 整除 3
                    # # The gaussian is not normalized, we want the center value to equal 1  self.sigma 2
                    # # 小一点
                    # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma_p  ** 2))
                    # # 就改个sigma就行了 size不用动
                    # # Check that any part of the gaussian is in-bounds
                    # ul_1 = [int(mu_x_1 - tmp_size_p), int(mu_y_1 - tmp_size_p)]
                    # br_1 = [int(mu_x_1 + tmp_size_p + 1), int(mu_y_1 + tmp_size_p + 1)]
                    # ul_2 = [int(mu_x_2 - tmp_size_p), int(mu_y_2 - tmp_size_p)]
                    # br_2 = [int(mu_x_2 + tmp_size_p + 1), int(mu_y_2 + tmp_size_p + 1)]
                    # if ul_1[0] >= heatmap_size[0] or ul_1[1] >= heatmap_size[1] \
                    #         or br_1[0] < 0 or br_1[1] < 0 or ul_2[0] >= heatmap_size[0] \
                    #         or ul_2[1] >= heatmap_size[1] or br_2[0] < 0 or br_2[1] < 0:
                    #     # If not, just return the image as is
                    #     # target_part_weight[index] = 0
                    #     continue
                    #
                    # # Usable gaussian range
                    # g_x_1 = max(0, -ul_1[0]), min(br_1[0], heatmap_size[0]) - ul_1[0]
                    # g_y_1 = max(0, -ul_1[1]), min(br_1[1], heatmap_size[1]) - ul_1[1]
                    # g_x_2 = max(0, -ul_2[0]), min(br_2[0], heatmap_size[0]) - ul_2[0]
                    # g_y_2 = max(0, -ul_2[1]), min(br_2[1], heatmap_size[1]) - ul_2[1]
                    # # Image range
                    # img_x_1 = max(0, ul_1[0]), min(br_1[0], heatmap_size[0])
                    # img_y_1 = max(0, ul_1[1]), min(br_1[1], heatmap_size[1])
                    # img_x_2 = max(0, ul_2[0]), min(br_2[0], heatmap_size[0])
                    # img_y_2 = max(0, ul_2[1]), min(br_2[1], heatmap_size[1])
                    #
                    # v1 = target_weight[lb[0]]
                    # v2 = target_weight[lb[1]]
                    # if v1 > 0.5 and v2 > 0.5:
                    #     target_part[index][img_y_1[0]:img_y_1[1], img_x_1[0]:img_x_1[1]] = \
                    #         g[g_y_1[0]:g_y_1[1], g_x_1[0]:g_x_1[1]]
                    #     target_part[index][img_y_2[0]:img_y_2[1], img_x_2[0]:img_x_2[1]] = \
                    #         g[g_y_2[0]:g_y_2[1], g_x_2[0]:g_x_2[1]]

            for joint_id in range(num_joints):
                feat_stride = 4
                mu_x = int(joints[joint_id][0] / 4 + 0.5)
                mu_y = int(joints[joint_id][1] / 4 + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

                # if self.use_different_joints_weight:
            joints_weight = 1
            target_weight = np.multiply(target_weight, joints_weight)
            # print("######")
            # print(target.shape)
            # print(target_part.shape)
            # print("######")

        return target, target_part, target_weight, target_part_weight