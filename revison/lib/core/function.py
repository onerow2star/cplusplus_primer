# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by lyh
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

# from apex import amp
import cv2
import numpy as np
import torch

# for test
# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold=np.inf)

from core.evaluate import accuracy
from core.inference import get_final_preds
import multiprocessing
from torchvision import transforms

from utils.transforms import flip_back
from utils.vis import save_debug_images

from utils.transforms import get_affine_transform


logger = logging.getLogger('train')


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # if model has two outputs
    tom = True

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target, target_part, target_weight, target_part_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        target_part = target_part.cuda(non_blocking=True)
        target_part_weight = target_part_weight.cuda(non_blocking=True)

        # compute output
        if tom:
            point_outputs, part_outputs = model(input.cuda(non_blocking=True))
            point_loss = criterion(point_outputs, target, target_weight)
            part_loss = criterion(part_outputs, target_part, target_part_weight)
            # loss = point_loss + 0.1 * part_loss  # the parameter can be changed
            loss = point_loss + 0.1 * part_loss
        else:
            point_outputs = model(input.cuda(non_blocking=True))
            loss = criterion(point_outputs, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(point_outputs.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # save image
            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred * 4, output,
            #                   prefix)



def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    # if model has two outputs
    # tom = False
    tom = True

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_part, target_weight, target_part_weight, meta) in enumerate(val_loader):

            # compute output
            outputs = model(input.cuda(non_blocking=True))

            if tom:
                output = outputs[0]
                po = outputs[1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped.cuda(non_blocking=True))

                if tom:
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            idx += num_images
            cip = []
            cip.extend(meta['image'])

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                # prefix = '{}_{}'.format(
                #     os.path.join(output_dir, 'val'), i
                # )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                   prefix)

            # show
            gtjoints = meta['gtjoints'].numpy()
            joints_vis = meta['joints_vis'].numpy()
            # print(gtjoints)
            # print(gtjoints.shape)
            # if i % config.PRINT_FREQ == 0:
            #     # print("c")
            #     # print(input.size(0))
            #     # 32 batchsize
            #     for bt in range(input.size(0)):
            #         # mc = np.transpose(preds, [1, 2, 0]) # yu ce jie guo de zuo biao
            #         # print(preds[bt].shape)
            #         # print(cip[bt])
            #         cid = cv2.imread(cip[bt], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #         sp = cid.shape
            #         # draw points
            #         num_joints = 16
            #
            #         #draw point
            #         # for j in range(num_joints):
            #         #     cv2.circle(cid, (int(preds[bt][j][0]), int(preds[bt][j][1])), 3, [0, 165, 255], 3) # blue pred
            #         #     cv2.circle(cid, (int(gtjoints[bt][j][0] + 0.5), int(gtjoints[bt][j][1] + 0.5)), 3, [0, 255, 0], 3)
            #
            #             # sfm = cv2.resize(fm[j].cpu().numpy(), (256, 256))
            #         limbs_links = [[0, 1], [1, 2], [2, 6], [3, 6], [4, 3], [5, 4], [6, 7], [7, 8], [8, 9], [10, 11],
            #                        [11, 12],
            #                        [12, 8], [13, 8], [14, 13], [15, 14]]
            #         skeletons = np.array(limbs_links)
            #         # print("c1")
            #         # for index, sk in enumerate(skeletons):
            #         #     if np.all(joints_vis[bt][sk][0][0] > 0):  # Test whether all array elements along a given axis evaluate to True
            #         #         if index in list([0,1,4,5,9,10,13,14]):
            #         #             cv2.line(cid, (int(gtjoints[bt][sk[0]][0]), int(gtjoints[bt][sk[0]][1])),
            #         #                     (int(gtjoints[bt][sk[1]][0]), int(gtjoints[bt][sk[1]][1])), [0, 255, 0], 5)
            #         #             cv2.line(cid, (int(preds[bt][sk[0]][0]), int(preds[bt][sk[0]][1])),
            #         #                     (int(preds[bt][sk[1]][0]), int(preds[bt][sk[1]][1])), [0, 165, 255], 5)
            #         # print("c2")
            #         for v in range(15):
            #             if v in list([0, 1, 4, 5, 9, 10, 13, 14]):
            #                 fm = np.clip(po[bt][v].cpu().numpy() * 255, 0, 255)
            #                 sfm = cv2.resize(fm, (256, 256)).astype(np.uint8)
            #                 r = 0
            #                 trans2 = get_affine_transform(c[bt], s[bt], r, (256, 256), inv=1)  # c s shi cheng dui de
            #                 osfm = cv2.warpAffine(
            #                     sfm, trans2, (sp[1], sp[0]),
            #                     flags=cv2.INTER_LINEAR
            #                 )
            #                 osfm = np.clip(osfm, 0, 255)
            #                 ch = cv2.applyColorMap(osfm, cv2.COLORMAP_JET)
            #                 mi = ch * 0.7 + cid * 0.3
            #                 # cv2.imwrite('/show/p{}_{}_{}.png'.format(idx,bt,v), mi.astype(np.uint8))
            #                 # print("c3")
            #                 # cv2.imshow('image', mi.astype(np.uint8))
            #                 # cv2.waitKey(0)
            #                 # cv2.imwrite('/show/p{}_{}_{}.jpg'.format(idx, bt, v), mi.astype(np.uint8))
            #                 cv2.imwrite('show/p{}_{}_{}.jpg'.format(idx, bt, v), mi.astype(np.uint8))
            # yong zui yuan shi de fang fa1
            # image_path shi ge yuan zu

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# Multi-scaletest by MaxChu719
def read_scaled_image(image_file, s, center, scale, image_size, COLOR_RGB, DATA_FORMAT, image_transform):
    if DATA_FORMAT == 'zip':
        from utils import zipreader
        data_numpy = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if COLOR_RGB:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    trans = get_affine_transform(center, s * scale, 0, image_size)
    images_warp = cv2.warpAffine(data_numpy, trans, tuple(image_size), flags=cv2.INTER_LINEAR)
    return image_transform(images_warp)

def validate_1(config, val_loader, val_dataset, model, criterion, output_dir,
            tb_log_dir, writer_dict=None, test_scale=[1.0]):
    test_scale = [0.8,0.9,1.0,1.1,1.2,1.3]
    # 2.0 baocuo
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    # if model has two outputs
    tom = False

    PRINT_FREQ = min(config.PRINT_FREQ//10, 5)
    # a // b a/b得到的最小整数
    image_size = np.array(config.MODEL.IMAGE_SIZE)
    final_test_scale = test_scale if test_scale is not None else config.TEST.SCALE_FACTOR
    with torch.no_grad():
        end = time.time()

        def scale_back_output(output_hm, s, output_size):
            hm_size = [output_hm.size(3), output_hm.size(2)]
            print(output_hm.shape)
            if s != 1.0:
                hm_w_margin = int(abs(1.0 - s) * hm_size[0] / 2.0)
                hm_h_margin = int(abs(1.0 - s) * hm_size[1] / 2.0)
                # print(hm_w_margin)
                # print(hm_h_margin)
                if s < 1.0:
                    hm_padding = torch.nn.ZeroPad2d((hm_w_margin, hm_w_margin, hm_h_margin, hm_h_margin))
                    resized_hm = hm_padding(output_hm)
                else:
                    resized_hm = output_hm[:, :, hm_h_margin:hm_size[0] - hm_h_margin, hm_w_margin:hm_size[1] - hm_w_margin]

                # print(resized_hm.shape)
                resized_hm = torch.nn.functional.interpolate(
                    resized_hm,
                    size=(output_size[1], output_size[0]),
                    mode='bilinear',  # bilinear bicubic
                    align_corners=False
                )
            else:
                resized_hm = output_hm
                if hm_size[0] != output_size[0] or hm_size[1] != output_size[1]:
                    resized_hm = torch.nn.functional.interpolate(
                        resized_hm,
                        size=(output_size[1], output_size[0]),
                        mode='bilinear',  # bilinear bicubic
                        align_corners=False
                    )

            # resized_hm = torch.nn.functional.normalize(resized_hm, dim=[2, 3], p=1)
            resized_hm = resized_hm/(torch.sum(resized_hm, dim=[2, 3], keepdim=True) + 1e-9)
            return resized_hm

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_transform = transforms.Compose([transforms.ToTensor(), normalize])
        thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())

        start_time = time.time()
        # for i, (input, target, target_weight, meta) in enumerate(val_loader):
        for i, (input, target, target_part, target_weight, target_part_weight, meta) in enumerate(val_loader):
            # compute output
            # print("Batch", i, "Batch Size", input.size(0))

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            outputs = []
            for sidx, s in enumerate(sorted(final_test_scale, reverse=True)):
                print("Test Scale", s)
                if s != 1.0:
                    image_files = meta["image"]
                    centers = meta["center"].numpy()
                    scales = meta["scale"].numpy()

                    # images_resized = []
                    # for (image_file, center, scale) in zip(image_files, centers, scales):
                    #     scaled_image = read_scaled_image(image_file, center, scale, config.DATASET.COLOR_RGB)
                    #     images_resized.append(scaled_image)

                    images_resized = thread_pool.starmap(read_scaled_image,
                                                         [(image_file, s, center, scale, image_size, config.DATASET.COLOR_RGB, config.DATASET.DATA_FORMAT, image_transform) for (image_file, center, scale) in zip(image_files, centers, scales)])

                    images_resized = torch.stack(images_resized, dim=0)
                else:
                    images_resized = input

                model_outputs = model(images_resized)
                if tom:
                    model_outputs = model_outputs[0]
                    # po = outputs[1]
                else:
                    model_outputs = model_outputs
                # print(model_outputs.shape)
                # torch.Size([1, 16, 64, 64])

                hm_size = [model_outputs.size(3), model_outputs.size(2)]
                # print(hm_size)
                # hm_size = image_size
                # hm_size = [128, 128]

                if config.TEST.FLIP_TEST:
                    print("Test Flip")
                    input_flipped = images_resized.flip(3)
                    outputs_flipped = model(input_flipped)
                    if tom:
                        output_flipped = outputs_flipped[0]
                    else:
                        output_flipped = outputs_flipped
                    output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
                    model_outputs = 0.5 * (model_outputs + output_flipped)
                    # print(model_outputs.shape)
                    # output_flipped_resized = scale_back_output(output_flipped, s, hm_size)
                    # outputs.append(output_flipped_resized)
                output_flipped_resized = scale_back_output(model_outputs, s, hm_size)
                outputs.append(output_flipped_resized)
            target_size = [target.size(3), target.size(2)]
            if hm_size[0] != target_size[0] or hm_size[1] != target_size[1]:
                target = torch.nn.functional.interpolate(
                    target,
                    size=hm_size,
                    mode='bilinear',  # bilinear bicubic
                    align_corners=False
                )
                target = torch.nn.functional.normalize(target, dim=[2, 3], p=2)
            for indv_output in outputs:
                _, avg_acc, _, _ = accuracy(indv_output.cpu().numpy(), target.cpu().numpy())
                print("Indv Accuracy", avg_acc)
            output = torch.stack(outputs, dim=0).mean(dim=0)
            loss = criterion(output, target, target_weight)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            print("Avg Accuracy", avg_acc)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc)
                logger.info(msg)

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                # save_debug_images(config, input, meta, target, pred*4, output, prefix)

        total_duration = time.time() - start_time
        logger.info("Total test time: {:.1f}".format(total_duration))
        name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums)

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def read_scaled_image(image_file, s, center, scale, image_size, COLOR_RGB, DATA_FORMAT, image_transform):
    if DATA_FORMAT == 'zip':
        from utils import zipreader
        data_numpy = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if COLOR_RGB:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    trans = get_affine_transform(center, s * scale, 0, image_size)
    images_warp = cv2.warpAffine(data_numpy, trans, tuple(image_size), flags=cv2.INTER_LINEAR)
    return image_transform(images_warp)




