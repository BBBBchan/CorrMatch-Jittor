import argparse
import os
import numpy as np
import yaml
from mpi4py import MPI
from tqdm import tqdm

import jittor as jt
from jittor import nn, Module
from model.semseg.deeplabv3plus import DeepLabV3Plus

from jittor.dataset import Dataset
from dataset.semi import SemiDataset
from util.utils import AverageMeter, intersectionAndUnion


def evaluate(model, loader, mode, cfg):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with jt.no_grad():
        for img, mask, ids, img_ori in tqdm(loader):
            img = img.cuda()
            b, _, h, w = img.shape
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                final = jt.zeros((b, 19, h, w)).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        res = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        pred = res['out']
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)[0]

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                res = model(img)
                pred = res['out'].argmax(dim=1)[0]

            intersection, union, target = \
                intersectionAndUnion(pred.numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = jt.array(intersection).cuda()
            reduced_union = jt.array(union).cuda()
            reduced_target = jt.array(target).cuda()

            # Use MPI to reduce the metrics across all processes
            comm = MPI.COMM_WORLD
            reduced_intersection_np = np.zeros_like(reduced_intersection.numpy())
            reduced_union_np = np.zeros_like(reduced_union.numpy())
            reduced_target_np = np.zeros_like(reduced_target.numpy())

            comm.Allreduce(reduced_intersection.numpy(), reduced_intersection_np, op=MPI.SUM)
            comm.Allreduce(reduced_union.numpy(), reduced_union_np, op=MPI.SUM)
            comm.Allreduce(reduced_target.numpy(), reduced_target_np, op=MPI.SUM)

            intersection_meter.update(reduced_intersection_np)
            union_meter.update(reduced_union_np)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0
    return_dict['iou_class'] = iou_class
    return_dict['mIOU'] = mIOU
    # print(mIOU, iou_class)

    return return_dict
