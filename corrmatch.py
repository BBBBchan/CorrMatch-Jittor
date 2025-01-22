import argparse
import logging
import os
import pprint
from mpi4py import MPI
from tqdm import tqdm
import numpy as np
import jittor as jt
from jittor import nn, Module
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import yaml
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from evaluate import evaluate
from util.utils import count_params, init_log
from util.thresh_helper import ThreshController
from jittor.einops import rearrange
import random


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def execute(self, input, target):
        """
        Args:
            input: (N, C, H, W) where C = number of classes
            target: (N, H, W) where each value is 0 <= targets[i, j, k] <= C-1 or equal to ignore_index
        Returns:
            loss: (N, H, W)
        """
        loss_per_pixel = jt.nn.cross_entropy_loss(input, target, ignore_index=self.ignore_index, reduction=self.reduction)
        
        return loss_per_pixel

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    logger.info('{}\n'.format(pprint.pformat(cfg)))

    os.makedirs(args.save_path, exist_ok=True)
    init_seeds(0)

    model = DeepLabV3Plus(cfg).cuda()
    optimizer = nn.SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    criterion_u = CrossEntropyLoss(reduction='none').cuda()
    criterion_kl = nn.KLDivLoss(reduction='none').cuda()

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = None  # Jittor does not require explicit samplers
    trainloader_l = trainset_l.set_attrs(batch_size=cfg['batch_size'], shuffle=True,
                                num_workers=4, drop_last=True)
    # print(len(trainloader_l))
    # exit()
    trainsampler_u = None
    trainloader_u = trainset_u.set_attrs(batch_size=cfg['batch_size'], shuffle=True,
                                num_workers=4, drop_last=True)
    valsampler = None
    valloader = valset.set_attrs(batch_size=1,
                                  shuffle=False, drop_last=False, num_workers=4)


    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    thresh_controller = ThreshController(nclass=21, momentum=0.999, thresh_init=cfg['thresh_init'])

    for epoch in range(cfg['epochs']):

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'

        res_val = evaluate(model, valloader, eval_mode, cfg)

        logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_loss_kl = 0.0
        total_loss_corr_ce, total_loss_corr_u = 0.0, 0.0
        total_mask_ratio = 0.0


        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        tbar = tqdm(total=len(trainloader_l))

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
                (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.float32().cuda(), mask_x.long().cuda()
            img_u_w = img_u_w.float32().cuda()
            img_u_s1, ignore_mask = img_u_s1.float32().cuda(), ignore_mask.long().cuda()
            cutmix_box1 = cutmix_box1.uint8().cuda()
            img_u_w_mix = img_u_w_mix.float32().cuda()
            img_u_s1_mix = img_u_s1_mix.float32().cuda()
            ignore_mask_mix = ignore_mask_mix.long().cuda()
            b, c, h, w = img_x.shape

            with jt.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix['out'].stop_grad()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)[0]
                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            res_w = model(jt.concat((img_x, img_u_w)), need_fp=True, use_corr=True)

            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            preds_corr_map = res_w['corr_map'].stop_grad()
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s['out']
            pred_u_s1_corr = res_s['corr_out']

            pred_u_w = pred_u_w.stop_grad()
            conf_u_w = pred_u_w.stop_grad().softmax(dim=1).max(dim=1)
            mask_u_w = pred_u_w.stop_grad().argmax(dim=1)[0]
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.copy(), conf_u_w.copy(), ignore_mask.copy()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.copy()
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            cutmix_box1_map = (cutmix_box1 == 1)

            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            mask_u_w_cutmixed1_copy = mask_u_w_cutmixed1.copy()
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]
            cutmix_box1_sample = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            ignore_mask_cutmixed1_sample = rearrange((ignore_mask_cutmixed1 != 255), 'n h w -> n 1 h w')
            corr_map_u_w_cutmixed1 = (corr_map_u_w_cutmixed1 * jt.logical_not(cutmix_box1_sample) * ignore_mask_cutmixed1_sample).bool()

            thresh_controller.thresh_update(pred_u_w.stop_grad(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()

            conf_fliter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255))
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w.copy()
            conf_fliter_u_w_sample = rearrange(conf_fliter_u_w_without_cutmix, 'n h w -> n 1 h w')

            segments = (corr_map_u_w_cutmixed1 * conf_fliter_u_w_sample).bool()

            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):

                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    high_conf_ratio = jt.sum(segment)/jt.sum(segment_ori)
                    if jt.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue
                    unique_cls, _,count = jt.unique(mask_u_w_cutmixed1[img_idx][segment==1], return_inverse=True, return_counts=True)
                    if jt.max(count) / jt.sum(count) > thresh_global:
                        top_class = unique_cls[jt.argmax(count,dim=0)[0]]
                        mask_u_w_cutmixed1[img_idx][segment_ori==1] = top_class
                        conf_fliter_u_w_without_cutmix[img_idx] = conf_fliter_u_w_without_cutmix[img_idx] | segment_ori
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w_without_cutmix | conf_fliter_u_w

            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w_without_cutmix
            loss_u_s1 = jt.sum(loss_u_s1) / jt.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = loss_u_corr_s1 * conf_fliter_u_w_without_cutmix
            loss_u_corr_s1 = jt.sum(loss_u_corr_s1) / jt.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_corr_s = loss_u_corr_s1

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = jt.sum(loss_u_corr_w) / jt.sum(ignore_mask != 255).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = nn.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = nn.log_softmax(pred_u_s1, dim=1)

            loss_u_kl_sa2wa = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl_sa2wa = jt.sum(loss_u_kl_sa2wa, dim=1) * conf_fliter_u_w
            loss_u_kl_sa2wa = jt.sum(loss_u_kl_sa2wa) / jt.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_kl = loss_u_kl_sa2wa

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = jt.sum(loss_u_w_fp) / jt.sum(ignore_mask != 255).item()

            loss = (0.5 * loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()


            total_loss += loss
            total_loss_x += loss_x
            total_loss_s += loss_u_s1
            total_loss_kl += loss_u_kl
            total_loss_w_fp += loss_u_w_fp
            total_loss_corr_ce += loss_x_corr
            total_loss_corr_u += loss_u_corr
            total_mask_ratio += ((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum() / \
                                (ignore_mask != 255).sum()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            
            tbar.set_description(' Total loss: {:.3f}, Loss x: {:.3f}, loss_corr_ce: {:.3f} '
                                     'Loss s: {:.3f}, Loss w_fp: {:.3f},  Mask: {:.3f}, loss_corr_u: {:.3f}'.format(
                    total_loss / (i + 1), total_loss_x / (i + 1), total_loss_corr_ce / (i + 1), total_loss_s / (i + 1),
                    total_loss_w_fp / (i + 1), total_mask_ratio / (i + 1), total_loss_corr_u / (i + 1)))
            tbar.update(1)

        
        tbar.close()

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'

        res_val = evaluate(model, valloader, eval_mode, cfg)
        mIOU = res_val['mIOU']
        class_IOU = res_val['iou_class']
        jt.sync_all()

        
        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.4f} \n'.format(eval_mode, mIOU))
        logger.info('***** ClassIOU ***** >>>> \n{}\n'.format(class_IOU))

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.3f.pkl' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            model.save(os.path.join(args.save_path, '%s_%.3f.pkl' % (cfg['backbone'], mIOU)))

        jt.sync_all()


if __name__ == '__main__':
    main()



