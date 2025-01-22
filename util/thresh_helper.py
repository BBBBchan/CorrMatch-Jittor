import jittor as jt
from jittor import nn, Module
import numpy as np
from mpi4py import MPI

# 初始化MPI环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.thresh_global = jt.array([thresh_init]).stop_grad()
        self.momentum = momentum
        self.nclass = nclass
        self.gpu_num = world_size

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        n, c, h, w = pred.shape
        
        pred_list = [jt.zeros_like(pred) for _ in range(world_size)]
        comm.Allgather([pred.numpy(), MPI.FLOAT], [np.concatenate(pred_list, axis=0), MPI.FLOAT])
        pred_gather = jt.array(np.concatenate(pred_list, axis=0))
        
        if ignore_mask is not None:
            ignore_mask_list = [jt.zeros_like(ignore_mask) for _ in range(world_size)]
            comm.Allgather([ignore_mask.numpy(), MPI.INT], [np.concatenate(ignore_mask_list, axis=0), MPI.INT])
            ignore_mask_gather = jt.array(np.concatenate(ignore_mask_list, axis=0))
        else:
            ignore_mask_gather = None
        
        mask_pred = pred_gather.argmax(dim=1)[0]
        pred_softmax = pred_gather.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)
        unique_cls = jt.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0
        for cls in unique_cls:
            cls_map = (mask_pred == cls)
            if ignore_mask_gather is not None:
                cls_map *= (ignore_mask_gather != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf
        if cls_num > 0:
            return_dict['new_global'] = new_global / cls_num
        else:
            return_dict['new_global'] = None

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g and thresh['new_global'] is not None:
            self.thresh_global.update(self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global'])

    def get_thresh_global(self):
        return self.thresh_global

