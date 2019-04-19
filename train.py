#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
# suppress caffe log
os.environ['GLOG_minloglevel'] = '2' 


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.pyplot import savefig
import caffe

from os import path
import sys
import ipdb
from tqdm import tqdm
from pprint import pprint
import time

#######################################


gpu_ids = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
print ('GPU ids:', gpu_ids)

caffe.set_mode_gpu()
caffe.set_device(0)


# files info
#######################################
root_path = '/home/zhangwenqiang/my-train-hand-segment/runs'
solver_proto_f = path.join(root_path, 'solver_segment_distilled.prototxt')
#######################################


def parse_solver_prototxt_to_dict(solver_proto_f):
    solver_dict = {}
    for ln in open(solver_proto_f):
        if ln.lstrip()[0] == '#':
            continue
        ln = ln.split('#')[0]
        key, val = map(str.strip, ln.split(':'))
        if val[0] in ["'", '"']:
            val = val[1:-1]
        else:
            try:
                val = float(val)
                if int(val) == val:
                    val = int(val)
            except Exception:
                pass
        solver_dict[key] = val
    return solver_dict


solver_dict = parse_solver_prototxt_to_dict(solver_proto_f)

pprint (solver_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

solver_params = dotdict(solver_dict)

net_proto_f = solver_params.net

print ('net: {} \nsolver: {} '.format(net_proto_f, solver_proto_f))


max_iter = solver_params.max_iter
display = solver_params.display
test_iter = solver_params.test_iter
test_interval = solver_params.test_interval
snapshot_prefix = solver_params.snapshot_prefix

best_rec_file = path.join(path.dirname(snapshot_prefix), 'best.txt')
print ('best rec is stored at {}'.format(best_rec_file))


# auxiliary hyperparams
plot_interval = min(display, test_interval)
print ('plot_interval:', plot_interval)

exp_name = 'test_v1'
print ('exp_name:', exp_name)

DEBUG_MODE = False
print ('DEBUG_MODE:', DEBUG_MODE)


solver = caffe.get_solver(solver_proto_f)

train_loss = np.zeros(int(np.ceil(max_iter * 1.0 / display)))
test_loss = np.zeros(int(np.ceil(max_iter * 1.0 / test_interval)))
test_iou = np.zeros(int(np.ceil(max_iter * 1.0 / test_interval)))

_train_loss = 0
_test_loss = 0
_iou = 0

example_cnt = 0

def update_plot(plot_name, it):
    plot_file = path.join(root_path, "{}.png".format(plot_name))
    if not os.path.exists(plot_file):
        print ('Plot to {}'.format(plot_file))
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # train loss -> 绿色
    ax1.plot(display * np.arange(it // display + 1), train_loss[:it // display + 1], 'g')
    # test loss -> 黄色
    ax1.plot(test_interval * np.arange(it // test_interval + 1), test_loss[:it // test_interval + 1], 'y')
    # test accuracy -> 红色
    ax2.plot(test_interval * np.arange(it // test_interval + 1), test_iou[:it // test_interval + 1], 'r')

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('miou')
    # plt.show()
    savefig(plot_file)
    plt.close()


def calc_miou_single(gt, pred):
    img_size = gt.shape[0]
    assert img_size == 128, 'img_size = 128'

    cls_c1 = 0.0
    cls_c11 = 0.0
    cls_c0 = 0.0
    cls_c00 = 0.0
        
    for r in range(img_size):
        for c in range(img_size):
            gt_val = gt[r, c]
            pre_val = pred[r, c]
            # gt: {0,1}
            # pred: before sigmoid, [-N, +M]
            if gt_val >= 0.5:
                cls_c1 += 1
                if pre_val >= 0:
                    cls_c11 += 1
            elif gt_val < 0.5:
                cls_c0 += 1
                if pre_val < 0:
                    cls_c00 += 1       
    mIoU = 1 / 2.0 * (cls_c00 / (cls_c0 + cls_c1 - cls_c11) + cls_c11 / (cls_c1 + cls_c0 - cls_c00))
    return mIoU


def calc_miou(gt_seg, pred_seg):
    batch_size = gt_seg.shape[0]
    batch_miou = 0
    for b in range(batch_size):
        batch_miou += calc_miou_single(gt_seg[b][0], pred_seg[b][0])
    return batch_miou
    

tbar = tqdm(range(max_iter))

cur_i = 0

previous_best = -1

for it in tbar:
    solver.step(1)
    _train_loss += solver.net.blobs['loss_stage1_L2'].data
    cur_i += 1
    tbar.set_description('train loss: {}'.format(_train_loss / cur_i))
    # display
    if it % display == 0:
        recent_train_loss = _train_loss / (1 if it == 0 else display)
        tqdm.write ('{} Iter [{}/{}], train loss: {:.2f}'.format(time.strftime("%m-%d %H:%M:%S", time.localtime()), it, max_iter, recent_train_loss))
        train_loss[it // display] = recent_train_loss
        _train_loss = 0
        cur_i = 0
    
    # test
    if it % test_interval == 0:
        tbar.set_description('testing ... ')
        for test_it in range(test_iter):
            test_net = solver.test_nets[0]
            test_net.forward()
            _test_loss += test_net.blobs['loss_stage1_L2'].data
            gt_seg = test_net.blobs['heatmap'].data
            pred_seg = test_net.blobs['up2'].data
            example_cnt += gt_seg.shape[0]
            _iou += calc_miou(gt_seg, pred_seg)

        recent_test_loss = _test_loss / test_iter
        test_loss[it // test_interval] = recent_test_loss

        if DEBUG_MODE:
            tqdm.write ('test example cnt:', example_cnt)
        
        recent_test_iou = _iou / example_cnt
        is_best = False
        if recent_test_iou > previous_best:
            previous_best = recent_test_iou
            is_best = True
        test_iou[it // test_interval] = recent_test_iou
        tqdm.write ('{} Iter [{}/{}], test loss: {:.2f}, test mIoU: {:.2%}'.format(time.strftime("%m-%d %H:%M:%S", time.localtime()), it, max_iter, recent_test_loss, recent_test_iou))

        _test_loss = 0
        _iou = 0
        example_cnt = 0

        if is_best:
            tqdm.write('is_best!')
            with open(best_rec_file, 'w') as f:
                f.write(str(previous_best))
            solver.snapshot()
    
    # plot
    if it % plot_interval == 0:
        update_plot(exp_name, it)
