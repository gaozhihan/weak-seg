import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import pickle
from PIL import Image
import random
from torchvision.transforms import functional as F

from arguments import get_args
import socket
from skimage.transform import resize
import matplotlib.pyplot as plt

from multiprocessing import Pool
import os

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import cv2


def pick_mask(image, mask, class_cur, map, adaptive_crf_setting):
    # should be pick_mask(self, maps, mask, preds), since within class function, so save any self. items
    mask_weight = mask
    num_class_cur = len(class_cur)
    score_color = np.zeros(adaptive_crf_setting['num_maps'])
    score_over_map = np.zeros(adaptive_crf_setting['num_maps'])
    score_map_iou = np.zeros(adaptive_crf_setting['num_maps']) # use to weight the confidence
    raw_map = np.argmax(mask, axis=0)

    # generate whole map sum score for use
    score_whole = np.zeros(num_class_cur)
    for i_idx, i_class in np.ndenumerate(class_cur):
        # score_whole[i_idx] = mask[i_class,:,:].sum()
        score_whole[i_idx] = (raw_map == i_class).sum()

    score_whole[score_whole==0] = 1 # just in case


    for i_map in range(adaptive_crf_setting['num_maps']):
        hist_cur = np.zeros([num_class_cur, adaptive_crf_setting['color_his_size'][0], adaptive_crf_setting['color_his_size'][1], adaptive_crf_setting['color_his_size'][2]])
        temp_map_overlap_soft = np.zeros(num_class_cur)
        temp_map_overlap_hard = np.zeros(num_class_cur)
        pre_map_whole = np.zeros(num_class_cur)
        for i_idx, i_class in np.ndenumerate(class_cur):
            mask_temp = np.zeros([image.shape[0], image.shape[1]],dtype = np.uint8)
            idx_temp = map[i_map,:,:] == i_class
            # calculate score based on color histogram separation
            mask_temp[idx_temp] = 1
            if mask_temp.sum() < 100: # based on my observation
                hist_cur[i_idx[0], :,:,:] = score_whole[i_idx]  # adaptive_crf_setting['num_pixel'], in case small object mess up with big ones
            else:
                hist_cur[i_idx[0], :,:,:] = cv2.calcHist([image.astype('float32')], adaptive_crf_setting['color_channels'], mask_temp, adaptive_crf_setting['color_his_size'], adaptive_crf_setting['color_ranges'])

            # calculate score based on consisitencey (overlap with raw mask)
            temp_map_overlap_soft[i_idx] = np.multiply(mask_temp, mask_weight[i_class,:,:]).sum()
            temp_map_overlap_hard[i_idx] = (np.multiply(mask_temp, raw_map == i_class)).sum()
            pre_map_whole[i_idx] = mask_temp.sum()


        # summary
        hist_cur = hist_cur.reshape([len(class_cur), -1])
        hist_cur = np.sort(hist_cur, axis=0)
        score_color[i_map] += hist_cur[:-1,:].sum()
        score_over_map[i_map] += temp_map_overlap_soft.sum()
        score_map_iou[i_map] = (temp_map_overlap_hard/(score_whole+pre_map_whole-temp_map_overlap_hard)).mean() # like iou

    iou_penalty = np.maximum(0.5 - score_map_iou, 0) * (raw_map>0).sum() # 100000
    score_overall = score_over_map - score_color - iou_penalty
    best_map_idx = np.argmax(score_overall)

    # np.set_printoptions(precision=3)
    # print(score_color)
    # print(score_over_map)
    # print(score_map_iou)
    # print(iou_penalty)
    # print(score_overall - score_overall.min())
    # print(best_map_idx)

    return best_map_idx, score_map_iou[best_map_idx], score_color[best_map_idx]



def crf(sm_mask_one, img_one, labels, num_class, input_size, mask_size, adaptive_crf_setting):
    sm_u = np.transpose(resize(np.transpose(sm_mask_one, [1,2,0]), input_size, mode='constant'),[2,0,1])
    map_buffer = np.zeros([adaptive_crf_setting['num_maps'], num_class, input_size[0], input_size[1]])
    map_pred_buffer = np.zeros([adaptive_crf_setting['num_maps'], input_size[0], input_size[1]])

    U = unary_from_softmax(sm_u)

    d = dcrf.DenseCRF2D(input_size[0], input_size[1], num_class)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q, tmp1, tmp2 = d.startInference()
    if adaptive_crf_setting['iters'][0] == 0:
        map_buffer[0,:,:,:] = np.asarray(Q).reshape((num_class, input_size[0], input_size[1]))
        # map_pred_buffer[0,:,:] = np.argmax(Q, axis=0).reshape((input_size[0], input_size[1]))
        map_pred_buffer[0,:,:] = np.argmax(map_buffer[0,:,:,:], axis=0)

    for i in range(adaptive_crf_setting['iters'][-1]):
        d.stepInference(Q, tmp1, tmp2)

        for ii in range(adaptive_crf_setting['num_maps']):
            if i+1 == adaptive_crf_setting['iters'][ii]:
                map_buffer[ii,:,:,:] = np.asarray(Q).reshape((num_class, input_size[0], input_size[1]))
                map_pred_buffer[ii,:,:] = np.argmax(Q, axis=0).reshape((input_size[0], input_size[1]))

    if (labels == None).any():
        map_raw = np.argmax(sm_mask_one, axis=0)
        class_cur = np.unique(map_raw)
    else:
        class_cur = np.nonzero(labels)[0]

    best_map_idx, map_iou_score, color_score = pick_mask(img_one, sm_u, class_cur, map_pred_buffer, adaptive_crf_setting)

    plt.figure()
    for i in range(adaptive_crf_setting['num_maps']):
        plt.subplot(1,adaptive_crf_setting['num_maps'], i+1); plt.imshow(map_pred_buffer[i,:,:]); plt.axis('off')

    return map_buffer[best_map_idx]


class STCRFLayer():
    def __init__(self, flag_multi_process = False):
        self.num_class = 21
        self.min_prob = 0.0001
        self.mask_size = [41, 41]
        self.input_size = [321, 321]
        self.num_iter = 5
        self.flag_multi_process = flag_multi_process
        if flag_multi_process:
            num_cores = os.cpu_count()
            self.pool = Pool(processes=num_cores)

        # define the params dictionary for crf
        self.adaptive_crf_setting = {}
        self.adaptive_crf_setting['iters'] = [0, 1, 3, 5]
        self.adaptive_crf_setting['color_his_size'] = [4, 4, 4]
        self.adaptive_crf_setting['num_color_bins'] = self.adaptive_crf_setting['color_his_size'][0]*self.adaptive_crf_setting['color_his_size'][1]*self.adaptive_crf_setting['color_his_size'][2]
        self.adaptive_crf_setting['color_channels'] = [0, 1, 2]
        self.adaptive_crf_setting['color_ranges'] = [0, 255, 0, 255, 0, 255]
        self.adaptive_crf_setting['color_score_scale'] = 1.5
        self.adaptive_crf_setting['num_maps'] = len(self.adaptive_crf_setting['iters'])
        self.adaptive_crf_setting['num_pixel'] = self.input_size[0] * self.input_size[1]
        self.adaptive_crf_setting['color_score_thr'] = self.adaptive_crf_setting['num_pixel'] * 0.04

    def run(self, sm_mask, img, labels=None):
        self.mask_size = [sm_mask.shape[-2], sm_mask.shape[-1]]
        self.input_size = [img.shape[-3], img.shape[-2]]
        self.adaptive_crf_setting['num_pixel'] = self.input_size[0] * self.input_size[1]
        self.adaptive_crf_setting['color_score_thr'] = self.adaptive_crf_setting['num_pixel'] * 0.04

        if self.flag_multi_process:
            return self.run_parallel(sm_mask, img, labels)
        else:
            return self.run_single(sm_mask, img, labels)

    def run_single(self, sm_mask, img, labels): # the input array are detached numpy already
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)
        for i in range(batch_size):
            result_big[i] = crf(sm_mask[i], img[i], labels[i].squeeze(), self.num_class, self.input_size, self.mask_size, self.adaptive_crf_setting)
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small

    def run_parallel(self, sm_mask, img, labels): # flag_train is for the strange dif between train & test in org SEC
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)

        # temp = self.pool.starmap(self.crf,[(sm_mask[i], img[i]) for i in range(batch_size)])
        temp = self.pool.starmap(crf,[(sm_mask[i], img[i], labels[i].squeeze(), self.num_class, self.input_size, self.mask_size, self.adaptive_crf_setting) for i in range(batch_size)])
        for i in range(batch_size):
            result_big[i] = temp[i]
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small


def mend_mask_by_labels(mask_np, labels):
    num_batch = labels.shape[0]
    num_class = labels.shape[1]

    for i_batch in range(num_batch):
        cur_class = np.nonzero(labels[i_batch])[0]
        cur_non_class = np.nonzero(labels[i_batch]==0)[0]
        num_cur_class = len(cur_class)
        max_value_non_cur_class = 1e-3
        min_value_cur_class = 0.96/num_cur_class

        # cut big value in non present class
        non_class_map = mask_np[i_batch,cur_non_class,:,:]
        # print((non_class_map>=min_value_cur_class).sum())
        non_class_map[non_class_map>=min_value_cur_class] = max_value_non_cur_class
        mask_np[i_batch,cur_non_class,:,:] = non_class_map

        # for cur class
        cur_class_map = mask_np[i_batch,cur_class,:,:]
        if num_cur_class>1:
            temp = np.max(cur_class_map, axis=0)
            idx_temp = temp<min_value_cur_class
            for i in range(num_cur_class):
                temp = cur_class_map[i]
                temp[idx_temp] = min_value_cur_class
                cur_class_map[i] = temp

        else:
            cur_class_map[cur_class_map<min_value_cur_class] = min_value_cur_class

        mask_np[i_batch,cur_class,:,:] = cur_class_map
        mask_np[i_batch,:,:,:] = mask_np[i_batch,:,:,:]/mask_np[i_batch,:,:,:].sum(axis=0)

    return mask_np


# just make sure the max value appear in the cur_class_map
def min_mend_mask_by_labels(mask_np, labels): # just make sure the max value appear in the cur_class_map
    num_batch = labels.shape[0]
    num_class = labels.shape[1]
    margin = 1e-4

    for i_batch in range(num_batch):
        # plt.figure()
        # plt.subplot(1,2,1); plt.imshow(np.argmax(mask_np[i_batch,:,:,:], axis=0)); plt.axis('off')
        cur_class = np.nonzero(labels[i_batch])[0]
        cur_non_class = np.nonzero(labels[i_batch]==0)[0]
        num_cur_class = len(cur_class)
        num_non_class = len(cur_non_class)

        non_class_map = mask_np[i_batch,cur_non_class,:,:]
        cur_class_map = mask_np[i_batch,cur_class,:,:]
        ceiling_map = np.maximum(cur_class_map.max(axis=0) - margin, margin)

        # regulate all the maps in non_class_map do not exceed the ceiling
        for i_map in range(num_non_class):
            non_class_map[i_map,:,:] = np.minimum(ceiling_map, non_class_map[i_map,:,:])

        mask_np[i_batch,cur_non_class,:,:] = non_class_map
        mask_np[i_batch,:,:,:] = mask_np[i_batch,:,:,:]/mask_np[i_batch,:,:,:].sum(axis=0)
        # plt.subplot(1,2,2); plt.imshow(np.argmax(mask_np[i_batch,:,:,:], axis=0)); plt.axis('off')

    return mask_np

# just make sure the max value appear in the cur_class_map
def min_mend_floor_mask_by_labels(mask_np, labels): # just make sure the max value appear in the cur_class_map
    num_batch = labels.shape[0]
    num_class = labels.shape[1]
    margin = 1e-4

    for i_batch in range(num_batch):
        # plt.figure()
        # plt.subplot(1,2,1); plt.imshow(np.argmax(mask_np[i_batch,:,:,:], axis=0)); plt.axis('off')
        cur_class = np.nonzero(labels[i_batch])[0]
        cur_non_class = np.nonzero(labels[i_batch]==0)[0]
        num_cur_class = len(cur_class)
        num_non_class = len(cur_non_class)

        non_class_map = mask_np[i_batch,cur_non_class,:,:]
        cur_class_map = mask_np[i_batch,cur_class,:,:]
        floor_map = non_class_map.max(axis=0) + margin

        # regulate all the maps in non_class_map do not exceed the ceiling
        for i_map in range(num_cur_class):
            cur_class_map[i_map,:,:] = np.maximum(floor_map, cur_class_map[i_map,:,:])

        mask_np[i_batch,cur_class,:,:] = cur_class_map
        mask_np[i_batch,:,:,:] = mask_np[i_batch,:,:,:]/mask_np[i_batch,:,:,:].sum(axis=0)
        # plt.subplot(1,2,2); plt.imshow(np.argmax(mask_np[i_batch,:,:,:], axis=0)); plt.axis('off')

    return mask_np
