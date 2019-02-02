import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from krahenbuhl2013 import CRF
from skimage.transform import resize
from joblib import Parallel, delayed
from multiprocessing import Pool
import os

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


# --- my modified part -----------------------------------------------
def crf(sm_mask_one, img_one, num_class, input_size, mask_size, num_iter, adaptive_crf_setting):

    map = np.zeros([adaptive_crf_setting['num_maps'], self.num_maps, self.H, self.W])
    sm_u = np.transpose(resize(np.transpose(sm_mask_one, [1,2,0]), input_size, mode='constant'),[2,0,1])

    U = unary_from_softmax(sm_u)

    d = dcrf.DenseCRF2D(input_size[0], input_size[1], num_class)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(num_iter)
    return np.array(Q).reshape((num_class, input_size[0], input_size[1]))





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

        self.W_input = 321
        self.H_input = 321

        # define the params dictionary for crf
        self.adaptive_crf_setting = {}
        self.adaptive_crf_setting['iters'] = [0, 1, 3, 5, 8]
        self.adaptive_crf_setting['color_his_size'] = [4, 4, 4]
        self.adaptive_crf_setting['num_color_bins'] = self.adaptive_crf_setting['color_his_size'][0]*self.adaptive_crf_setting['color_his_size'][1]*self.adaptive_crf_setting['color_his_size'][2]
        self.adaptive_crf_setting['color_channels'] = [0, 1, 2]
        self.adaptive_crf_setting['color_ranges'] = [0, 255, 0, 255, 0, 255]
        self.adaptive_crf_setting['color_score_scale'] = 1.5
        self.adaptive_crf_setting['num_maps'] = len(self.adaptive_crf_setting['iters'])
        self.adaptive_crf_setting['num_pixel'] = self.H_input * self.W_input
        self.adaptive_crf_setting['color_score_thr'] = self.adaptive_crf_setting['num_pixel'] * 0.04



    def set_shape(self, img):
        self.H_input = img.shape[2]
        self.W_input = img.shape[3]
        self.adaptive_crf_setting['num_pixel'] = self.H_input * self.W_input
        self.adaptive_crf_setting['color_score_thr'] = self.adaptive_crf_setting['num_pixel'] * 0.04



    def run(self, sm_mask, img):
        self.set_shape(img)

        if self.flag_multi_process:
            return self.run_parallel(sm_mask, img, self.adaptive_crf_setting)
        else:
            return self.run_single(sm_mask, img)

    def run_single(self, sm_mask, img): # the input array are detached numpy already
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)
        for i in range(batch_size):

            sm_u = np.transpose(resize(np.transpose(sm_mask[i], [1,2,0]), self.input_size, mode='constant'),[2,0,1])

            U = unary_from_softmax(sm_u)

            d = dcrf.DenseCRF2D(self.input_size[0], self.input_size[1], self.num_class)
            d.setUnaryEnergy(U)

            d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

            Q = d.inference(self.num_iter)
            result_big[i] = np.array(Q).reshape((self.num_class, self.input_size[0], self.input_size[1]))
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small

    def run_parallel(self, sm_mask, img): # flag_train is for the strange dif between train & test in org SEC
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)

        # temp = self.pool.starmap(self.crf,[(sm_mask[i], img[i]) for i in range(batch_size)])
        temp = self.pool.starmap(crf,[(sm_mask[i], img[i], self.num_class, self.input_size, self.mask_size, self.num_iter) for i in range(batch_size)])
        for i in range(batch_size):
            result_big[i] = temp[i]
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small
