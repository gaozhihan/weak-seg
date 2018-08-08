#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Wednesday, May 16, 2018 PM11:50:32 HKT
Info:
'''
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import pydensecrf.densecrf as dcrf
import torch
import numpy as np
import matplotlib.pyplot as plt

def crf_processing(u, img):
    '''
    # u : np array, size(channel, hight*width)
         initial unary term, normalize to one for every point cross channel
    # img : np array, size(height, weight, 3)
         range from (0,255)
    '''

    # -----------------------------------------------------------
    W = img.shape[1]
    H = img.shape[0]
    channel = u.shape[0]

    # ---------------------------------------------------------
    iters = [1, 3, 10, 15]
    max_num_iters = len(iters)
    kl = np.zeros(max_num_iters)
    map = np.zeros([max_num_iters, H, W])
    flag_visual = True
    # -----------------------------------------------------------

    d = dcrf.DenseCRF2D(W, H, channel)
    d.setUnaryEnergy(unary_from_softmax(u))

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=(args.sxy, args.sxy), srgb=(args.srgb,args.srgb,args.srgb), rgbim=img.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(30, 30), srgb=(7,7,7), rgbim=img.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q, tmp1, tmp2 = d.startInference()
    for i in range(iters[-1]):
        d.stepInference(Q, tmp1, tmp2)

        for ii in range(max_num_iters):
            if i == iters[ii]:
                kl[ii] = d.klDivergence(Q) / (H*W)
                map[ii,:,:] = np.argmax(Q, axis=0).reshape((H,W))


    if flag_visual:
        num_plot = len(iters)
        plt.figure(figsize=((1 + max_num_iters)*5,5))

        plt.subplot(1,(1 + max_num_iters),1); plt.imshow(img/255); plt.title('Input image')


        for i in range(2,(1 + max_num_iters)):
            plt.subplot(1,(1 + max_num_iters),i); plt.imshow(map[i-2,:,:]); plt.title('{} steps, KL={:.2f}'.format(iters[i-2], kl[i-2])); plt.axis('off');

        print('done')

    return map[max_num_iters-1,:,:].reshape(H*W)



def pow_norm(input_, thr_root, k_root):
    temp = input_ / thr_root
    temp = torch.min(torch.pow(temp, k_root), temp)
    return temp

#def sig_norm(self, input_):
    #temp = self.normalize_to_01(input_)
    #temp = temp * prob_scale - prob_scale * self.thr_sig
    #return F.sigmoid(temp)

def fast_hist(a, b, n):
    #b = b*(b!=255)
    k = ((a >= 0) & (a < n)) & (b!=255)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)





#====================================================================================
eps=0.000001

def normalize_to_01(input_):
    #min_value = input.min()
    #temp = input_ - min_value
    temp = input_ - np.min(np.min(input_, axis=1, keepdims=True), axis=2, keepdims=True)
    max_value = np.max(np.max(temp, axis=1, keepdims=True), axis=2, keepdims=True)
    max_value[max_value<eps]=eps
    return (temp/max_value)

def sig_norm(input, thr_sig=0.1, prob_scale=10):
    #thr_sig = 0.1
    #prob_scale = 10
    #temp = (normalize_to_01(input) - thr_sig) * prob_scale
    temp = (input - thr_sig) * prob_scale
    temp = 1/(1+np.exp(-temp))
    return temp

def pow_norm(input, thr_pow=0.1, pow=0.1):
    #thr_pow = 0.1
    #pow = 0.1
    #temp = normalize_to_01(input)
    temp = input
    temp = temp / thr_pow
    temp = np.minimum(temp,np.power(temp,pow))
    #return normalize_to_01(temp)
    return temp


def sig_norm_reserve_range(input_):
    # for SEC, thr_sig = 0.5, prob_scale = 8
    thr_sig = 0.1
    prob_scale = 6
    input_shape = input_.shape
    min_value = np.min(np.min(input_, axis=1, keepdims=True), axis=2, keepdims=True)
    max_value = np.max(np.max(input_, axis=1, keepdims=True), axis=2, keepdims=True)
    #range = input.max() - min_value
    range = max_value-min_value
    temp = (normalize_to_01(input_) - thr_sig) * prob_scale
    temp = 1/(1+np.exp(-temp))
    temp = normalize_to_01(temp) * range + min_value
    return temp

def pow_norm_reserve_range(input_):
    # for SEC, thr_pow = 0.5, pow = 0.3
    thr_pow = 0.3
    pow = 0.5
    #min_value = input.min()
    min_value = np.min(np.min(input_, axis=1, keepdims=True), axis=2, keepdims=True)
    max_value = np.max(np.max(input_, axis=1, keepdims=True), axis=2, keepdims=True)
    #range = input.max() - min_value
    range = max_value-min_value
    #range = input.max() - min_value
    temp = normalize_to_01(input_)
    temp = temp / thr_pow
    temp = np.minimum(temp,np.power(temp,pow))
    temp = normalize_to_01(temp) * range + min_value
    return temp

def norm_channel_sum_to_01(input_):
    input_sum = np.sum(input_, axis=0, keepdims=True)
    input_sum[input_sum<eps]=eps
    return input_/input_sum

def generate_u(attention_map, preds=None):
    """
    input:
        attention_map:      (numpy.ndarray)
                            size(channel, height, weight)
        preds:              (numpy.ndarray)
                            size(channel,) a 21 channel vector like (0,0,1,0....)

    output:
        U: (numpy.ndarray)
            size(channel, height, weight)
    """

    if preds is None:
        U = attention_map
        if args.flag_pow:
            U = pow_norm_reserve_range(U)
            #for i in range(attention_map.shape[0]):
                #U[i,:,:] = pow_norm_reserve_range(U[i,:,:])

        if args.flag_sig:
            U = sig_norm_reserve_range(U)
            #for i in range(attention_map.shape[0]):

    else:
        attention_map = attention_map * (attention_map >= 0)
        idx = np.squeeze(np.asarray(np.nonzero(preds)),0)
        #U = np.zeros([len(idx),attention_map.shape[1],attention_map.shape[2]])
        selected_u = attention_map[idx, :, :]
        #for i in range(len(idx)):
            #U[i,:,:] = normalize_to_01(attention_map[idx[i],:,:])
        U=normalize_to_01(selected_u)

        #for i in range(len(idx)):
        if args.flag_pow:
            U = normalize_to_01(pow_norm(U, args.thr_pow, args.pow))

        if args.flag_sig:
            U = sig_norm(U, args.thr_sig, args.prob_scale)

        if idx[0] == 0:
            temp = U[1:,:,:].sum(0)
            temp = normalize_to_01(temp)
            U[0,:,:] = 1 - temp

        U = U * 0.9 + 0.05
        U = norm_channel_sum_to_01(U)

    return U
