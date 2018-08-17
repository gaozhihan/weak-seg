'''
PyDenseCRF:
https://github.com/lucasb-eyer/pydensecrf

reference of opencv color histogram:
https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
https://docs.opencv.org/3.3.1/dd/d0d/tutorial_py_2d_histogram.html

'''

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2


class CRF():
    def __init__(self, args):
        self.flag_visual = False
        self.iters = [1, 3, 10, 15, 25]
        self.H , self.W = args.input_size
        self.N_labels = args.num_classes

        self.num_maps = len(self.iters)
        self.flag_pre_method = 1

        # parameters for pick_mask (based on color hist and overlap with mask)
        self.color_his_size = [32, 32, 32]
        self.color_channels = [0, 1, 2]
        self.color_ranges = [0, 256, 0, 256, 0, 256]
        self.num_pixel = self.H * self.W
        self.color_score_scale = 1.5

    def set_shape(self, mask_gt):
        self.H, self.W = mask_gt.shape
        self.map = np.zeros([self.num_maps, self.H, self.W])
        self.num_pixel = self.H * self.W


    def preprocess_mask(self, mask, preds):

        if self.flag_pre_method == 1: # problem: 1. since no use preds, other class appear; 2. other activation distort the present class
            mask[0,:,:] -= np.max(mask[1:,:,:], axis=0)  # get background

            # softmax
            temp_exp = np.exp(mask)
            mask = temp_exp / np.sum(temp_exp,axis=1,keepdims=True)

        return mask


    def spacial_norm_preds_only(self, mask, class_cur):
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(class_cur)
        temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])
        temp_min = np.min(temp_cur, axis=1, keepdims=True)
        temp_cur = temp_cur - temp_min
        temp_max = np.max(temp_cur, axis=1, keepdims=True)
        temp_max[temp_max == 0] = 1
        temp_cur = temp_cur / temp_max

        if class_cur[0] == 0 and num_class_cur > 1:
            # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
            temp_cur[0,np.sum(temp_cur[1:,:], axis=0)>0] = 0

        temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])
        temp = temp * 0.9 + 0.05

        return temp



    def spacial_norm_sig_pred_only(self, mask, class_cur): # can use without relu
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(class_cur)
        temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])
        # temp_cur[temp_cur>80] = 80 # in case overflow
        temp_cur[temp_cur<-80] = -80
        # temp_cur = temp_cur * 0.2
        temp_cur = 1/(1+np.exp(-temp_cur))

        if class_cur[0] == 0 and num_class_cur > 1:
            # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
            temp_cur[0,np.max(temp_cur[1:,:], axis=0)>0.5] = 0.005

        temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])
        temp = temp * 0.9

        return temp



    def softmax_norm_preds_only(self, mask, class_cur):  # so far looks good
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(class_cur)
        temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])

        # caution: np.finfo(np.float32).max = 3.4028235e+38, and np.exp(90) = 1.2204032943178408e+39, np.exp(80) = 5.54062238439351e+34
        temp_cur[temp_cur>80] = 80

        if class_cur[0] == 0 and num_class_cur > 1:
            # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
            temp_cur[0,np.max(temp_cur[1:,:], axis=0)>0] = 0

        temp_cur = np.exp(temp_cur) / np.sum(np.exp(temp_cur), axis=0)

        temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])

        return temp


    def spacial_norm(self, mask):
        temp_f = mask[1:,:,:].reshape([self.N_labels-1,-1])
        min_f = np.min(temp_f, axis=1, keepdims=True)
        temp_f = temp_f - min_f
        max_f = np.max(temp_f, axis=1, keepdims=True)
        max_f[max_f==0] = 1
        temp_f = temp_f/max_f
        temp_b = 1 - np.sum(temp_f, axis=0)
        temp_b[temp_b<0] = 0

        mask[1:,:,:] = temp_f.reshape([temp_f.shape[0],mask.shape[1], mask.shape[2]])
        mask[0,:,:] = temp_b.reshape([mask.shape[1], mask.shape[2]])

        mask = mask

        return mask * 0.9 + 0.05


    def pick_mask(self, image, mask, class_cur):
        # should be pick_mask(self, maps, mask, preds), since within class function, so save any self. items
        num_class_cur = len(class_cur)
        score_color = np.zeros(self.num_maps)
        score_over_map = np.zeros(self.num_maps)
        for i_map in range(self.num_maps):
            hist_cur = np.zeros([num_class_cur, self.color_his_size[0], self.color_his_size[1], self.color_his_size[2]])
            for i_idx, i_class in np.ndenumerate(class_cur):
                mask_temp = np.zeros([self.H, self.W],dtype = np.uint8)
                idx_temp = self.map[i_map,:,:] == i_class
                # calculate score based on color histogram separation
                mask_temp[idx_temp] = 1
                if mask_temp.sum() < 50: # based on my observation
                    hist_cur[i_idx[0], :,:,:] = self.num_pixel
                else:
                    hist_cur[i_idx[0], :,:,:] = cv2.calcHist([image], self.color_channels, mask_temp, self.color_his_size, self.color_ranges)

                # calculate score based on consisitencey (overlap with raw mask)
                if i_class !=  0:
                    score_over_map[i_map] += np.multiply(mask_temp, mask[i_class,:,:]).sum()

            # summary
            hist_cur = hist_cur.reshape([len(class_cur), -1])
            score_color[i_map] = np.min(hist_cur, axis=0).sum()

        return np.argmax(score_over_map - score_color*self.color_score_scale)


    def map2mask(self, mask_org, class_cur, map_best):
        map_s_gt = np.zeros(mask_org.shape)
        for i_class in class_cur:
            temp_map = np.zeros((self.H, self.W))
            temp_map[map_best==i_class] = 1
            map_s_gt[i_class,:,:] = resize(temp_map, (mask_org.shape[1], mask_org.shape[2]), mode='constant', anti_aliasing=True)

        return map_s_gt



    def runCRF(self, labels, mask_gt, mask_org, img, preds, preds_only ):  # run CRF on one frame, all input are numpy
        if self.flag_visual:
            mask_gt[mask_gt==255] = 0

        self.kl = np.zeros(self.num_maps)
        self.map = np.zeros([self.num_maps, self.H, self.W])
        mask_res = np.zeros((self.N_labels, self.H, self.W))
        # class_cur = np.nonzero(preds)[0]
        class_cur = np.nonzero(labels)[0]
        if preds_only:
            # mask = self.spacial_norm_preds_only(mask_org, class_cur)
            # mask = self.softmax_norm_preds_only(mask_org, class_cur)
            mask = self.spacial_norm_sig_pred_only(mask_org, class_cur)

        else:
            mask = self.spacial_norm(mask_org)

        for i in range(self.N_labels):
            mask_res[i,:,:] = resize(mask[i,:,:], (self.H, self.W), mode='constant', anti_aliasing=True)


        U = unary_from_softmax(mask_res)

        d = dcrf.DenseCRF2D(self.W, self.H, self.N_labels)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)


        Q, tmp1, tmp2 = d.startInference()
        raw_map = np.argmax(Q, axis=0).reshape((self.H,self.W))
        for i in range(self.iters[-1]):
            d.stepInference(Q, tmp1, tmp2)

            for ii in range(self.num_maps):
                if i+1 == self.iters[ii]:
                    self.kl[ii] = d.klDivergence(Q) / (self.H*self.W)
                    self.map[ii,:,:] = np.argmax(Q, axis=0).reshape((self.H,self.W))


        if self.flag_visual:
            self.num_plot = len(self.iters)
            plt.figure(figsize=((3 + self.num_maps)*5,5))

            plt.subplot(1,(3 + self.num_maps),1); plt.imshow(img/255); plt.title('Input image')
            plt.subplot(1,(3 + self.num_maps),2); plt.imshow(mask_gt); plt.title('true mask')
            plt.subplot(1,(3 + self.num_maps),3); plt.imshow(raw_map); plt.title('raw mask')

            for i in range(4,(4 + self.num_maps)):
                plt.subplot(1,(3 + self.num_maps),i); plt.imshow(self.map[i-4,:,:]); plt.title('{} steps, KL={:.2f}'.format(self.iters[i-4], self.kl[i-4])); plt.axis('off')

        best_map_idx = self.pick_mask(img, mask_res, class_cur)
        return (self.map2mask(mask_org, class_cur, self.map[best_map_idx,:,:]), self.map[best_map_idx,:,:])




