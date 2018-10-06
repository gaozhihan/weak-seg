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
from scipy.signal import medfilt2d
import krahenbuhl2013
import scipy.ndimage as nd

class CRF():
    def __init__(self, args):
        self.flag_visual = False
        self.iters = [0, 3, 10, 20]
        self.H , self.W = args.input_size
        self.N_labels = args.num_classes
        self.train_flag = not args.test_flag
        self.color_vote = args.color_vote
        self.fix_CRF_itr = args.fix_CRF_itr
        if self.fix_CRF_itr:
            self.iters = [0, 10]

        self.num_maps = len(self.iters)
        self.flag_pre_method = 1

        # parameters for pick_mask (based on color hist and overlap with mask)
        self.color_his_size = [4, 4, 4]
        self.num_color_bins = self.color_his_size[0]*self.color_his_size[1]*self.color_his_size[2]
        self.color_channels = [0, 1, 2]
        self.color_ranges = [0, 255, 0, 255, 0, 255]
        self.num_pixel = self.H * self.W
        self.color_score_scale = 1.5
        self.color_score_thr = self.H * self.W * 0.04


    def set_shape(self, mask_gt):
        self.H, self.W = mask_gt.shape
        self.num_pixel = self.H * self.W
        self.color_score_thr = self.H * self.W * 0.04


    def preprocess_mask(self, mask, preds):

        if self.flag_pre_method == 1: # problem: 1. since no use preds, other class appear; 2. other activation distort the present class
            mask[0,:,:] -= np.max(mask[1:,:,:], axis=0)  # get background

            # softmax
            temp_exp = np.exp(mask)
            mask = temp_exp / np.sum(temp_exp,axis=1,keepdims=True)

        return mask


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    normalization methods
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def spacial_norm_preds_only(self, mask, class_cur):
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(class_cur)
        temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])
        # temp_min = np.min(temp_cur, axis=1, keepdims=True)
        # temp_cur = temp_cur - temp_min
        temp_cur[temp_cur<0] = 0    # manual relu
        temp_max = np.max(temp_cur, axis=1, keepdims=True)
        temp_max[temp_max == 0] = 1
        temp_cur = temp_cur / temp_max

        if class_cur[0] == 0 and num_class_cur > 1:
            # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
            temp_cur[0,np.sum(temp_cur[1:,:], axis=0)>0.1] = 0

        temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])
        temp = temp * 0.9 + 0.05

        return temp


    def sig_pred_only(self, mask, class_cur): # can use without relu
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(class_cur)
        temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])

        # temp_cur[temp_cur>80] = 80 # in case overflow
        temp_cur[temp_cur<-80] = -80
        # temp_cur = temp_cur * 0.2

        # scale the positive part
        idx_temp = temp_cur > 0
        temp_cur[idx_temp] = (temp_cur[idx_temp] / temp_cur[idx_temp].max() - 0.2) * 10

        temp_cur = 1/(1+np.exp(-temp_cur))

        if class_cur[0] == 0 and num_class_cur > 1:
            # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
            temp_cur[0,np.max(temp_cur[1:,:], axis=0)>0.5] = 0.005

        temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])
        temp = temp * 0.9 + 0.05

        return temp


    def softmax_norm_preds_only(self, mask, class_cur):  # so far looks good
        # mask[mask<0] = 0
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(class_cur)
        temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])

        # caution: np.finfo(np.float32).max = 3.4028235e+38, and np.exp(90) = 1.2204032943178408e+39, np.exp(80) = 5.54062238439351e+34
        temp_cur[temp_cur>80] = 80
        # temp_cur = temp_cur * 0.2
        # temp_cur = temp_cur/temp_cur.max()*10

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


    def channel_norm(self, mask):  # the same as SEC: https://github.com/kolesman/SEC/blob/master/deploy/demo.py
        mask_exp = np.exp(mask - np.max(mask, axis=0, keepdims=True))
        mask = mask_exp / np.sum(mask_exp, axis=0, keepdims=True)
        eps = 0.00001
        mask[mask < eps] = eps

        return mask


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    mask generation
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def multi_iter_CRF(self, mask_res, img):
        U = np.log(np.transpose(mask_res,(1,2,0)))
        img_uint8 = img.astype('uint8')
        map = np.zeros([self.num_maps, img.shape[0], img.shape[1]])

        i_map = 0
        for idx, val in enumerate(self.iters):
            map[idx,:,:] = np.argmax(krahenbuhl2013.CRF(img_uint8, U, maxiter=val, scale_factor=1.0), axis=2)

        if self.iters[0] == 0:
            raw_map = map[0,:,:]
        return map


    def color_mask_vote(self, mask, img, class_cur):
        num_class_cur = len(class_cur)
        img_quantize = (img/(256/self.color_his_size[0])).astype(np.uint8)
        mask_cur = mask[class_cur,:,:]
        hist_cur = np.zeros([num_class_cur, self.color_his_size[0], self.color_his_size[1], self.color_his_size[2]])
        hist_score = np.zeros([num_class_cur, self.color_his_size[0], self.color_his_size[1], self.color_his_size[2]])
        hist_whole = cv2.calcHist([img], self.color_channels, None, self.color_his_size, self.color_ranges)
        hist_whole_no_zeros = hist_whole.copy() # for division
        hist_whole_no_zeros[hist_whole_no_zeros==0] = 1

        for i_idx, i_class in np.ndenumerate(class_cur):
            idx_except = np.ones(num_class_cur,dtype=np.bool)
            idx_except[i_idx] = False
            mask_max_except = np.max(mask_cur[idx_except,:,:],axis=0)
            # mask_pos = mask_cur[i_idx,:,:] > 0.05 # determined by normalization step
            if i_class == 0:
                cur_region_mask = (mask[i_class,:,:]>0.1).astype(np.uint8)
            else:
                cur_region_mask = (mask[i_class,:,:]>0.2).astype(np.uint8)
            hist_cur[i_idx,:,:,:] = cv2.calcHist([img], self.color_channels, cur_region_mask, self.color_his_size, self.color_ranges)
            hist_score[i_idx,:,:,:] = np.minimum(hist_cur[i_idx,:,:,:],hist_whole)/hist_whole_no_zeros

            if i_class == 0:
                select_bin = hist_score[i_idx,:,:,:].squeeze()>0.6
            else:
                select_bin = hist_score[i_idx,:,:,:].squeeze()>0.5

            select_color_idx = np.asarray(np.nonzero(select_bin))
            select_pix_idx = np.full((img.shape[0], img.shape[1]), False, dtype=bool)
            for i_color in range(select_color_idx.shape[1]):
                temp0 = (img_quantize[:,:,0] == select_color_idx[0,i_color])
                temp1 = (img_quantize[:,:,1] == select_color_idx[1,i_color])
                temp2 = (img_quantize[:,:,2] == select_color_idx[2,i_color])

                temp = np.logical_and(np.logical_and(temp0,temp1),temp2)
                select_pix_idx = np.logical_or(select_pix_idx, temp)

            # process (refine) the mask e.g. mark selected color as confident to be this class
            select_pix_idx = np.logical_and(select_pix_idx,mask_max_except<0.22)
            select_pix_idx = np.logical_and(select_pix_idx, mask_cur[i_idx,:,:].squeeze()>0.05)
            if i_class == 0:
                mask[i_class,select_pix_idx] = 0.65 #(0.85 - (np.sum(mask_cur, axis=0) - mask_cur[i_idx,:,:])).squeeze()[select_pix_idx] # confident this class
            else:
                mask[i_class,select_pix_idx] = 0.85

        return mask


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    mask selection
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pick_mask(self, image, mask, class_cur, num_pixel_org_img, H_img, W_img, map):
        # should be pick_mask(self, maps, mask, preds), since within class function, so save any self. items
        mask_weight = mask
        num_class_cur = len(class_cur)
        score_color = np.zeros(self.num_maps)
        score_over_map = np.zeros(self.num_maps)
        score_map_iou = np.zeros(self.num_maps) # use to weight the confidence
        raw_map = np.argmax(mask, axis=0)

        # generate whole map sum score for use
        score_whole = np.zeros(num_class_cur)
        for i_idx, i_class in np.ndenumerate(class_cur):
            # score_whole[i_idx] = mask[i_class,:,:].sum()
            score_whole[i_idx] = (raw_map == i_class).sum()

        score_whole[score_whole==0] = 1 # just in case


        for i_map in range(self.num_maps):
            hist_cur = np.zeros([num_class_cur, self.color_his_size[0], self.color_his_size[1], self.color_his_size[2]])
            temp_map_overlap_soft = np.zeros(num_class_cur)
            temp_map_overlap_hard = np.zeros(num_class_cur)
            pre_map_whole = np.zeros(num_class_cur)
            for i_idx, i_class in np.ndenumerate(class_cur):
                mask_temp = np.zeros([H_img, W_img],dtype = np.uint8)
                idx_temp = map[i_map,:,:] == i_class
                # calculate score based on color histogram separation
                mask_temp[idx_temp] = 1
                if mask_temp.sum() < 100: # based on my observation
                    hist_cur[i_idx[0], :,:,:] = num_pixel_org_img
                else:
                    hist_cur[i_idx[0], :,:,:] = cv2.calcHist([image], self.color_channels, mask_temp, self.color_his_size, self.color_ranges)

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

        iou_penalty = np.maximum(0.5 - score_map_iou, 0) * 100000
        score_overall = score_over_map - score_color - iou_penalty
        best_map_idx = np.argmax(score_overall)
        # print(score_color)
        # print(score_map_iou)
        # print(score_overall - score_overall.min())
        # print(iou_penalty)

        return best_map_idx, score_map_iou[best_map_idx], score_color[best_map_idx]



    def choose_and_weigh(self, map_iou_score, color_score):
        idx = np.argmin(color_score,axis=0)

        if map_iou_score[idx] < 0.2:
            return idx, 0.0
        else:
            return idx, np.maximum((self.color_score_thr - color_score[idx]), 0)/self.color_score_thr



    def map2mask(self, mask_org, class_cur, map_best):
        map_s_gt = np.zeros(mask_org.shape)
        for i_class in class_cur:
            temp_map = np.zeros(map_best.shape)
            temp_map[map_best==i_class] = 1
            # map_s_gt[i_class,:,:] = resize(temp_map, (mask_org.shape[1], mask_org.shape[2]), mode='constant', anti_aliasing=True)
            map_s_gt[i_class,:,:] = resize(temp_map, (mask_org.shape[1], mask_org.shape[2]), mode='constant')

        return map_s_gt


    def runCRF(self, labels, mask_gt, mask_org, img, preds, preds_only ):  # run CRF on one frame, all input are numpy
        # set shape, so CRF can run in original image shape
        H_img, W_img = mask_gt.shape
        map = np.zeros([self.num_maps, H_img, W_img])
        num_pixel_org_img = H_img * W_img
        color_score_thr = H_img * W_img * 0.04

        if self.flag_visual:
            mask_gt[mask_gt==255] = 0
            plt.close("all")

        mask_res = np.zeros((self.N_labels, H_img, W_img))
        # class_cur = np.nonzero(preds)[0]
        class_cur = np.nonzero(labels)[0]

        mask_exp = np.exp(mask_org - np.max(mask_org, axis=0, keepdims=True))
        mask = mask_exp / np.sum(mask_exp, axis=0, keepdims=True)


        if len(class_cur) == 1:
            pre_mask = np.full((H_img, W_img), class_cur[0], dtype=np.float64)
            confidence = 0.0

            if self.train_flag:
                return self.map2mask(mask_org, class_cur, pre_mask), pre_mask, confidence, (mask - 0.05)/0.9
            else:
                return self.map2mask(mask_org, class_cur, pre_mask), pre_mask

        eps = 0.00001
        for i in range(self.N_labels):
            #temp = resize(mask[i,:,:], (H_img, W_img), mode='constant', anti_aliasing=True)
            temp = resize(mask[i,:,:], (H_img, W_img), mode='constant')
            temp[temp < eps] = eps
            mask_res[i,:,:] = temp

        #pre_mask = np.argmax(krahenbuhl2013.CRF(img.astype('uint8'), np.log(np.transpose(mask_res,(1,2,0))), scale_factor=1.0), axis=2)
        map = self.multi_iter_CRF(mask_res, img)
        confidence = 0.0

        if self.flag_visual:
            self.num_plot = len(self.iters)
            plt.figure(figsize=((3 + self.num_maps)*5,5))

            plt.subplot(1,(3 + self.num_maps),1); plt.imshow(img/255); plt.title('Input image')
            plt.subplot(1,(3 + self.num_maps),2); plt.imshow(mask_gt); plt.title('true mask')
            plt.subplot(1,(3 + self.num_maps),3); plt.imshow(map[0,:,:]); plt.title('raw mask')

            for i in range(3,(3 + self.num_maps)):
                plt.subplot(1,(2 + self.num_maps),i); plt.imshow(map[i-3,:,:]); plt.title('{} steps'.format(self.iters[i-3])); plt.axis('off')

        best_map_idx, map_iou_score, color_score = self.pick_mask(img, mask_res, class_cur, num_pixel_org_img, H_img, W_img, map)
        best_maps = np.expand_dims(map[best_map_idx,:,:], axis=0)
        best_maps_iou = np.expand_dims(map_iou_score, axis=0)
        best_color_scores = np.expand_dims(color_score, axis=0)

        if self.fix_CRF_itr:
            best_map_idx = 1

        # -----------------------------if self.color_vote = True, no need go further--------------------------------
        if not self.color_vote:
            pre_mask = best_maps[0,:,:]
            pre_mask = medfilt2d(pre_mask, kernel_size=5)
            if map_iou_score < 0.2:
                confidence = 0.0
            else:
                confidence = np.maximum((color_score_thr - color_score), 0)/color_score_thr

            if self.flag_visual:
                plt.figure()
                plt.imshow(pre_mask)

            if self.train_flag:
                return self.map2mask(mask_org, class_cur, pre_mask), pre_mask, confidence
            else:
                return self.map2mask(mask_org, class_cur, pre_mask), pre_mask


        # -----------------------------start color vote ----------------------------------------------------------------
        mask_res = self.color_mask_vote(mask_res, img, class_cur)
        map = self.multi_iter_CRF(mask_res, img)

        if self.flag_visual:
            self.num_plot = len(self.iters)
            plt.figure(figsize=((3 + self.num_maps)*5,5))

            plt.subplot(1,(3 + self.num_maps),1); plt.imshow(img/255); plt.title('Input image')
            plt.subplot(1,(3 + self.num_maps),2); plt.imshow(mask_gt); plt.title('true mask')
            plt.subplot(1,(3 + self.num_maps),3); plt.imshow(map[0,:,:]); plt.title('raw mask')

            for i in range(3,(3 + self.num_maps)):
                plt.subplot(1,(2 + self.num_maps),i); plt.imshow(map[i-3,:,:]); plt.title('{} steps'.format(self.iters[i-3])); plt.axis('off')

        best_map_idx, map_iou_score, color_score = self.pick_mask(img, mask_res, class_cur, num_pixel_org_img, H_img, W_img, map)
        best_maps = np.concatenate((best_maps,np.expand_dims(map[best_map_idx,:,:], axis=0)), axis=0)
        best_maps_iou = np.concatenate((best_maps_iou,np.expand_dims(map_iou_score, axis=0)), axis=0)
        best_color_scores = np.concatenate((best_color_scores,np.expand_dims(color_score, axis=0)), axis=0)

        idx_the_best, confidence = self.choose_and_weigh(best_maps_iou, best_color_scores)

        if self.fix_CRF_itr:
            best_map_idx = 1

        pre_mask = best_maps[idx_the_best,:,:]
        #pre_mask = medfilt2d(pre_mask, kernel_size=5)

        if self.flag_visual:
            plt.figure()
            plt.imshow(pre_mask)

        if self.train_flag:
            return self.map2mask(mask_org, class_cur, pre_mask), pre_mask, confidence, (mask - 0.05)/0.9
        else:
            return self.map2mask(mask_org, class_cur, pre_mask), pre_mask




