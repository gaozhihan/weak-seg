import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import matplotlib.pyplot as plt
from skimage.transform import resize


class CRF():
    def __init__(self, args):
        self.flag_visual = True
        self.iters = [1, 3, 10, 15]
        self.H , self.W = args.input_size
        self.N_labels = args.num_classes

        self.max_num_iters = len(self.iters)
        self.kl = np.zeros(self.max_num_iters)
        self.map = np.zeros([self.max_num_iters, self.H, self.W])
        self.flag_pre_method = 1


    def preprocess_mask(self, mask, preds):

        if self.flag_pre_method == 1: # problem: 1. since no use preds, other class appear; 2. other activation distort the present class
            mask[0,:,:] -= np.max(mask[1:,:,:], axis=0)  # get background

            # softmax
            temp_exp = np.exp(mask)
            mask = temp_exp / np.sum(temp_exp,axis=1,keepdims=True)

        # temp_f = mask[1:,:,:].reshape([self.N_labels-1,-1])
        # min_f = np.min(temp_f, axis=1, keepdims=True)
        # temp_f = temp_f - min_f
        # max_f = np.max(temp_f, axis=1, keepdims=True)
        # max_f[max_f==0] = 1
        # temp_f = temp_f/max_f
        # temp_b = 1 - np.sum(temp_f, axis=0)
        # temp_b[temp_b<0] = 0
        #
        # mask[1:,:,:] = temp_f.reshape([temp_f.shape[0],mask.shape[1], mask.shape[2]])
        # mask[0,:,:] = temp_b.reshape([mask.shape[1], mask.shape[2]])

        return mask


    def spacial_norm_preds_only(self, mask, preds_cur):
        temp = np.zeros(mask.shape)
        # spactial normalize
        num_class_cur = len(preds_cur)
        temp_cur = mask[preds_cur,:,:].reshape([num_class_cur, -1])
        temp_min = np.min(temp_cur, axis=1, keepdims=True)
        temp_cur = temp_cur - temp_min
        temp_max = np.max(temp_cur, axis=1, keepdims=True)
        temp_max[temp_max == 0] = 1
        temp_cur = temp_cur / temp_max

        if preds_cur[0] == 0 and num_class_cur > 1:
            temp_cur[0,:] = 1 - np.sum(temp_cur[1:,:], axis=0)
            temp_cur[0,temp_cur[0,:]<0] = 0

        temp[preds_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])
        temp = temp * 0.9 + 0.05

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



    def runCRF(self, labels, mask_gt, mask_org, img, preds, preds_only ):  # run CRF on one frame, all input are numpy

        mask_res = np.zeros((self.N_labels, self.H, self.W))
        if preds_only:
            preds_cur = np.nonzero(preds)[0]
            mask = self.spacial_norm_preds_only(mask_org, preds_cur)

        else:
            mask = self.spacial_norm(mask_org)

        for i in range(self.N_labels):
            mask_res[i,:,:] = resize(mask[i,:,:], (self.H, self.W))


        U = unary_from_softmax(mask_res)

        d = dcrf.DenseCRF2D(self.W, self.H, self.N_labels)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)


        Q, tmp1, tmp2 = d.startInference()
        for i in range(self.iters[-1]):
            d.stepInference(Q, tmp1, tmp2)

            for ii in range(self.max_num_iters):
                if i == self.iters[ii]:
                    self.kl[ii] = d.klDivergence(Q) / (self.H*self.W)
                    self.map[ii,:,:] = np.argmax(Q, axis=0).reshape((self.H,self.W))


        if self.flag_visual:
            self.num_plot = len(self.iters)
            plt.figure(figsize=((2 + self.max_num_iters)*5,5))

            plt.subplot(1,(2 + self.max_num_iters),1); plt.imshow(img/255); plt.title('Input image')
            plt.subplot(1,(2 + self.max_num_iters),2); plt.imshow(mask_gt); plt.title('true mask')

            for i in range(3,(3 + self.max_num_iters)):
                plt.subplot(1,(2 + self.max_num_iters),i); plt.imshow(self.map[i-3,:,:]); plt.title('{} steps, KL={:.2f}'.format(self.iters[i-3], self.kl[i-3])); plt.axis('off')

            print('done')






