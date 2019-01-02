import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from krahenbuhl2013 import CRF
from skimage.transform import resize
from multiprocessing import Pool
import os
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

class SEC_NN(nn.Module):
    def __init__(self):
        super(SEC_NN, self).__init__()
        self.features = nn.Sequential( # the Sequential name has to be 'vgg feature'. the params name will be like feature.0.weight ,
        nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2),(1, 1),ceil_mode=True),
        nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2),(1, 1),ceil_mode=True),
        nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2),(1, 1),ceil_mode=True),
        nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(1, 1),(1, 1),ceil_mode=True),
        nn.Conv2d(512,512,(3, 3),padding=2, dilation=2),
        nn.ReLU(),
        nn.Conv2d(512,512,(3, 3),padding=2, dilation=2),
        nn.ReLU(),
        nn.Conv2d(512,512,(3, 3),padding=2, dilation=2),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(1, 1),(1, 1),ceil_mode=True),
        nn.AvgPool2d((3, 3),(1, 1),(1, 1),ceil_mode=True),#AvgPool2d,
        nn.Conv2d(512,1024,(3, 3),padding =12, dilation=12),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(1024,1024,(1, 1)),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(1024,21,(1, 1)) # 1024 / 512
        # nn.Softmax2d()
        )

        self.softmax2d = nn.Softmax2d()
        self.min_prob = 0.0001
        # self.mask2pre = nn.AdaptiveAvgPool2d(1)
        # self.mask2pre = nn.AdaptiveMaxPool2d(1)
        self.mask2pre = nn.Sequential(
            nn.MaxPool2d(5, stride=2),
            nn.AdaptiveAvgPool2d(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        fc_mask = self.features(x)
        sm_mask = self.softmax2d(fc_mask)+self.min_prob
        sm_mask = sm_mask / sm_mask.sum(dim=1, keepdim=True)
        # sm_mask = self.softmax2d(sm_mask)
        preds = self.mask2pre(sm_mask)

        return sm_mask, preds


class SeedingLoss(nn.Module):

    def __init__(self):
        super(SeedingLoss, self).__init__()
        self.thr = 0.5
        self.mask_size = [41, 41]

    def forward(self, sm_mask, attention_mask, labels, super_pixel, flag_use_cuda):
        # if super_pixel has too few unique elements, cues = 0
        # RuntimeError: unique is currently CPU-only, and lacks CUDA support. Pull requests welcome!
        cues = torch.from_numpy(np.zeros(sm_mask.shape).astype('float32'))

        for i_batch in range(sm_mask.shape[0]):
            if len(super_pixel[i_batch].unique()) > 10:
                cues[i_batch] = torch.from_numpy(resize(attention_mask[i_batch].permute([1,2,0]).numpy(), self.mask_size, mode='constant')).permute([2,0,1])

        if flag_use_cuda:
            cues = cues.cuda()

        thr_value = cues.max()*self.thr
        cues[cues < thr_value] = 0
        cues[cues >= thr_value] = 1.0  # hard cues

        count = len(cues.nonzero())
        max_val = cues.max()
        if max_val > 0:
            cues = cues/max_val

        loss = -50*(cues * sm_mask.log()).sum()/count
        return loss

#----------------------------------------------------------------------------------------------------------------
def crf(sm_mask_one, img_one, num_class, input_size, mask_size, num_iter):
    sm_u = np.transpose(resize(np.transpose(sm_mask_one, [1,2,0]), input_size, mode='constant'),[2,0,1])

    U = unary_from_softmax(sm_u)

    d = dcrf.DenseCRF2D(input_size[0], input_size[1], num_class)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(num_iter)
    return np.array(Q).reshape((num_class, input_size[0], input_size[1]))



class CRFLayer():
    def __init__(self, flag_multi_process = False):
        self.num_class = 21
        self.min_prob = 0.0001
        self.mask_size = [41, 41]
        self.input_size = [321, 321]
        self.num_iter = 5
        if flag_multi_process:
            num_cores = os.cpu_count()
            self.pool = Pool(processes=num_cores)


    def run(self, sm_mask, img): # the input array are detached numpy already
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)
        for i in range(batch_size):

            sm_u = np.transpose(resize(np.transpose(sm_mask[i], [1,2,0]), self.input_size, mode='constant'),[2,0,1])

            U = unary_from_softmax(sm_u)

            d = dcrf.DenseCRF2D(self.input_size[0], self.input_size[1], self.num_class)
            d.setUnaryEnergy(U)

            d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

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


class ConstrainLossLayer(nn.Module):
    def __init__(self):
        super(ConstrainLossLayer, self).__init__()
        self.num_pixel = 41 * 41

    def forward(self, crf_sm_mask, sm_mask, flag_use_cuda):
        temp = torch.from_numpy(crf_sm_mask.astype('float32'))
        if flag_use_cuda:
            temp = temp.cuda()
        return ((temp * (temp/sm_mask).log()).sum()/self.num_pixel)/sm_mask.shape[0]
