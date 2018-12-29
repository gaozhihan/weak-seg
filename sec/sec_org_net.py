import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from krahenbuhl2013 import CRF
from skimage.transform import resize
from joblib import Parallel, delayed
from multiprocessing import Pool
import os

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

        return fc_mask, sm_mask


# ===========================================================================================
class CRFLayer():
    def __init__(self, flag_multi_process = False):
        self.num_class = 21
        self.min_prob = 0.0001
        self.mask_size = [41, 41]
        self.input_size = [321, 321]
        if flag_multi_process:
            num_cores = os.cpu_count()
            self.pool = Pool(processes=num_cores)


    def run(self, mask, img, flag_train): # flag_train is for the strange dif between train & test in org SEC
        batch_size = mask.shape[0]
        unary = np.transpose(mask, [0, 2, 3, 1])

        if flag_train:
            result = np.zeros(unary.shape)
            for i in range(batch_size):
                result[i] = CRF(np.round(resize(img[i]/255.0, self.mask_size, mode='constant')*255), unary[i], scale_factor=12.0)

            result = np.transpose(result, [0, 3, 1, 2])
            result[result < self.min_prob] = self.min_prob
            result = result / np.sum(result, axis=1, keepdims=True)

            return np.log(result)

        else:
            result = np.zeros([batch_size, self.input_size[0], self.input_size[1]])
            unary[unary < self.min_prob] = self.min_prob
            for i in range(batch_size):
                u_temp = resize(unary[i], self.input_size, mode='constant')
                result[i, :, :] = np.argmax(CRF(img[i], np.log(u_temp), scale_factor=1.0), axis=2)

            return result

# for parallel running ------------------------------
    def run_parallel(self, mask, img, flag_train): # flag_train is for the strange dif between train & test in org SEC
        batch_size = mask.shape[0]
        unary = np.transpose(mask, [0, 2, 3, 1])

        if flag_train:
            result = np.zeros(unary.shape)

            temp = self.pool.starmap(CRF,[(resize(img[i]/255.0, self.mask_size, mode='constant')*255, unary[i], 10, 12.0) for i in range(batch_size)])
            for i in range(batch_size):
                result[i] = temp[i]

            result = np.transpose(result, [0, 3, 1, 2])
            result[result < self.min_prob] = self.min_prob
            result = result / np.sum(result, axis=1, keepdims=True)

            return np.log(result)

        else:
            result = np.zeros([batch_size, self.input_size[0], self.input_size[1]])
            unary[unary < self.min_prob] = self.min_prob

            temp = self.pool.starmap(CRF,[(img[i], np.log(resize(unary[i], self.input_size, mode='constant')), 10, 1.0) for i in range(batch_size)])
            for i in range(batch_size):
                result[i, :, :] = np.argmax(temp[i], axis=2)

            return result




# SEC: seeding loss,  expansion loss,  constrain-to-boundary loss
class SeedingLoss(nn.Module):

    def __init__(self):
        super(SeedingLoss, self).__init__()

    def forward(self, sm_mask, cues):
        count = cues.sum()
        loss = -(cues * sm_mask.log()).sum()/count
        return loss


class ConstrainLossLayer(nn.Module):
    def __init__(self):
        super(ConstrainLossLayer, self).__init__()
        self.num_pixel = 41 * 41

    def forward(self, fc_crf_log, sm_mask, flag_use_cuda):
        temp = torch.from_numpy(fc_crf_log.astype('float32')).exp()
        if flag_use_cuda:
            temp = temp.cuda()
        return ((temp * (temp/sm_mask).log()).sum()/self.num_pixel)/sm_mask.shape[0]


class ExpandLossLayer(nn.Module):
    def __init__(self, flag_use_cuda):
        super(ExpandLossLayer, self).__init__()
        self.total_pixel_num = 41 * 41
        self.q_fg = 0.996
        self.w_fg = np.array([self.q_fg ** i for i in range(self.total_pixel_num)])
        self.z_fg = self.w_fg.sum()
        self.w_fg_norm = torch.from_numpy((self.w_fg/self.z_fg).astype('float32'))

        self.q_bg = 0.999
        self.w_bg = np.array([self.q_bg ** i for i in range(self.total_pixel_num)])
        self.z_bg = self.w_bg.sum()
        self.w_bg_norm = torch.from_numpy((self.w_bg/self.z_bg).astype('float32'))

        if flag_use_cuda:
            self.w_fg_norm = self.w_fg_norm.cuda()
            self.w_bg_norm = self.w_bg_norm.cuda()



    def forward(self, sm_mask, labels): # output prediction acc as by product
        batch_size = labels.shape[0]
        loss = 0.0
        for i_batch in range(batch_size):
            # for background
            # g_bg = (sm_mask[i_batch,0,:,:].reshape([self.total_pixel_num]).sort(descending=True)[0]*self.w_bg_norm).sum()
            # if labels[i_batch, 0]>0:
            #     loss -= g_bg.log()
            # else:
            #     loss -= (1 - g_bg).log()
            # # loss -= (labels[i_batch, 0]*g_bg.log() + (1-labels[i_batch, 0])*(1 - g_bg).log())
            # loss -= (sm_mask[i_batch,0,:,:].reshape([self.total_pixel_num]).sort(descending=True)[0]*self.w_bg_norm).sum().log()

            # for exist foreground
            loss_temp = 0.0
            for i_exi_class in labels[i_batch].nonzero():
                if i_exi_class == 0:
                    loss -= (sm_mask[i_batch, i_exi_class, :, :].reshape([self.total_pixel_num]).sort(descending=True)[0]*self.w_bg_norm).sum().log()
                else:
                    loss_temp -= (sm_mask[i_batch, i_exi_class, :, :].reshape([self.total_pixel_num]).sort(descending=True)[0]*self.w_fg_norm).sum().log()

            loss += loss_temp/len(labels[i_batch, 1:].nonzero())

            # for non-exist foreground
            loss_temp = 0.0
            for i_non_exi_class in (labels[i_batch]==0).nonzero():
                loss_temp -= (sm_mask[i_batch, i_non_exi_class, :, :].reshape([self.total_pixel_num]).max()).log()

            loss += loss_temp/len((labels[i_batch]==0).nonzero())

        return loss/batch_size
