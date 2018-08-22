import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def cam_extract(feat_conv, fc_weight, relu_flag = False):
    num_class = fc_weight.shape[0]
    num_map = fc_weight.shape[1]
    H = feat_conv.shape[2]
    W = feat_conv.shape[3]
    mask = torch.zeros(feat_conv.shape[0], num_class, H, W)
    for i in range(feat_conv.shape[0]):
        mask[i,:,:,:] = torch.matmul(fc_weight, feat_conv[i,:,:,:].view(num_map, -1)).view(num_class, H, W)

    if relu_flag:
        mask = F.relu(mask)
    # mask = F.softmax(mask,dim=1)
    return mask


class iou_calculator():
    def __init__(self, num_class = 21):
        self.num_class = num_class
        self.iou_sum = np.zeros([num_class,num_class])
        self.tedges = np.arange(-0.5, num_class, 1)
        self.pedges = np.arange(-0.5, num_class, 1)

    def iou_clear(self):
        self.iou_sum = np.zeros([self.num_class,self.num_class])

    def add_iou_mask_pair(self, mask_gt, mask_pred):
        # ignore boundary
        k = (mask_gt!=255)
        iou_this_pair = np.histogram2d(mask_gt[k].reshape(-1), mask_pred[k].reshape(-1), bins = (self.tedges, self.pedges))
        self.iou_sum += iou_this_pair[0]

    def cal_cur_iou(self):
        intersec = np.diag(self.iou_sum)
        union = (self.iou_sum.sum(0) + self.iou_sum.sum(1) - intersec)
        return intersec/union




class weighted_pool(nn.Module):
    def __init__(self, batch_size, num_classes, map_size, no_bg, flag_use_cuda):
        super(weighted_pool, self).__init__()
        self.dim = map_size[0] * map_size[1]
        self.batch_size = batch_size
        self.df = 0.996 # foreground decay
        self.db = 0.999 # background decay
        weight = np.ones([batch_size, num_classes, self.dim])

        sf = 1.0
        sb = 1.0

        for i in range(self.dim):
            if no_bg:
                weight[:,0:,i] = weight[:,0:,i] * sf
            else:
                weight[:,0,i] = weight[:,0,i] * sb

            weight[:,1:,i] = weight[:,1:,i] * sf

            sb = sb * self.db
            sf = sf * self.df

        weight[:,0,:] = weight[:,0,:]/np.sum(weight[0,0,:])
        weight[:,1:,:] = weight[:,1:,:]/np.sum(weight[0,1,:])

        self.pool_weight = torch.from_numpy(weight.astype('float32'))
        if flag_use_cuda:
            self.pool_weight = self.pool_weight.cuda()


    def forward(self, outputs):
        outputs = outputs.view(outputs.size()[0],outputs.size()[1], -1)

        if outputs.shape[0] != self.batch_size:
            dim_temp = outputs.shape[0]
            outputs = torch.mul(outputs, self.pool_weight[:dim_temp,:,:])
        else:
            outputs = torch.mul(outputs, self.pool_weight)

        outputs = torch.sum(outputs, dim=2)

        return outputs


class MapCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MapCrossEntropyLoss, self).__init__()
        self.scaler = 0.5

    def forward(self, map, map_s_gt):
        return F.binary_cross_entropy(F.sigmoid(map), map_s_gt) * self.scaler


class MapWeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MapWeightedCrossEntropyLoss, self).__init__()

    def forward(self, map, map_s_gt, weight):
        num_batch = map.shape[0]
        loss = 0
        for i in range(num_batch):
            loss += F.binary_cross_entropy(F.sigmoid(map[i,:,:,:]), map_s_gt[i,:,:,:]) * weight[i]
        return loss/num_batch






