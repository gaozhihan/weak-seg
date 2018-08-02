import torch
import torch.nn as nn
import  numpy as np


def cam_extract(feat_conv, fc_weight):
    cam_map = torch.matmul(fc_weight, feat_conv.view(2048,-1))
    return cam_map.view(-1,21,7,7)


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
