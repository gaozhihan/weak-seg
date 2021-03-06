import torch
import torch.nn as nn
import numpy as np
import common_function
import pickle

class SEC_NN(nn.Module):
    def __init__(self, batch_size, num_classes, map_size, no_bg, flag_use_cuda):
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
        nn.Dropout(0.2),
        nn.Conv2d(1024,21,(1, 1)), # 1024 / 512
        # nn.Softmax2d()
        )

        #self.mask2label_pool = nn.AdaptiveMaxPool2d(1)
        # self.mask2label_pool = nn.AdaptiveAvgPool2d(1)
        #self.mask2label_pool = nn.Sequential(  # need to devide, since this is sum but not average
        #    nn.ReLU(),
        #    nn.LPPool2d(2, (29, 29), stride=(29, 29)))
        #self.mask2label_pool = common_function.weighted_pool(batch_size, num_classes, map_size, no_bg, flag_use_cuda)
        self.mask2label_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1),
            nn.AdaptiveAvgPool2d(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        mask = self.features(x)
        output = self.mask2label_pool(mask)

        return mask, output



class weighted_pool_mul_class_loss(nn.Module):
    def __init__(self, batch_size, num_classes, map_size, no_bg, flag_use_cuda):
        super(weighted_pool_mul_class_loss, self).__init__()
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


    def forward(self, labels, outputs):
        outputs = outputs.view(outputs.size()[0],outputs.size()[1], -1)

        if outputs.shape[0] != self.batch_size:
            dim_temp = outputs.shape[0]
            outputs = torch.mul(outputs, self.pool_weight[:dim_temp,:,:])
        else:
            outputs = torch.mul(outputs, self.pool_weight)

        outputs = torch.sum(outputs, dim=2)

        loss = nn.functional.multilabel_soft_margin_loss(outputs, labels)
        return loss, outputs


## ---------------------- implement CRF -----------------------------------------
class SeedLossLayer(nn.Module):
    def __init__(self):
        infile = open('models/localization_cues.pickle','rb')
        new_dict = pickle.load(infile)
        infile.close()

        self.cues = np.zeros(())

    def forward(self, labels, ids):
