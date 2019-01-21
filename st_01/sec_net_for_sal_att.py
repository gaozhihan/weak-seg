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
import matplotlib.pyplot as plt

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
        nn.ReLU()
        # nn.Softmax2d()
        )

        self.drop05 = nn.Dropout(0.5)
        self.last_conv = nn.Conv2d(1024,21,(1, 1)) # 1024 / 512

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
        conv_f0 = self.features(x)
        fc_mask =self.drop05(conv_f0)
        fc_mask = self.last_conv(fc_mask)
        sm_mask = self.softmax2d(fc_mask)+self.min_prob
        sm_mask = sm_mask / sm_mask.sum(dim=1, keepdim=True)
        # sm_mask = self.softmax2d(sm_mask)
        preds = self.mask2pre(sm_mask)

        return conv_f0, sm_mask, preds
