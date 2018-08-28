# use vgg16 weight
import torch
import torch.nn as nn
import vgg
import torch.nn.functional as F

class DecoupleNet(nn.Module):

    def __init__(self, num_classes):
        super(DecoupleNet, self).__init__()
        # vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'AM', 'A512', 'A512', 'A512']
        self.features = vgg.make_layers(vgg_cfg)

        # E-A
        self.drop1 = nn.Dropout(p=0.4)
        self.ea_conv = nn.Conv2d(512, num_classes, kernel_size = 1)
        self.drop2 = nn.Dropout(p=0.4)

        # D-A
        self.da_conv = nn.Conv2d(512, num_classes, kernel_size = 1)

        # classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def init_weight(self):
        nn.init.xavier_uniform(self.ea_conv.weight.data)
        nn.init.constant(self.ea_conv.bias.data, 0)
        nn.init.xavier_uniform(self.da_conv.weight.data)
        nn.init.constant(self.da_conv.bias.data, 0)

    def forward(self, x):
        x = self.features(x)
        #E-A
        ea_x = self.drop1(x)
        ea_x = self.ea_conv(ea_x)
        ea_x = self.drop2(ea_x)
        outputs = self.avg_pool(ea_x)
        return ea_x, outputs
