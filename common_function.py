import torch


def cam_extract(feat_conv, fc_weight):
    cam_map = torch.matmul(fc_weight, feat_conv.view(2048,-1))
    return cam_map.view(-1,21,7,7)
