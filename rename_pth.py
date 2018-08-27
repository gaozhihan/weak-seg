import torch
import torch.nn as nn
import sec
import numpy as np

model_path = 'models/sec_pytorch.pth' # 'vgg16'
x = torch.load(model_path)
y = {}
for key_old in x:
    key_new = 'features.' + key_old
    y[key_new] = x[key_old]

torch.save(y, './models/sec_rename.pth')
