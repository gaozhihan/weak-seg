import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from voc_data import VOCData
import torch.nn.functional as F
import time
import socket
import os
import sec
import torchvision.models.resnet as resnet
from arguments import get_args
import numpy as np
import matplotlib.pyplot as plt

args = get_args()

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()

if host_name == 'sunting':
    args.batch_size = 5
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 18
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
elif host_name == 'ram-lab':
    args.data_dir = '/data_shared/Docker/ltai/ws/decoupled_net/data/VOC2012/VOC2012_SEG_AUG'
    if args.model == 'SEC':
        args.batch_size = 50
    elif args.model == 'resnet':
        args.batch_size = 100


if args.model == 'SEC':
    # model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # 'vgg16'
    model_path = 'models/0506/top_val_rec_SEC_05.pth' # 'vgg16'
    net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)
    #net.load_state_dict(model_zoo.load_url(model_url), strict = False)
    net.load_state_dict(torch.load(model_path), strict = True)

elif args.model == 'resnet':
    model_path = 'models/0506/resnet50_feat.pth'
    net = resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = False)

print(args)
thrs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
thrs_num = len(thrs)

TP_train = torch.from_numpy(np.zeros(thrs_num)); TP_eval = torch.from_numpy(np.zeros(thrs_num))
T_train = torch.from_numpy(np.zeros(thrs_num));  T_eval = torch.from_numpy(np.zeros(thrs_num))
P_train = torch.from_numpy(np.zeros(thrs_num));  P_eval = torch.from_numpy(np.zeros(thrs_num))

recall_train = np.zeros(thrs_num)
acc_train = np.zeros(thrs_num)
recall_eval = np.zeros(thrs_num)
acc_eval = np.zeros(thrs_num)

if flag_use_cuda:
    net.cuda()
    TP_train = TP_train.cuda()
    TP_eval = TP_eval.cuda()
    T_train = T_train.cuda()
    T_eval = T_eval.cuda()
    P_train = P_train.cuda()
    P_eval = P_eval.cuda()

dataloader = VOCData(args)

start = time.time()
for phase in ['train', 'val']:
    if phase == 'train':
        net.train(True)

        for data in dataloader.dataloaders["train"]:
            inputs, labels = data
            if flag_use_cuda:
                inputs = inputs.cuda(); labels = labels.cuda()

            with torch.no_grad():
                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                elif args.model == 'resnet':
                    outputs = net(inputs)

            for i in range(thrs_num):
                # preds = (torch.sigmoid(outputs.squeeze().data)>thrs[i])
                preds = outputs.squeeze().data>thrs[i]
                TP_train[i] += torch.sum(preds.long() == (labels*2-1).data.long()).double()
                T_train[i] += torch.sum(labels.data.long()==1).double()
                P_train[i] += torch.sum(preds.long()==1).double()

    else:  # evaluation
        net.train(False)
        start = time.time()
        for data in dataloader.dataloaders["val"]:
            inputs, labels = data
            if flag_use_cuda:
                inputs = inputs.cuda(); labels = labels.cuda()

            with torch.no_grad():
                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                elif args.model == 'resnet':
                    outputs = net(inputs)

            for i in range(thrs_num):
                # preds = (torch.sigmoid(outputs.squeeze().data)>thrs[i])
                preds = outputs.squeeze().data>thrs[i]
                TP_eval[i] += torch.sum(preds.long() == (labels*2-1).data.long()).double()
                T_eval[i] += torch.sum(labels.data.long()==1).double()
                P_eval[i] += torch.sum(preds.long()==1).double()

time_took = time.time() - start

for i in range(thrs_num):
    if flag_use_cuda:
        recall_train[i] = TP_train.cpu().numpy()[i] / T_train.cpu().numpy()[i] if T_train.cpu().numpy()[i]!=0 else 0
        acc_train[i] = TP_train.cpu().numpy()[i] / P_train.cpu().numpy()[i] if P_train.cpu().numpy()[i]!=0 else 0
        recall_eval[i] = TP_eval.cpu().numpy()[i] / T_eval.cpu().numpy()[i] if T_eval.cpu().numpy()[i]!=0 else 0
        acc_eval[i] = TP_eval.cpu().numpy()[i] / P_eval.cpu().numpy()[i] if P_eval.cpu().numpy()[i]!=0 else 0
    else:
        recall_train[i] = TP_train.numpy()[i] / T_train.numpy()[i] if T_train.numpy()[i]!=0 else 0
        acc_train[i] = TP_train.numpy()[i] / P_train.numpy()[i] if P_train.numpy()[i]!=0 else 0
        recall_eval[i] = TP_eval.numpy()[i] / T_eval.numpy()[i] if T_eval.numpy()[i]!=0 else 0
        acc_eval[i] = TP_eval.numpy()[i] / P_eval.numpy()[i] if P_eval.numpy()[i]!=0 else 0

# print('TP_train: {};   T_train: {};   P_train: {};   acc_train: {};   recall_train: {} '.format(TP_train, T_train, P_train, acc_train, recall_train))
# print('TP_eval: {};   T_eval: {};   P_eval: {};   acc_eval: {};   recall__eval: {} '.format(TP_eval, T_eval, P_eval, acc_eval, recall_eval))
plt.plot(recall_train,acc_train,'bo',recall_eval,acc_eval,'r+')
plt.xlabel('recall')
plt.ylabel('accuracy')

print("done")
