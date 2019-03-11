import sys
sys.path.append("/home/weak-seg")

import torch
import torch.nn as nn
import torch.optim as optim
import time
import socket
from multi_scale.voc_data_mul_scale import VOCData
# import st_resnet.resnet_st
import st_resnet.resnet_st_more_drp
import psa.network.resnet38_cls as resnet38_cls
from arguments import get_args
import datetime
import numpy as np
import random
from skimage.transform import resize
# import matplotlib.pyplot as plt

args = get_args()
args.need_mask_flag = False
args.model = 'my_resnet'
args.input_size = [321, 321]
max_size = [385, 385]
# max_size = [321, 321]
args.output_size = [41, 41]
args.rand_gray = True

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
now = datetime.datetime.now()
date_str = str(now.day) + '_' + str(now.day)

if host_name == 'sunting':
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/resnet50_feat.pth'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/resnet50_feat.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_rec_my_resnet_9_9.pth'
    args.batch_size = 10
else:
    args.data_dir = '/home/VOC2012_SEG_AUG'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/resnet50_feat.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_rec_my_resnet_9_9.pth'
    args.batch_size = 8

# net = st_resnet.resnet_st.resnet50(pretrained=False, num_classes=args.num_classes)
# net = st_resnet.resnet_st_more_drp.resnet50(pretrained=False, num_classes=args.num_classes)
# net.load_state_dict(torch.load(model_path), strict = False)
net = resnet38_cls.Net()

if args.loss == 'BCELoss':
    criterion = nn.BCELoss()
elif args.loss == 'MultiLabelSoftMarginLoss':
    criterion = nn.MultiLabelSoftMarginLoss()

print(args)
# print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)

optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

max_acc = 0
max_recall = 0

for epoch in range(args.epochs):
    train_loss = 0.0
    eval_loss = 0.0
    TP_train = 0; TP_eval = 0
    T_train = 0;  T_eval = 0
    P_train = 0;  P_eval = 0
    main_scheduler.step()
    start = time.time()
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train(True)

            for data in dataloader.dataloaders["train"]:
                inputs, labels = data

                rand_scale = random.uniform(0.67, 1.0)
                cur_size = [round(max_size[0] * rand_scale), round(max_size[1] * rand_scale)]
                inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')

                max_val = max(max(inputs.max(), -inputs.min()), 1.0).numpy()
                for i in range(inputs.shape[0]):
                    inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))

                # plt.imshow(np.transpose(inputs[0].detach().numpy(), (1,2,0)))

                if flag_use_cuda:
                    inputs = torch.from_numpy(inputs_resize).cuda(); labels = labels.cuda()
                else:
                    inputs = torch.from_numpy(inputs_resize)

                optimizer.zero_grad()

                layer4_feature, fc = net(inputs)
                preds = torch.sigmoid(fc)

                loss = criterion(preds.squeeze(), labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()

                TP_train += np.logical_and(preds_thr_numpy.squeeze(),labels_numpy).sum()
                T_train += labels_numpy.sum()
                P_train += preds_thr_numpy.sum()

        else:  # evaluation
            net.train(False)
            start = time.time()
            for data in dataloader.dataloaders["val"]:
                inputs, labels = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                with torch.no_grad():
                    layer4_feature, fc = net(inputs)
                    preds = torch.sigmoid(fc)

                loss = criterion(preds.squeeze(), labels)

                eval_loss += loss.item() * inputs.size(0)

                preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()

                TP_eval += np.logical_and(preds_thr_numpy.squeeze(),labels_numpy).sum()
                T_eval += labels_numpy.sum()
                P_eval += preds_thr_numpy.sum()

    time_took = time.time() - start
    epoch_train_loss = train_loss / dataloader.dataset_sizes["train"]
    epoch_eval_loss = eval_loss / dataloader.dataset_sizes["val"]

    if flag_use_cuda:
        recall_train = TP_train / T_train if T_train!=0 else 0
        acc_train = TP_train / P_train if P_train!=0 else 0
        recall_eval = TP_eval / T_eval if T_eval!=0 else 0
        acc_eval = TP_eval / P_eval if P_eval!=0 else 0
    else:
        recall_train = TP_train / T_train if T_train!=0 else 0
        acc_train = TP_train / P_train if P_train!=0 else 0
        recall_eval = TP_eval / T_eval if T_eval!=0 else 0
        acc_eval = TP_eval / P_eval if P_eval!=0 else 0

    # print('TP_train: {};   T_train: {};   P_train: {};   acc_train: {};   recall_train: {} '.format(TP_train, T_train, P_train, acc_train, recall_train))
    # print('TP_eval: {};   T_eval: {};   P_eval: {};   acc_eval: {};   recall__eval: {} '.format(TP_eval, T_eval, P_eval, acc_eval, recall_eval))

    if acc_eval > max_acc:
        print('save model ' + args.model + ' with val acc: {}'.format(acc_eval))
        torch.save(net.state_dict(), './multi_scale/models/st_pure_gray_top_val_acc_'+ args.model + '_' + date_str + '.pth')
        max_acc = acc_eval

    if recall_eval > max_recall:
        print('save model ' + args.model + ' with val recall: {}'.format(recall_eval))
        torch.save(net.state_dict(), './multi_scale/models/st_pure_gray_top_val_rec_'+ args.model + '_' + date_str + '.pth')
        max_recall = recall_eval

    print('Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss, acc_train, recall_train, epoch_eval_loss, acc_eval, recall_eval))


print("done")
