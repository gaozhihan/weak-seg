import torch
import torch.nn as nn
import torch.optim as optim
from voc_data import VOCData
import time
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np

args = get_args()
args.need_mask_flag = False
args.model = 'SEC'
args.input_size = [321,321]
args.output_size = [41, 41]

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
now = datetime.datetime.now()
date_str = str(now.day) + '_' + str(now.day)

if host_name == 'sunting':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/vgg16-397923af.pth'
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    args.batch_size = 24

net = st_01.sec_net.SEC_NN()
net.load_state_dict(torch.load(model_path), strict = False)

if args.loss == 'BCELoss':
    criterion = nn.BCELoss()
elif args.loss == 'MultiLabelSoftMarginLoss':
    criterion = nn.MultiLabelSoftMarginLoss()

print(args)

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
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                optimizer.zero_grad()

                sm_mask, preds = net(inputs)

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
                    sm_mask, preds = net(inputs)

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
        torch.save(net.state_dict(), './models/st_01_top_val_acc_'+ args.model + '_' + date_str + '.pth')
        max_acc = acc_eval

    if recall_eval > max_recall:
        print('save model ' + args.model + ' with val recall: {}'.format(recall_eval))
        torch.save(net.state_dict(), './st_01/models/st_01_top_val_rec_'+ args.model + '_' + date_str + '.pth')
        max_recall = recall_eval

    print('Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss, acc_train, recall_train, epoch_eval_loss, acc_eval, recall_eval))


print("done")
