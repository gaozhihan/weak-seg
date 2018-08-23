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
import my_resnet2
from arguments import get_args

args = get_args()
#args.input_size = [256,256]

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
    elif args.model == 'my_resnet':
        args.batch_size = 32


if args.model == 'SEC':
    # model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # 'vgg16'
    model_path = 'models/vgg16-397923af.pth' # 'vgg16'
    net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)
    #net.load_state_dict(model_zoo.load_url(model_url), strict = False)
    net.load_state_dict(torch.load(model_path), strict = False)

elif args.model == 'resnet':
    model_path = 'models/resnet50_feat.pth'
    net = resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = False)

elif args.model == 'my_resnet':
    model_path = 'models/resnet50_feat.pth'
    net = my_resnet2.resnet50(pretrained=False, num_classes=args.num_classes)
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
    train_loss1 = 0.0
    eval_loss1 = 0.0
    TP_train1 = 0; TP_eval1 = 0
    T_train1 = 0;  T_eval1 = 0
    P_train1 = 0;  P_eval1 = 0

    train_loss2 = 0.0
    eval_loss2 = 0.0
    TP_train2 = 0; TP_eval2 = 0
    T_train2 = 0;  T_eval2 = 0
    P_train2 = 0;  P_eval2 = 0


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

                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                    preds = outputs.squeeze().data>args.threshold
                elif args.model == 'resnet' or args.model == 'my_resnet':
                    outputs1, outputs2, outputs_seg = net(inputs)
                    outputs1 = torch.sigmoid(outputs1)
                    outputs2 = torch.sigmoid(outputs2)
                    preds1 = outputs1.squeeze().data>args.threshold
                    preds2 = outputs2.squeeze().data>args.threshold

                loss1 = criterion(outputs1.squeeze(), labels.detach())
                loss2 = criterion(outputs2.squeeze(), labels.detach())
                (loss1+loss2).backward()  # independent backward would cause Error: Trying to backward through the graph a second time ...
                optimizer.step()

                train_loss1 += loss1.item() * inputs.size(0)
                TP_train1 += torch.sum(preds1.long() == (labels*2-1).data.long())
                T_train1 += torch.sum(labels.data.long()==1)
                P_train1 += torch.sum(preds1.long()==1)

                train_loss2 += loss2.item() * inputs.size(0)
                TP_train2 += torch.sum(preds2.long() == (labels*2-1).data.long())
                T_train2 += torch.sum(labels.data.long()==1)
                P_train2 += torch.sum(preds2.long()==1)


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
                        preds = outputs.squeeze().data>args.threshold
                    elif args.model == 'resnet' or args.model == 'my_resnet':
                        outputs1, outputs2, outputs_seg = net(inputs)
                        outputs1 = torch.sigmoid(outputs1)
                        outputs2 = torch.sigmoid(outputs2)
                        preds1 = outputs1.squeeze().data>args.threshold
                        preds2 = outputs2.squeeze().data>args.threshold

                loss1 = criterion(outputs1.squeeze(), labels)
                eval_loss1 += loss1.item() * inputs.size(0)
                TP_eval1 += torch.sum(preds1.long() == (labels*2-1).data.long())
                T_eval1 += torch.sum(labels.data.long()==1)
                P_eval1 += torch.sum(preds1.long()==1)

                loss2 = criterion(outputs2.squeeze(), labels)
                eval_loss2 += loss2.item() * inputs.size(0)
                TP_eval2 += torch.sum(preds2.long() == (labels*2-1).data.long())
                T_eval2 += torch.sum(labels.data.long()==1)
                P_eval2 += torch.sum(preds2.long()==1)



    time_took = time.time() - start
    epoch_train_loss1 = train_loss1 / dataloader.dataset_sizes["train"]
    epoch_eval_loss1 = eval_loss1 / dataloader.dataset_sizes["val"]
    epoch_train_loss2 = train_loss2 / dataloader.dataset_sizes["train"]
    epoch_eval_loss2 = eval_loss2 / dataloader.dataset_sizes["val"]

    if flag_use_cuda:
        recall_train1 = TP_train1.cpu().numpy() / T_train1.cpu().numpy() if T_train1!=0 else 0
        acc_train1 = TP_train1.cpu().numpy() / P_train1.cpu().numpy() if P_train1!=0 else 0
        recall_eval1 = TP_eval1.cpu().numpy() / T_eval1.cpu().numpy() if T_eval1!=0 else 0
        acc_eval1 = TP_eval1.cpu().numpy() / P_eval1.cpu().numpy() if P_eval1!=0 else 0
    else:
        recall_train1 = TP_train1.numpy() / T_train1.numpy() if T_train1!=0 else 0
        acc_train1 = TP_train1.numpy() / P_train1.numpy() if P_train1!=0 else 0
        recall_eval1 = TP_eval1.numpy() / T_eval1.numpy() if T_eval1!=0 else 0
        acc_eval1 = TP_eval1.numpy() / P_eval1.numpy() if P_eval1!=0 else 0


    if flag_use_cuda:
        recall_train2 = TP_train2.cpu().numpy() / T_train2.cpu().numpy() if T_train2!=0 else 0
        acc_train2 = TP_train2.cpu().numpy() / P_train2.cpu().numpy() if P_train2!=0 else 0
        recall_eval2 = TP_eval2.cpu().numpy() / T_eval2.cpu().numpy() if T_eval2!=0 else 0
        acc_eval2 = TP_eval2.cpu().numpy() / P_eval2.cpu().numpy() if P_eval2!=0 else 0
    else:
        recall_train2 = TP_train2.numpy() / T_train2.numpy() if T_train2!=0 else 0
        acc_train2 = TP_train2.numpy() / P_train2.numpy() if P_train2!=0 else 0
        recall_eval2 = TP_eval2.numpy() / T_eval2.numpy() if T_eval2!=0 else 0
        acc_eval2 = TP_eval2.numpy() / P_eval2.numpy() if P_eval2!=0 else 0

    # print('TP_train: {};   T_train: {};   P_train: {};   acc_train: {};   recall_train: {} '.format(TP_train, T_train, P_train, acc_train, recall_train))
    # print('TP_eval: {};   T_eval: {};   P_eval: {};   acc_eval: {};   recall__eval: {} '.format(TP_eval, T_eval, P_eval, acc_eval, recall_eval))

    if acc_eval1 > max_acc:
        print('save model ' + args.model + ' with val acc: {}'.format(acc_eval1))
        torch.save(net.state_dict(), './models/top_val_acc_'+ args.model + '_18.pth')
        max_acc = acc_eval1

    if recall_eval1 > max_recall:
        print('save model ' + args.model + ' with val recall: {}'.format(recall_eval1))
        torch.save(net.state_dict(), './models/top_val_rec_'+ args.model + '_18.pth')
        max_recall = recall_eval1

    print('1 Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss1, acc_train1, recall_train1, epoch_eval_loss1, acc_eval1, recall_eval1))
    print('2 Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss2, acc_train2, recall_train2, epoch_eval_loss2, acc_eval2, recall_eval2))


print("done")
