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

args = get_args()

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()

if host_name == 'sunting':
    args.batch_size = 5
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
elif host_name == 'sunting-ThinkCenter-M90':
    args.batch_size = 18
elif host_name == 'ram-lab':
    args.data_dir = '/data_shared/Docker/ltai/ws/decoupled_net/data/VOC2012/VOC2012_SEG_AUG'
    if args.model == 'SEC':
        args.batch_size = 50
    elif args.model == 'resnet':
        args.batch_size = 100


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

                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                elif args.model == 'resnet':
                    outputs = net(inputs)

                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs.squeeze().data)>0.5)
                TP_train += torch.sum(preds.long() == (labels*2-1).data.long())
                T_train += torch.sum(labels.data.long()==1)
                P_train += torch.sum(preds.long()==1)


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

                loss = criterion(outputs.squeeze(), labels)
                eval_loss += loss.item() * inputs.size(0)

                #preds = (torch.sigmoid(outputs.squeeze().data)>0.5)
                preds = outputs.squeeze().data>0.3
                TP_eval += torch.sum(preds.long() == (labels*2-1).data.long())
                T_eval += torch.sum(labels.data.long()==1)
                P_eval += torch.sum(preds.long()==1)

    time_took = time.time() - start
    epoch_train_loss = train_loss / dataloader.dataset_sizes["train"]
    epoch_eval_loss = eval_loss / dataloader.dataset_sizes["val"]

    if flag_use_cuda:
        recall_train = TP_train.cpu().numpy() / T_train.cpu().numpy() if T_train!=0 else 0
        acc_train = TP_train.cpu().numpy() / P_train.cpu().numpy() if P_train!=0 else 0
        recall_eval = TP_eval.cpu().numpy() / T_eval.cpu().numpy() if T_eval!=0 else 0
        acc_eval = TP_eval.cpu().numpy() / P_eval.cpu().numpy() if P_eval!=0 else 0
    else:
        recall_train = TP_train.numpy() / T_train.numpy() if T_train!=0 else 0
        acc_train = TP_train.numpy() / P_train.numpy() if P_train!=0 else 0
        recall_eval = TP_eval.numpy() / T_eval.numpy() if T_eval!=0 else 0
        acc_eval = TP_eval.numpy() / P_eval.numpy() if P_eval!=0 else 0

    # print('TP_train: {};   T_train: {};   P_train: {};   acc_train: {};   recall_train: {} '.format(TP_train, T_train, P_train, acc_train, recall_train))
    # print('TP_eval: {};   T_eval: {};   P_eval: {};   acc_eval: {};   recall__eval: {} '.format(TP_eval, T_eval, P_eval, acc_eval, recall_eval))

    if acc_eval > max_acc:
        print('save model ' + args.model + ' with val acc: {}'.format(acc_eval))
        torch.save(net.state_dict(), './models/top_val_acc_'+ args.model + '_06.pth')
        max_acc = acc_eval

    if recall_eval > max_recall:
        print('save model ' + args.model + ' with val recall: {}'.format(recall_eval))
        torch.save(net.state_dict(), './models/top_val_rec_'+ args.model + '_06.pth')
        max_recall = recall_eval

    print('Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss, acc_train, recall_train, epoch_eval_loss, acc_eval, recall_eval))


print("done")
