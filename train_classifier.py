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
from arguments import get_args

args = get_args()

# model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # 'vgg16'
model_path = 'models/vgg16-397923af.pth' # 'vgg16'

host_name = socket.gethostname()

if host_name == 'sunting':
    args.batch_size = 5
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
elif host_name == 'sunting-ThinkCenter-M90':
    args.batch_size = 18
elif host_name == 'ram-lab':
    args.batch_size = 50
    args.data_dir = '/data_shared/Docker/ltai/ws/decoupled_net/data/VOC2012/VOC2012_SEG_AUG'


flag_use_cuda = torch.cuda.is_available()
net = sec.SEC_NN()

#net.load_state_dict(model_zoo.load_url(model_url), strict = False)
net.load_state_dict(torch.load(model_path), strict = False)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)
criterion = sec.weighted_pool_mul_class_loss(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)

optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

# criterion = nn.MultiLabelSoftMarginLoss()
max_acc = 0

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

                outputs = net(inputs)

                optimizer.zero_grad()
                loss, outputs = criterion(labels, outputs)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)


        else:  # evaluation
            net.train(False)
            start = time.time()
            for data in dataloader.dataloaders["val"]:
                inputs, labels = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                with torch.no_grad():
                    outputs = net(inputs)
                    loss, outputs = criterion(labels, outputs)

                eval_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs.squeeze().data)>0.5)
                TP_eval += torch.sum(preds.long() == (labels*2-1).data.long())
                T_eval += torch.sum(labels.data.long()==1)
                P_eval += torch.sum(preds.long()==1)

    time_took = time.time() - start
    epoch_train_loss = train_loss / dataloader.dataset_sizes["train"]
    epoch_eval_loss = eval_loss / dataloader.dataset_sizes["val"]

    recall_train = TP_train / T_train if T_train!=0 else 0
    acc_train = TP_train / P_train if P_train!=0 else 0
    recall_eval = TP_eval / T_eval if T_eval!=0 else 0
    acc_eval = TP_eval / P_eval if P_eval!=0 else 0

    if acc_eval > max_acc:
        torch.save(net.state_dict(), './models/top_val_acc.pth')
        max_acc = acc_eval

    print('Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss, acc_train, recall_train, epoch_eval_loss, acc_eval, recall_eval))


print("done")
