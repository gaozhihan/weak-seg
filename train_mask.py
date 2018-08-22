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
import my_resnet
from arguments import get_args
import common_function
import CRF_all_class
import numpy as np

args = get_args()
args.need_mask_flag = True

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()

if host_name == 'sunting':
    args.batch_size = 5 # if this is 1, the dimension of feat_conv.shape in cam_extract will have problem
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
elif host_name == 'sunting-ThinkCenter-M90':
    args.batch_size = 18
elif host_name == 'ram-lab':
    args.data_dir = '/data_shared/Docker/ltai/ws/decoupled_net/data/VOC2012/VOC2012_SEG_AUG'
    if args.model == 'SEC':
        args.batch_size = 50
    elif args.model == 'resnet':
        args.batch_size = 100
    elif args.model == 'my_resnet':
        args.batch_size = 36


if args.model == 'SEC':
    # model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # 'vgg16'
    model_path = 'models/0506/top_val_rec_SEC_05_CPU.pth' # 'vgg16'
    net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)
    net.load_state_dict(torch.load(model_path), strict = True)

elif args.model == 'resnet':
    model_path = 'models/top_val_acc_resnet_CPU.pth'
    net = resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)
    features_blob = []
    params = list(net.parameters())
    fc_weight = params[-2]
    def hook_feature(module, input, output):
        features_blob.append(output.data)
    net._modules.get('layer4').register_forward_hook(hook_feature)

elif args.model == 'my_resnet':
    model_path = 'models/top_val_acc_my_resnet_drp_28_no_spread_CPU.pth'  # top_val_acc_my_resnet_drp_CPU
    net = my_resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)
    features_blob = []
    params = list(net.parameters())
    fc_weight = params[-2]
    def hook_feature(module, input, output):
        features_blob.append(output.data)
    net._modules.get('layer4').register_forward_hook(hook_feature)

criterion1 = nn.MultiLabelSoftMarginLoss()
# criterion2 = common_function.MapCrossEntropyLoss()
criterion2 = common_function.MapWeightedCrossEntropyLoss()
print(args)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)
crf = CRF_all_class.CRF(args)


optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

max_acc = 0
max_recall = 0
max_iou = 0
iou_obj = common_function.iou_calculator()


for epoch in range(args.epochs):
    train_loss1 = 0.0
    train_loss2 = 0.0
    eval_loss1 = 0.0
    eval_loss2 = 0.0
    TP_train = 0; TP_eval = 0
    T_train = 0;  T_eval = 0
    P_train = 0;  P_eval = 0
    main_scheduler.step()
    start = time.time()
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train(True)

            for data in dataloader.dataloaders["train"]:
                inputs, labels, mask_gt, img = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                optimizer.zero_grad()
                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                    preds = outputs.data>args.threshold
                elif args.model == 'resnet' or args.model == 'my_resnet':
                    outputs = net(inputs)
                    outputs = torch.sigmoid(outputs)
                    preds = outputs.data>args.threshold
                    mask = common_function.cam_extract(features_blob[0], fc_weight, args.relu_mask)
                    features_blob.clear()

                mask_s_gt_np = np.zeros(mask.shape,dtype=np.float32)
                confidence = np.zeros(mask.shape[0])
                for i in range(labels.shape[0]):
                    if flag_use_cuda:
                        mask_s_gt_np[i,:,:,:], mask_pred, confidence[i] = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().cpu().numpy(), args.preds_only)
                    else:
                        mask_s_gt_np[i,:,:,:], mask_pred, confidence[i] = crf.runCRF(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().numpy(), args.preds_only)

                    iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pred)
                    # obj = CRF_lei.CAM_iou(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().numpy())
                    # iou_np+=obj.run()

                mask_s_gt = torch.from_numpy(mask_s_gt_np)
                loss1 = criterion1(outputs, labels)
                loss2 = criterion2(mask, mask_s_gt, confidence)
                loss1.backward()
                loss2.backward()
                optimizer.step()

                train_loss1 += loss1.item() * inputs.size(0)
                train_loss2 += loss2.item() * inputs.size(0)

                TP_train += torch.sum(preds.long() == (labels*2-1).data.long())
                T_train += torch.sum(labels.data.long()==1)
                P_train += torch.sum(preds.long()==1)

            temp_iou = iou_obj.cal_cur_iou()
            print('current train iou is:')
            print(temp_iou, temp_iou.mean())
            iou_obj.iou_clear()

        else:  # evaluation
            net.train(False)
            start = time.time()
            for data in dataloader.dataloaders["val"]:
                inputs, labels, mask_gt, img = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                with torch.no_grad():

                    if args.model == 'SEC':
                        mask, outputs = net(inputs)
                        preds = outputs.data>args.threshold
                    elif args.model == 'resnet' or args.model == 'my_resnet':
                        outputs = net(inputs)
                        outputs = torch.sigmoid(outputs)
                        preds = outputs.data>args.threshold
                        mask = common_function.cam_extract(features_blob[0], fc_weight, args.relu_mask)
                        features_blob.clear()

                    mask_s_gt_np = np.zeros(mask.shape,dtype=np.float32)
                    confidence = np.zeros(mask.shape[0])
                    for i in range(labels.shape[0]):
                        if flag_use_cuda:
                            mask_s_gt_np[i,:,:,:], mask_pred, confidence[i] = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().cpu().numpy(), args.preds_only)
                        else:
                            mask_s_gt_np[i,:,:,:], mask_pred, confidence[i] = crf.runCRF(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().numpy(), args.preds_only)

                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pred)
                        # obj = CRF_lei.CAM_iou(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().numpy())
                        # iou_np+=obj.run()

                mask_s_gt = torch.from_numpy(mask_s_gt_np)
                loss1 = criterion1(outputs, labels)
                loss2 = criterion2(mask, mask_s_gt, confidence)
                eval_loss1 += loss1.item() * inputs.size(0)
                eval_loss2 += loss2.item() * inputs.size(0)

                TP_eval += torch.sum(preds.long() == (labels*2-1).data.long())
                T_eval += torch.sum(labels.data.long()==1)
                P_eval += torch.sum(preds.long()==1)


    time_took = time.time() - start
    epoch_train_loss1 = train_loss1 / dataloader.dataset_sizes["train"]
    epoch_train_loss2 = train_loss2 / dataloader.dataset_sizes["train"]
    epoch_eval_loss1 = eval_loss1 / dataloader.dataset_sizes["val"]
    epoch_eval_loss2 = eval_loss2 / dataloader.dataset_sizes["val"]

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
        torch.save(net.state_dict(), './models/M_top_val_acc_'+ args.model + '.pth')
        max_acc = acc_eval

    if recall_eval > max_recall:
        print('save model ' + args.model + ' with val recall: {}'.format(recall_eval))
        torch.save(net.state_dict(), './models/M_top_val_rec_'+ args.model + '.pth')
        max_recall = recall_eval

    # for iou
    temp_iou = iou_obj.cal_cur_iou()
    print('current eval iou is:')
    cur_eval_iou = temp_iou.mean()
    print(temp_iou, cur_eval_iou)
    iou_obj.iou_clear()

    if cur_eval_iou > max_iou:
        print('save model ' + args.model + ' with val mean iou: {}'.format(cur_eval_iou))
        torch.save(net.state_dict(), './models/M_top_val_iou_'+ args.model + '.pth')
        max_iou = cur_eval_iou



    print('Epoch: {} took {:.2f}, Train loss1: {:.4f}, loss2: {:.4f} , Acc: {:.4f}, Recall: {:.4f}; eval loss1: {:.4f}, loss2: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss1, epoch_train_loss2, acc_train, recall_train, epoch_eval_loss1, epoch_eval_loss2, acc_eval, recall_eval))


print("done")
