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
import common_function
import numpy as np
import CRF_all_class
import datetime

args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]
args.cross_entropy_weight = False

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

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
    model_path = 'models/top_val_acc_my_resnet2_9_1_CPU.pth'
    net = resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)
    feature_blob = []
    params = list(net.parameters())
    fc_weight = params[-2]
    def hook_feature(module, input, output):
        feature_blob.append(output.data)
    net._modules.get('layer4').register_forward_hook(hook_feature)

elif args.model == 'my_resnet':
    model_path = 'models/top_val_acc_my_resnet2_9_1_CPU.pth'
    net = my_resnet2.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = False)
    feature_blob = []
    params = list(net.parameters())
    fc_weight = params[-4] # for my_resnet, it's fc_weight = params[-2], for my_resnet2, it's fc_weight = params[-4]
    def hook_feature(module, input, output):
        feature_blob.append(output.data)
    net._modules.get('layer4').register_forward_hook(hook_feature)


# fix non seg head
for child in net.children():
    for param in child.parameters():
        param.requires_grad = False

net.seg_conv.weight.requires_grad = True
net.seg_conv.bias.requires_grad = True


if args.loss == 'BCELoss':
    criterion1 = nn.BCELoss()
elif args.loss == 'MultiLabelSoftMarginLoss':
    criterion1 = nn.MultiLabelSoftMarginLoss()

criterion_seed = common_function.SeedingLoss()
criterion_expension = nn.BCELoss() # classification
criterion_boundary = nn.KLDivLoss(size_average=False) # after CRF; the input given is expected to contain log-probabilities

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)
crf = CRF_all_class.CRF(args)


# optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
optimizer = optim.Adam(net.seg_conv.parameters(), lr=args.lr)
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

max_acc = 0
max_recall = 0
max_iou = 0
iou_obj = common_function.iou_calculator()

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

    train_expension_loss = 0.0
    eval_expension_loss = 0.0
    train_seed_loss = 0.0
    eval_seed_loss = 0.0
    train_boundary_loss = 0.0
    eval_boundary_loss = 0.0

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
                    preds = outputs.squeeze().data>args.threshold
                elif args.model == 'resnet' or args.model == 'my_resnet':
                    outputs1, outputs2, outputs_seg = net(inputs)
                    outputs1 = torch.sigmoid(outputs1)
                    outputs2 = torch.sigmoid(outputs2)
                    preds1 = outputs1.squeeze().data>args.threshold
                    preds2 = outputs2.squeeze().data>args.threshold
                    cam_mask = common_function.cam_extract(feature_blob[0], fc_weight, args.relu_mask)
                    feature_blob.clear()
                    mask = cam_mask # or mask = outputs_seg

                mask_s_gt_np = np.zeros(mask.shape,dtype=np.float32)
                mask_seed = np.zeros(mask.shape,dtype=np.float32)
                confidence = np.zeros(mask.shape[0])
                for i in range(labels.shape[0]):
                    if flag_use_cuda:
                        mask_s_gt_np[i,:,:,:], mask_pred, confidence[i], mask_seed[i,:,:,:] = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds1[i,:].detach().cpu().numpy(), args.preds_only)
                    else:
                        mask_s_gt_np[i,:,:,:], mask_pred, confidence[i], mask_seed[i,:,:,:] = crf.runCRF(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds1[i,:].detach().numpy(), args.preds_only)

                    iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pred)

                mask_s_gt = torch.from_numpy(mask_s_gt_np)
                loss1 = criterion1(outputs1.squeeze(), labels)
                loss2 = criterion1(outputs2.squeeze(), labels)
                if flag_use_cuda:
                    seed_loss = criterion_seed(outputs_seg, torch.from_numpy(mask_seed).cuda(), labels)
                    boundary_loss = criterion_boundary(F.logsigmoid(outputs_seg), mask_s_gt.cuda())/(labels.shape[0]*outputs_seg.shape[2]*outputs_seg.shape[3])
                else:
                    seed_loss = criterion_seed(outputs_seg, torch.from_numpy(mask_seed), labels)
                    boundary_loss = criterion_boundary(F.logsigmoid(outputs_seg), mask_s_gt)/(labels.shape[0]*outputs_seg.shape[2]*outputs_seg.shape[3])

                expension_loss = criterion_expension(outputs2.squeeze(), labels)  # notice that expension_loss = loss2

                (seed_loss + expension_loss + boundary_loss).backward()  # independent backward would cause Error: Trying to backward through the graph a second time ...
                optimizer.step()

                train_loss1 += loss1.item() * inputs.size(0)
                TP_train1 += torch.sum(preds1.long() == (labels*2-1).data.long())
                T_train1 += torch.sum(labels.data.long()==1)
                P_train1 += torch.sum(preds1.long()==1)

                train_loss2 += loss2.item() * inputs.size(0)
                TP_train2 += torch.sum(preds2.long() == (labels*2-1).data.long())
                T_train2 += torch.sum(labels.data.long()==1)
                P_train2 += torch.sum(preds2.long()==1)

                train_expension_loss += expension_loss.item() * inputs.size(0)
                train_seed_loss += seed_loss.item() * inputs.size(0)
                train_boundary_loss += boundary_loss.item() * inputs.size(0)

            temp_iou = iou_obj.cal_cur_iou()
            print('current train iou is :')
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
                        preds = outputs.squeeze().data>args.threshold
                    elif args.model == 'resnet' or args.model == 'my_resnet':
                        outputs1, outputs2, outputs_seg = net(inputs)
                        outputs1 = torch.sigmoid(outputs1)
                        outputs2 = torch.sigmoid(outputs2)
                        preds1 = outputs1.squeeze().data>args.threshold
                        preds2 = outputs2.squeeze().data>args.threshold
                        cam_mask = common_function.cam_extract(feature_blob[0], fc_weight, args.relu_mask)
                        feature_blob.clear()
                        mask = cam_mask # or mask = outputs_seg

                    mask_s_gt_np = np.zeros(mask.shape,dtype=np.float32)
                    mask_seed = np.zeros(mask.shape,dtype=np.float32)
                    confidence = np.zeros(mask.shape[0])
                    for i in range(labels.shape[0]):
                        if flag_use_cuda:
                            mask_s_gt_np[i,:,:,:], mask_pred, confidence[i], mask_seed[i,:,:,:] = crf.runCRF(preds1[i,:].cpu().numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds1[i,:].detach().cpu().numpy(), args.preds_only)
                        else:
                            mask_s_gt_np[i,:,:,:], mask_pred, confidence[i], mask_seed[i,:,:,:] = crf.runCRF(preds1[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds1[i,:].detach().numpy(), args.preds_only)

                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pred)

                mask_s_gt = torch.from_numpy(mask_s_gt_np)
                loss1 = criterion1(outputs1.squeeze(), labels)
                loss2 = criterion1(outputs2.squeeze(), labels)
                if flag_use_cuda:
                    seed_loss = criterion_seed(outputs_seg, torch.from_numpy(mask_seed).cuda(), labels)
                    boundary_loss = criterion_boundary(outputs_seg, mask_s_gt.cuda())/(labels.shape[0]*outputs_seg.shape[2]*outputs_seg.shape[3])
                else:
                    seed_loss = criterion_seed(outputs_seg, torch.from_numpy(mask_seed), labels)
                    boundary_loss = criterion_boundary(outputs_seg, mask_s_gt)/(labels.shape[0]*outputs_seg.shape[2]*outputs_seg.shape[3])

                expension_loss = criterion_expension(outputs2.squeeze(), labels)

                loss1 = criterion1(outputs1.squeeze(), labels)
                eval_loss1 += loss1.item() * inputs.size(0)
                TP_eval1 += torch.sum(preds1.long() == (labels*2-1).data.long())
                T_eval1 += torch.sum(labels.data.long()==1)
                P_eval1 += torch.sum(preds1.long()==1)

                loss2 = criterion1(outputs2.squeeze(), labels)
                eval_loss2 += loss2.item() * inputs.size(0)
                TP_eval2 += torch.sum(preds2.long() == (labels*2-1).data.long())
                T_eval2 += torch.sum(labels.data.long()==1)
                P_eval2 += torch.sum(preds2.long()==1)

                eval_expension_loss += expension_loss.item() * inputs.size(0)
                eval_seed_loss += seed_loss.item() * inputs.size(0)
                eval_boundary_loss += boundary_loss.item() * inputs.size(0)

    time_took = time.time() - start
    epoch_train_loss1 = train_loss1 / dataloader.dataset_sizes["train"]
    epoch_eval_loss1 = eval_loss1 / dataloader.dataset_sizes["val"]
    epoch_train_loss2 = train_loss2 / dataloader.dataset_sizes["train"]
    epoch_eval_loss2 = eval_loss2 / dataloader.dataset_sizes["val"]
    epoch_train_seed_loss = train_seed_loss / dataloader.dataset_sizes["train"]
    epoch_eval_seed_loss = eval_seed_loss / dataloader.dataset_sizes["val"]
    epoch_train_expension_loss = train_expension_loss / dataloader.dataset_sizes["train"]
    epoch_eval_expension_loss = eval_expension_loss / dataloader.dataset_sizes["val"]
    epoch_train_boundary_loss = train_seed_loss / dataloader.dataset_sizes["train"]
    epoch_eval_boundary_loss = eval_seed_loss / dataloader.dataset_sizes["val"]


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

    if acc_eval1 > max_acc:
        print('save model ' + args.model + '2' + ' with val acc: {}'.format(acc_eval1))
        torch.save(net.state_dict(), './models/top_val_acc_'+ args.model + '2_' + date_str + '.pth')
        max_acc = acc_eval1

    if recall_eval1 > max_recall:
        print('save model ' + args.model + ' with val recall: {}'.format(recall_eval1))
        torch.save(net.state_dict(), './models/top_val_rec_'+ args.model + '2_' + date_str + '.pth')
        max_recall = recall_eval1

    temp_iou = iou_obj.cal_cur_iou()
    cur_eval_iou = temp_iou.mean()
    print('current eval iou is :')
    print(temp_iou, temp_iou.mean())
    iou_obj.iou_clear()

    if cur_eval_iou > max_iou:
        print('save model ' + args.model + ' with val mean iou: {}'.format(cur_eval_iou))
        torch.save(net.state_dict(), './models/M_top_val_iou_'+ args.model + '2_' + date_str + '.pth')
        max_iou = cur_eval_iou

    print('1 Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss1, acc_train1, recall_train1, epoch_eval_loss1, acc_eval1, recall_eval1))
    print('2 Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss2, acc_train2, recall_train2, epoch_eval_loss2, acc_eval2, recall_eval2))
    print('Train seed loss is: {:.6f}, expension loss is: {:.6f}, boundary loss is: {:.6f};  eval seed loss is: {:.6f}, expension loss is: {:.6f}, boundary loss is: {:.6f}.'.format(epoch_train_seed_loss, epoch_train_expension_loss, epoch_train_boundary_loss, epoch_eval_seed_loss, epoch_eval_expension_loss, epoch_eval_boundary_loss))

print("done")
