import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
#from voc_data import VOCData
from voc_data_org_size_batch import VOCData
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
import my_resnet3
import decoupled_net
import CRF_sec

args = get_args()
args.need_mask_flag = True
args.test_flag = True
args.model = 'my_resnet' # resnet; my_resnet; SEC; my_resnet3; decoupled
model_path = 'models/top_val_acc_my_resnet_25' # sec: sec_rename; resnet: top_val_acc_resnet; my_resnet: top_val_acc_my_resnet_25; my_resnet3: top_val_rec_my_resnet3_27; decoupled: top_val_acc_decoupled_28
args.input_size = [321,321]
args.output_size = [41, 41]
args.origin_size = True
args.color_vote = True
args.fix_CRF_itr = False
args.preds_only = True
args.CRF_model = 'my_CRF' # SEC_CRF or my_CRF

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()

if host_name == 'sunting':
    args.batch_size = 5
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    model_path = model_path + '_CPU'
elif host_name == 'sunting-ThinkCenter-M90':
    args.batch_size = 18
elif host_name == 'ram-lab':
    args.data_dir = '/data_shared/Docker/ltai/ws/decoupled_net/data/VOC2012/VOC2012_SEG_AUG'
    if args.model == 'SEC':
        args.batch_size = 50
    elif args.model == 'resnet':
        args.batch_size = 100
    elif args.model == 'my_resnet':
        args.batch_size = 30
    elif args.model == 'decoupled':
        args.batch_size = 38

model_path = model_path + '.pth'

# if args.origin_size:
#     args.batch_size = 1

if args.model == 'SEC':
    # model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # 'vgg16'
    # model_path = 'models/0506/top_val_rec_SEC_05_CPU.pth' # 'vgg16'
    args.input_size = [321,321]
    args.output_size = [41, 41]
    net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)
    net.load_state_dict(torch.load(model_path), strict = True)

elif args.model == 'resnet':
    net = resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)
    features_blob = []
    params = list(net.parameters())
    fc_weight = params[-2]
    def hook_feature(module, input, output):
        features_blob.append(output.data)
    net._modules.get('layer4').register_forward_hook(hook_feature)

elif args.model == 'my_resnet':
    net = my_resnet.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)
    features_blob = []
    params = list(net.parameters())
    fc_weight = params[-2]
    def hook_feature(module, input, output):
        features_blob.append(output.data)
    net._modules.get('layer4').register_forward_hook(hook_feature)

elif args.model == 'my_resnet3':
    net = my_resnet3.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)
    print(net.seg2label_pool)

elif args.model == 'decoupled':
    args.input_size = [321,321]
    args.output_size = [40, 40]
    net = decoupled_net.DecoupleNet(args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)


criterion1 = nn.MultiLabelSoftMarginLoss()
criterion2 = common_function.MapCrossEntropyLoss()
print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)
if args.CRF_model == 'my_CRF':
    crf = CRF_all_class.CRF(args)
else:
    crf = CRF_sec.CRF(args)

max_acc = 0
max_recall = 0
iou_obj = common_function.iou_calculator()

net.train(False)
with torch.no_grad():
    train_loss1 = 0.0
    train_loss2 = 0.0
    eval_loss1 = 0.0
    eval_loss2 = 0.0
    TP_train = 0; TP_eval = 0
    T_train = 0;  T_eval = 0
    P_train = 0;  P_eval = 0
    start = time.time()
    for phase in ['train', 'val']:
        if phase == 'train':

            for data in dataloader.dataloaders["train"]:
                inputs, labels, mask_gt, img = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                    outputs = outputs.squeeze()
                    preds = outputs.data>args.threshold
                elif args.model == 'resnet' or args.model == 'my_resnet':
                    outputs = net(inputs)
                    outputs = torch.sigmoid(outputs)
                    preds = outputs.data>args.threshold
                    mask = common_function.cam_extract(features_blob[0], fc_weight, args.relu_mask)
                    features_blob.clear()
                elif args.model == 'my_resnet3' or args.model == 'decoupled':
                    mask, outputs = net(inputs)
                    outputs = outputs.squeeze()
                    outputs = torch.sigmoid(outputs)
                    preds = outputs.squeeze().data>args.threshold

                mask_s_gt_np = np.zeros(mask.shape,dtype=np.float32)

                if args.batch_size == 1:
                    outputs = outputs.unsqueeze(0)
                    preds = preds.unsqueeze(0)

                for i in range(labels.shape[0]):

                    if args.origin_size:
                        if type(mask_gt) is tuple:
                            crf.set_shape(mask_gt[i])
                        else:
                            crf.set_shape(mask_gt[i,:,:].numpy())

                    if type(img) is tuple:
                        if flag_use_cuda:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i], mask[i,:,:,:].detach().cpu().numpy(), img[i], preds[i,:].detach().cpu().numpy(), args.preds_only)
                        else:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].numpy(), mask_gt[i], mask[i,:,:,:].detach().numpy(), img[i], preds[i,:].detach().numpy(), args.preds_only)
                        iou_obj.add_iou_mask_pair(mask_gt[i], mask_pred)
                    else:
                        if flag_use_cuda:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().cpu().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().cpu().numpy(), args.preds_only)
                        else:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().numpy(), args.preds_only)
                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pred)

                mask_s_gt = torch.from_numpy(mask_s_gt_np)
                loss1 = criterion1(outputs, labels)
                loss2 = criterion2(mask.cpu(), mask_s_gt)

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
            start = time.time()
            for data in dataloader.dataloaders["val"]:
                inputs, labels, mask_gt, img = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()

                if args.model == 'SEC':
                    mask, outputs = net(inputs)
                    outputs = outputs.squeeze()
                    preds = outputs.data>args.threshold
                elif args.model == 'resnet' or args.model == 'my_resnet':
                    outputs = net(inputs)
                    outputs = torch.sigmoid(outputs)
                    preds = outputs.data>args.threshold
                    mask = common_function.cam_extract(features_blob[0], fc_weight, args.relu_mask)
                    features_blob.clear()
                elif args.model == 'my_resnet3' or args.model == 'decoupled':
                    mask, outputs = net(inputs)
                    outputs = outputs.squeeze()
                    outputs = torch.sigmoid(outputs)
                    preds = outputs.squeeze().data>args.threshold

                mask_s_gt_np = np.zeros(mask.shape,dtype=np.float32)

                if args.batch_size == 1:
                    outputs = outputs.unsqueeze(0)
                    preds = preds.unsqueeze(0)

                for i in range(labels.shape[0]):
                    if args.origin_size:
                        if type(mask_gt) is tuple:
                            crf.set_shape(mask_gt[i])
                        else:
                            crf.set_shape(mask_gt[i,:,:].numpy())

                    if type(img) is tuple:
                        if flag_use_cuda:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i], mask[i,:,:,:].detach().cpu().numpy(), img[i], preds[i,:].detach().cpu().numpy(), args.preds_only)
                        else:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].numpy(), mask_gt[i], mask[i,:,:,:].detach().numpy(), img[i], preds[i,:].detach().numpy(), args.preds_only)
                        iou_obj.add_iou_mask_pair(mask_gt[i], mask_pred)
                    else:
                        if flag_use_cuda:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].cpu().numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().cpu().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().cpu().numpy(), args.preds_only)
                        else:
                            mask_s_gt_np[i,:,:,:], mask_pred = crf.runCRF(labels[i,:].numpy(), mask_gt[i,:,:].numpy(), mask[i,:,:,:].detach().numpy(), img[i,:,:,:].numpy(), preds[i,:].detach().numpy(), args.preds_only)
                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pred)

            mask_s_gt = torch.from_numpy(mask_s_gt_np)
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(mask.cpu(), mask_s_gt)
            eval_loss1 += loss1.item() * inputs.size(0)
            eval_loss2 += loss2.item() * inputs.size(0)

            TP_eval += torch.sum(preds.long() == (labels*2-1).data.long())
            T_eval += torch.sum(labels.data.long()==1)
            P_eval += torch.sum(preds.long()==1)

            temp_iou = iou_obj.cal_cur_iou()
            print('current eval iou is:')
            print(temp_iou, temp_iou.mean())
            iou_obj.iou_clear()

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


    print('It took {:.2f}, Train loss1: {:.4f}, loss2: {:.4f} , Acc: {:.4f}, Recall: {:.4f}; eval loss1: {:.4f}, loss2: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(time_took, epoch_train_loss1, epoch_train_loss2, acc_train, recall_train, epoch_eval_loss1, epoch_eval_loss2, acc_eval, recall_eval))


print("done")
