# only test iou of the sm_mask
import torch
import torch.nn as nn
from voc_data_w_superpixel_snapped_at_sal import VOCData
import time
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize

args = get_args()
args.origin_size = False
args.model = 'SEC'
args.input_size = [321,321]
args.output_size = [41, 41]
args.need_mask_flag = True
flag_crf = True

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
now = datetime.datetime.now()
date_str = str(now.day) + '_' + str(now.month)

if host_name == 'sunting':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/snapped_attention/'
    model_path = '/home/sunting/Documents/program/weak-seg/models/sec_rename_CPU.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_01_top_val_rec_SEC_31_31.pth'
    args.super_pixel_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_attention/'
    args.batch_size = 24

net = st_01.sec_net.SEC_NN()
net.load_state_dict(torch.load(model_path), strict = False)
crf_layer = st_01.sec_net.CRFLayer(True)

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)

iou_obj = common_function.iou_calculator()

net.train(False)
with torch.no_grad():

    train_iou = 0
    eval_iou = 0

    train_seed_loss = 0.0
    train_BCE_loss = 0.0
    eval_seed_loss = 0.0
    eval_BCE_loss = 0.0
    TP_train = 0; TP_eval = 0
    T_train = 0;  T_eval = 0
    P_train = 0;  P_eval = 0

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()  # saliency_mask = saliency_mask.cuda(); attention_mask = attention_mask.cuda()

        sm_mask, preds = net(inputs)

        if flag_crf:
            sm_mask = crf_layer.run_parallel(sm_mask.detach().cpu().numpy(), img.numpy())
            for i in range(labels.shape[0]):
                temp = np.transpose(sm_mask[i,:,:,:], [1,2,0])
                temp = resize(temp, args.input_size, mode='constant')
                mask_pre = np.argmax(temp, axis=2)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)
        else:
            for i in range(labels.shape[0]):
                temp = np.transpose(sm_mask[i,:,:,:].detach().cpu().numpy(), [1,2,0])
                temp = resize(temp, args.input_size, mode='constant')
                mask_pre = np.argmax(temp, axis=2)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

    train_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())

    # for evaluation data ---------------------------------------------------------------
    for data in dataloader.dataloaders["val"]:
        inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()  # saliency_mask = saliency_mask.cuda(); attention_mask = attention_mask.cuda()

        with torch.no_grad():
            sm_mask, preds = net(inputs)

        if flag_crf:
            sm_mask = crf_layer.run_parallel(sm_mask.detach().cpu().numpy(), img.numpy())
            for i in range(labels.shape[0]):
                temp = np.transpose(sm_mask[i,:,:,:], [1,2,0])
                temp = resize(temp, args.input_size, mode='constant')
                mask_pre = np.argmax(temp, axis=2)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)
        else:
            for i in range(labels.shape[0]):
                temp = np.transpose(sm_mask[i,:,:,:].detach().cpu().numpy(), [1,2,0])
                temp = resize(temp, args.input_size, mode='constant')
                mask_pre = np.argmax(temp, axis=2)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

    eval_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    print('cur eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())

print("done")
