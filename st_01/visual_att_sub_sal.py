# only test iou of the sm_mask
import torch
import torch.nn as nn
from voc_data_w_superpixel_snapped_at_sal import VOCData
import time
import socket
import st_01.sec_net_for_sal_att
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize
import matplotlib.pyplot as plt

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
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/vgg16-397923af.pth' # 'vgg16'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_01_multi_scale_top_val_acc_SEC_rename_for_att_sal.pth'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/SEC_rename_for_att_sal.pth'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_01_top_val_rec_SEC_31_for_att_sal.pth'
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

net = st_01.sec_net_for_sal_att.SEC_NN()
net.load_state_dict(torch.load(model_path), strict = True)

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)

net.train(False)
with torch.no_grad():

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()  # saliency_mask = saliency_mask.cuda(); attention_mask = attention_mask.cuda()

        # obtain saliency and attention
        conv_f0, sm_mask, preds = net(inputs)

        conv_f0_np = conv_f0.squeeze().numpy()
        sm_mask_np = sm_mask.squeeze().numpy()
        img_np = img.squeeze().numpy().astype('uint8')
        mask_gt_np = mask_gt.squeeze().numpy()
        mask_gt_np[mask_gt_np==255] = 0

        sal_conv_f0 = np.sum(conv_f0_np, axis=0)
        if sal_conv_f0.max() != 0:
            sal_conv_f0 = (sal_conv_f0 - sal_conv_f0.min())/sal_conv_f0.max()

        sal_sm_mask = np.sum(sm_mask_np[1:,:,:], axis=0) # important! do NOT include background
        if sal_sm_mask.max() != 0:
            sal_sm_mask = (sal_sm_mask - sal_sm_mask.min())/sal_sm_mask.max()

        cur_class = np.nonzero(labels.squeeze().numpy()[1:])[0]
        num_cur_class = len(cur_class)

        # visualization --------------------------
        # attention
        plt.figure(); plt.title('attention')

        if num_cur_class > 1: # more than one class present, can use subtract attention
            sal_cur_class = sm_mask_np[cur_class+1,:,:].sum(axis=0)

            for idx, i_class in enumerate(cur_class):
                plt.subplot(2,num_cur_class,idx+1); plt.imshow(sm_mask_np[i_class+1], cmap='gray'); plt.axis('off')
                sub_att_temp = np.maximum(sm_mask_np[i_class+1]* 2 - sal_cur_class, 0.0)
                plt.subplot(2,num_cur_class,num_cur_class+idx+1); plt.imshow(sub_att_temp, cmap='gray'); plt.axis('off')

        else:
            for idx, i_class in enumerate(cur_class):
                plt.subplot(1,num_cur_class,idx+1); plt.imshow(sm_mask_np[i_class+1], cmap='gray'); plt.axis('off')

        # saliency
        plt.figure()
        plt.subplot(1,6,1); plt.imshow(img_np); plt.title('img'); plt.axis('off')
        plt.subplot(1,6,2); plt.imshow(mask_gt_np); plt.title('gt'); plt.axis('off')
        plt.subplot(1,6,3); plt.imshow(sal_conv_f0, cmap='gray'); plt.title('sal f4'); plt.axis('off')
        plt.subplot(1,6,4); plt.imshow(sal_sm_mask, cmap='gray'); plt.title('sal sm'); plt.axis('off')
        plt.subplot(1,6,5); plt.imshow(np.maximum(sal_conv_f0,sal_sm_mask), cmap='gray'); plt.title('sal max'); plt.axis('off')
        plt.subplot(1,6,6); plt.imshow((sal_conv_f0+sal_sm_mask)/2, cmap='gray'); plt.title('sal avg'); plt.axis('off')

        plt.close('all')


    # for evaluation data ---------------------------------------------------------------
    for data in dataloader.dataloaders["val"]:
        inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()  # saliency_mask = saliency_mask.cuda(); attention_mask = attention_mask.cuda()

        conv_f0, sm_mask, preds = net(inputs)





print("done")
