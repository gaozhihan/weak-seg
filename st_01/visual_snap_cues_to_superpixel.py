# only test iou of the sm_mask
import torch
import torch.nn as nn
from compare_cues import VOCData
import time
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize
import matplotlib.pyplot as plt

args = get_args()
args.need_mask_flag = True
args.test_flag = True
args = get_args()
args.origin_size = False
args.model = 'SEC'
args.input_size = [321,321]
args.output_size = [41, 41]
args.need_mask_flag = True

flag_use_cuda = torch.cuda.is_available()

args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
args.batch_size = 1

print(args)

dataloader = VOCData(args)

with torch.no_grad():

    for phase in ['train', 'val']:
        if phase == 'train':

            for data in dataloader.dataloaders["train"]:
                inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask, cues = data

                mask_gt_np = mask_gt.squeeze().numpy()
                img_np = img.squeeze().numpy().astype('uint8')
                temp_np = attention_mask.squeeze().numpy()
                thr_value = temp_np.max()*0.3
                temp_np[temp_np < thr_value] = 0
                attention_mask_np = np.argmax(temp_np, axis=0)

                temp_np = cues.squeeze().numpy()
                cues_np = np.argmax(temp_np, axis=0)

                plt.subplot(1,4,1); plt.imshow(img_np); plt.title('Input image')
                plt.subplot(1,4,2); plt.imshow(mask_gt_np); plt.title('gt')
                plt.subplot(1,4,3); plt.imshow(attention_mask_np); plt.title('at mask')
                plt.subplot(1,4,4); plt.imshow(cues_np); plt.title('cues')

                plt.close('all')

        else:  # evaluation
            # for data in dataloader.dataloaders["val"]:
            #     inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data



                plt.close('all')
