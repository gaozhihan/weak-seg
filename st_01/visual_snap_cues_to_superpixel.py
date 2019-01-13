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


def snap_to_superpixel(saliency_mask, img, seg):

    if img.max() > 1:
        img = img / 255.0

    img_shape = img.shape[:2]
    saliency_mask_rez = resize(saliency_mask, img_shape, mode='constant')
    saliency_mask_snapped = np.zeros(img_shape)

    num_seg = int(seg.max()) + 1

    for i_seg in range(num_seg):
        cur_seg = (seg == i_seg)
        cur_saliency_region = saliency_mask_rez[cur_seg]
        saliency_mask_snapped[cur_seg] = cur_saliency_region.mean()

    return saliency_mask_snapped


def snap_cues_to_superpixel(img_np, labels_np, super_pixel_np, cues_np):

    if img_np.max() > 1:
            img_np = img_np / 255.0

    cur_class = np.nonzero(labels_np)[0]
    num_cur_class = len(cur_class)

    snapped_cues = np.zeros((num_cur_class, img_np.shape[0], img_np.shape[1]))

    for i in range(num_cur_class):
            # plt.subplot(2,(2 + num_cur_class),3+i); plt.imshow(attention[cur_class[i],:,:]); plt.title('raw attention {}'.format(cur_class[i])); plt.axis('off')
            snapped_cues[i,:,:] = snap_to_superpixel(cues_np[cur_class[i],:,:], img_np.squeeze(), super_pixel_np)

    return snapped_cues



if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.test_flag = True
    args = get_args()
    args.origin_size = False
    args.model = 'SEC'
    args.input_size = [321,321]
    args.output_size = [41, 41]
    args.need_mask_flag = True
    flag_view_thresholded_at = True
    thr_ratio = 0.3

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
                    cues_np = cues.squeeze().numpy()
                    labels_np = labels.squeeze().numpy()
                    super_pixel_np = super_pixel.squeeze().numpy()

                    snapped_cues = snap_cues_to_superpixel(img_np.squeeze(), labels_np, super_pixel_np, cues_np)

                    cur_class = np.nonzero(labels_np)[0]
                    num_cur_class = len(cur_class)

                    plt.subplot(2, num_cur_class+1, 1); plt.imshow(img_np); plt.title('img')
                    temp = mask_gt.squeeze().numpy()
                    temp[temp == 255] = 0
                    plt.subplot(2, num_cur_class+1, num_cur_class+2); plt.imshow(temp); plt.title('img')
                    for idx, i_class in enumerate(cur_class):
                        plt.subplot(2, num_cur_class+1, 2+idx); plt.imshow(cues_np[i_class])
                        if flag_view_thresholded_at:
                            temp = snapped_cues[idx]
                            thr = temp.max() * thr_ratio
                            temp[temp<thr] = 0
                            temp[temp>=thr] = 1
                            plt.subplot(2, num_cur_class+1, num_cur_class+3+idx); plt.imshow(temp)
                        else:
                            plt.subplot(2, num_cur_class+1, num_cur_class+3+idx); plt.imshow(snapped_cues[idx]) # view

                    plt.close('all')

            else:  # evaluation
                # for data in dataloader.dataloaders["val"]:
                #     inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data

                    plt.close('all')
