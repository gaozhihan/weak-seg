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



# assume batch size = 1
def snap_anntention_to_superpixel(outputs, img, mask_gt, flag_classify, labels, super_pixel_path):
    with torch.no_grad():
        mask_gt[mask_gt==255] = 0
        mask_gt = mask_gt.squeeze()
        if img.max() > 1:
            img = img.squeeze() / 255.0

        if flag_classify:
            features = outputs[:-1]
            predictions = outputs[-1]

        else:
            features = outputs

        cur_class = np.nonzero(labels[1:])[0]
        # print(cur_class)
        num_cur_class = len(cur_class)
        attention = features[-1].squeeze()
        super_pixel = np.load(super_pixel_path)
        snapped_attention = np.zeros((num_cur_class, img.shape[0], img.shape[1]))

        # plt.subplot(2,(2 + num_cur_class),1); plt.imshow(img); plt.title('Input image'); plt.axis('off')
        # plt.subplot(2,(2 + num_cur_class),3 + num_cur_class); plt.imshow(mask_gt); plt.title('true mask'); plt.axis('off')
        # plt.subplot(2,(2 + num_cur_class),2); plt.imshow(super_pixel); plt.title('super pixel'); plt.axis('off')
        # plt.subplot(2,(2 + num_cur_class),4 + num_cur_class); plt.imshow(mark_boundaries(img,super_pixel)); plt.title('boundary'); plt.axis('off')

        for i in range(num_cur_class):
            # plt.subplot(2,(2 + num_cur_class),3+i); plt.imshow(attention[cur_class[i],:,:]); plt.title('raw attention {}'.format(cur_class[i])); plt.axis('off')
            snapped_attention[i,:,:] = snap_to_superpixel(attention[cur_class[i],:,:], img, super_pixel)
            # plt.subplot(2,(2 + num_cur_class),num_cur_class+5+i); plt.imshow(snapped_attention[i,:,:]); plt.title('snapped attention {}'.format(cur_class[i])); plt.axis('off')

        # plt.close('all')
        return  snapped_attention


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
