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
import os
from PIL import Image
from torchvision import transforms
import pickle


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
    flag_view_thresholded = True
    thr_ratio = 0.3
    input_size = [321,321]
    output_size = [41, 41]
    num_classes = 21

    data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    # cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_02_w_conf.pickle"

    with open(os.path.join(data_dir, "ImageSets", 'train.txt'),"r") as f:
        all_files = f.readlines()
        file_list = [item.split()[0] for item in all_files]

    cues_data_SEC = pickle.load(open(cues_pickle_dir,'rb'))

    num_img = len(file_list)
    img_transforms = transforms.Compose([
                        transforms.Resize(input_size),
                        transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                    ])


    for idx in range(num_img):
        img_name = os.path.join(data_dir, "images", file_list[idx]+".png")
        mask_name = os.path.join(data_dir, "segmentations", file_list[idx]+".png")
        super_pixel_name = super_pixel_dir + file_list[idx] + '.npy'

        img = Image.open(img_name)
        img_resize = img.resize(input_size)
        img_array = np.array(img_resize).astype(np.float32)
        mask = np.array(Image.open(mask_name).resize(input_size))

        # img_ts = img_transforms(img)
        # img_ts = img_ts.float()*255.0

        super_pixel = np.load(super_pixel_name)

        with open(os.path.join(data_dir, "ImageSets", "class_label.pickle"),"rb") as f:
            label_dict = pickle.load(f)

        img_id_dic_SEC = {}
        with open(sec_id_img_name_list_dir) as f:
            for line in f:
                (key, val) = line.split()
                img_id_dic_SEC[key] = int(val)

        label = np.zeros(num_classes, dtype=np.float32)
        for item in label_dict[file_list[idx]]:
            label[item] = 1

        # ---- get cues from image name
        img_id_sec = img_id_dic_SEC[file_list[idx]+".png"]
        cues = cues_data_SEC['%i_cues' % img_id_sec]
        cues_np = np.zeros([num_classes, output_size[0], output_size[1]], dtype='float32')
        cues_np[cues[0], cues[1], cues[2]] = 1.0

        snapped_cues = snap_cues_to_superpixel(img_array, label, super_pixel.astype('int32'), cues_np)

        # ----- visualization --------------------------
        cur_class = np.nonzero(label)[0]
        num_cur_class = len(cur_class)

        plt.subplot(2, num_cur_class+1, 1); plt.imshow(img_resize); plt.title('img'); plt.axis('off')
        temp = mask
        temp[temp == 255] = 0
        plt.subplot(2, num_cur_class+1, num_cur_class+2); plt.imshow(temp); plt.title('gt'); plt.axis('off')
        for idx, i_class in enumerate(cur_class):
            plt.subplot(2, num_cur_class+1, 2+idx); plt.imshow(cues_np[i_class]); plt.title('org cues'); plt.axis('off')
            if flag_view_thresholded:
                temp = snapped_cues[idx]
                thr = temp.max() * thr_ratio
                temp[temp<thr] = 0
                temp[temp>=thr] = 1
                plt.subplot(2, num_cur_class+1, num_cur_class+3+idx); plt.imshow(temp); plt.title('snapped cues'); plt.axis('off')
            else:
                plt.subplot(2, num_cur_class+1, num_cur_class+3+idx); plt.imshow(snapped_cues[idx]); plt.title('snapped cues'); plt.axis('off') # view

        plt.close('all')



