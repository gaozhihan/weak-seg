# foreground cues:
# 1. if one class, use max(avg saliency, attention) as attention;
# 2. if multiple foreground class, sub attention maps;
# 3. background cues: 1-max(layer1_sum, layer2_sum)

import torch
import torch.nn as nn
from voc_data_img_name import VOCData
import socket
from arguments import get_args
import matplotlib.pyplot as plt
from skimage.transform import resize
import common_function
import numpy as np
import net_saliency
import pickle

def generate_cues(outputs, img, mask_gt, flag_classify, labels, output_size, num_class):
    with torch.no_grad():
        if flag_classify:
            features = outputs[:-1]
            predictions = outputs[-1]

        else:
            features = outputs

        num_maps = len(features)
        num_img = features[0].shape[0]  # here always 1

        cur_class = np.nonzero(labels[1:])[0]
        num_cur_class = len(cur_class)
        attention = features[-1].squeeze()

        cues_float = np.zeros((num_class, output_size[0], output_size[1]))

        shape_mask = features[0][0][0].shape

        for i_img in range(num_img):
            temp_img = img[i_img]
            temp_gt_mask = mask_gt[i_img]
            temp_gt_mask[temp_gt_mask==255] = 0

            sum_layer_mask_per_img = np.zeros(shape_mask)
            max_layer_mask_per_img = np.zeros(shape_mask)
            # if flag_classify:
            #     print(idx2label[pred_hard[i_img]])

            # for foreground classes
            for i_map in range(num_maps):
                temp_feature = features[i_map][i_img].sum(axis=0)  #[1:] means no background
                # normalization
                if temp_feature.max() != 0:
                    temp_feature = (temp_feature - temp_feature.min())/temp_feature.max()

                temp = resize(temp_feature, shape_mask, mode='constant')
                sum_layer_mask_per_img = sum_layer_mask_per_img + temp
                max_layer_mask_per_img = np.maximum(max_layer_mask_per_img, temp)

            if sum_layer_mask_per_img.max() > 0:
                sum_layer_mask_per_img = (sum_layer_mask_per_img - sum_layer_mask_per_img.min())/sum_layer_mask_per_img.max()

            if max_layer_mask_per_img.max() > 0:
                max_layer_mask_per_img = (max_layer_mask_per_img - max_layer_mask_per_img.min())/max_layer_mask_per_img.max()

            if num_cur_class > 1:
                sum_att_temp = attention[cur_class,:,:].sum(axis=0)
                for i in range(num_cur_class):
                    temp = np.maximum(attention[cur_class[i],:,:]*2-sum_att_temp,0.0)
                    temp = resize(temp, output_size, mode='constant')
                    cues_float[cur_class[i]+1] = temp

            else:
                temp1 = resize(attention[cur_class[0],:,:], output_size, mode='constant')
                temp2 = resize(sum_layer_mask_per_img, output_size, mode='constant')
                cues_float[cur_class[0]+1] = np.maximum(temp1, temp2)

            # for background
            if labels[0]>0: # background not always exist
                cues_float[0] = resize(1.0-max_layer_mask_per_img, output_size, mode='constant')

    return cues_float

# ----------------------------------------------------------------
def snap_to_superpixel(saliency_mask, img, seg):

    if img.max() > 1:
        img = img / 255.0

    img_shape = img.shape[:2]
    saliency_mask_rez = resize(saliency_mask, img_shape, mode='constant')
    saliency_mask_snapped = np.zeros(img_shape)

    num_seg = int(seg.max()) + 1

    if num_seg > 50:
        for i_seg in range(num_seg):
            cur_seg = (seg == i_seg)
            cur_saliency_region = saliency_mask_rez[cur_seg]
            saliency_mask_snapped[cur_seg] = cur_saliency_region.mean()
    else:
        saliency_mask_snapped = saliency_mask_rez

    return saliency_mask_snapped


def snap_cues_to_superpixel(cues_float, img, labels, super_pixel_path):

    num_class = len(labels)
    super_pixel = np.load(super_pixel_path)
    cur_class = np.nonzero(labels)[0]

    snapped_cues_float = np.zeros((num_class, img.shape[0], img.shape[1]))
    for i_class in cur_class:
        snapped_cues_float[i_class,:,:] = snap_to_superpixel(cues_float[i_class,:,:], img, super_pixel)

    return snapped_cues_float


def thr_mask_conflict_resolve(snapped_cues_float, labels):

    thr_fg_ratio = 0.25
    thr_bg_ratio = 0.85
    snapped_cues_hard = np.zeros(snapped_cues_float.shape, dtype='int16')
    cur_class = np.nonzero(labels)[0]
    for i_class in cur_class:
        if i_class > 0:
            thr_val = snapped_cues_float[i_class].max() * thr_fg_ratio
        else:
            thr_val = snapped_cues_float[i_class].max() * thr_bg_ratio

        temp = snapped_cues_float[i_class]
        temp[temp < thr_val] = 0
        temp[temp >= thr_val] = 1
        snapped_cues_hard[i_class] = temp.astype('int16')

    # resolve conflict, just background give way for foreground
    temp = snapped_cues_hard[1:,:,:].sum(axis=0)
    temp_idx = temp > 0
    temp = snapped_cues_hard[0]
    temp[temp_idx] = 0
    snapped_cues_hard[0] = temp

    return snapped_cues_hard

#------------------ the main function -------------------------
if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.test_flag = True

    args.input_size = [321,321] #[224,224]
    args.output_size = [41, 41] #[29,29]
    args.origin_size = False
    args.color_vote = True
    args.fix_CRF_itr = False
    args.preds_only = True
    args.CRF_model = 'my_CRF' # SEC_CRF or my_CRF
    flag_classify = False
    host_name = socket.gethostname()

    flag_visualization = False

    if host_name == 'sunting':
        args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
        sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
        save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_01_soft_snapped_w_conf.pickle'
        super_pixel_path = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    elif host_name == 'ram-lab-server01':
        args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
        sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
        save_cue_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_cue_01_soft_snapped_w_conf.pickle'
        super_pixel_path = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/super_pixel/'

    output_size = [41,41]
    num_class = 21

    flag_use_cuda = torch.cuda.is_available()

    args.batch_size = 1
    net_decouple = net_saliency.decouple_net_saliency_with_pred(flag_classify)

    print(args)

    if flag_use_cuda:
        net_decouple.cuda()

    dataloader = VOCData(args)
    img_id_dic_SEC = {}
    with open(sec_id_img_name_list_dir) as f:
        for line in f:
            (key, val) = line.split()
            img_id_dic_SEC[key] = int(val)


    net_decouple.train(False)
    with torch.no_grad():
        new_cues_dict = {}
        for data in dataloader.dataloaders["train"]:
            inputs, labels, mask_gt, img, img_name = data
            if flag_use_cuda:
                inputs = inputs.cuda()

            outputs, preds = net_decouple(inputs)

            # calculate confidence
            conf = preds[labels.cpu().detach().squeeze().numpy()>0]
            # print(conf)

            cues_float = generate_cues(outputs, img, mask_gt, flag_classify, labels.detach().squeeze().numpy(), output_size, num_class)

            snapped_cues_float = snap_cues_to_superpixel(cues_float, img.cpu().detach().squeeze().numpy(), labels.cpu().detach().squeeze().numpy(), super_pixel_path+img_name[0]+'.npy')
            snapped_cues_float = np.transpose(resize(np.transpose(snapped_cues_float, [1,2,0]), output_size, mode='constant'),[2,0,1])
            snapped_cues_hard = thr_mask_conflict_resolve(snapped_cues_float, labels.cpu().detach().squeeze().numpy())

            cur_class = np.nonzero(labels.cpu().detach().squeeze().numpy())[0]
            num_cur_class = len(cur_class)

            if flag_visualization:
                temp_img = img.squeeze().numpy()
                temp_gt_mask = mask_gt.squeeze().numpy()
                temp_gt_mask[temp_gt_mask==255] = 0
                plt.subplot(3,(2 + num_cur_class),1); plt.imshow(temp_img/255); plt.title('Input image'); plt.axis('off')
                plt.subplot(3,(2 + num_cur_class),2); plt.imshow(temp_gt_mask); plt.title('true mask'); plt.axis('off')

                for idx, i_class in enumerate(cur_class):
                    plt.subplot(3,(2 + num_cur_class),3+idx); plt.imshow(cues_float[i_class], cmap='gray'); plt.axis('off')
                    plt.subplot(3,(2 + num_cur_class),num_cur_class+5+idx); plt.imshow(snapped_cues_float[i_class], cmap='gray'); plt.axis('off')
                    plt.subplot(3,(2 + num_cur_class),num_cur_class*2+7+idx); plt.imshow(snapped_cues_hard[i_class]); plt.axis('off')

                plt.close('all')

            img_id_sec = img_id_dic_SEC[img_name[0] + '.png']
            cue_name = '%i_cues' % img_id_sec
            conf_name = '%i_conf' % img_id_sec

            new_cues_dict[cue_name] = np.asarray(snapped_cues_hard.nonzero(), dtype='int16')
            new_cues_dict[conf_name] = conf

        pickling_on = open(save_cue_path,"wb")
        pickle.dump(new_cues_dict, pickling_on)
        pickling_on.close()







