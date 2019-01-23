# modification compared with 'gen_save_st_cue_01':
# aware that sometimes one class can be a small obj, using saliency will include a lot of noise, so check first:
# for one class case, replace max(avg saliency, attention) as attention if one of the following condition satisfies:
# 1. in hard mask, iou > thr; 2. find the rectangles contain the activation in the two masks, if their iou > thr
# another change is the general threshold to generate foreground cues_hard from 0.3 to 0.25 times max val

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

def smart_comb(att, sal):
    att_hard = np.zeros(att.shape)
    sal_hard = np.zeros(sal.shape)

    thr_temp = att.max() * 0.25
    att_hard[att>thr_temp] = 1

    thr_temp = sal.max() * 0.25
    sal_hard[sal>thr_temp] = 1

    # iou
    temp = att_hard + sal_hard
    intercection = (temp > 1).sum()
    union = (temp > 0).sum()

    if intercection/union > 0.4:
        return sal
    else:
        # bbx
        temp_idx = np.where(att_hard > 0)
        att_bbox_mask = np.zeros(att_hard.shape)
        att_bbox_mask[temp_idx[0].min():temp_idx[0].max(), temp_idx[1].min():temp_idx[1].max()] = 1

        temp_idx = np.where(sal_hard > 0)
        sal_bbox_mask = np.zeros(sal_hard.shape)
        sal_bbox_mask[temp_idx[0].min():temp_idx[0].max(), temp_idx[1].min():temp_idx[1].max()] = 1

        temp = att_hard + sal_hard
        intercection = (temp > 1).sum()
        union = (temp > 0).sum()

        if intercection/union > 0.65:
            return sal
        else:
            return att


def smart_comb_test(att, sal):
    att_hard = np.zeros(att.shape)
    sal_hard = np.zeros(sal.shape)

    thr_temp = att.max() * 0.25
    att_hard[att>thr_temp] = 1

    thr_temp = sal.max() * 0.25
    sal_hard[sal>thr_temp] = 1

    # iou
    temp = att_hard + sal_hard
    intercection = (temp > 1).sum()
    union = (temp > 0).sum()
    print('iou is: {:.4f}'.format(intercection/union))

    # bbx
    temp_idx = np.where(att_hard > 0)
    att_bbox_mask = np.zeros(att_hard.shape)
    att_bbox_mask[temp_idx[0].min():temp_idx[0].max(), temp_idx[1].min():temp_idx[1].max()] = 1

    temp_idx = np.where(sal_hard > 0)
    sal_bbox_mask = np.zeros(sal_hard.shape)
    sal_bbox_mask[temp_idx[0].min():temp_idx[0].max(), temp_idx[1].min():temp_idx[1].max()] = 1

    temp = att_bbox_mask + sal_bbox_mask
    intercection = (temp > 1).sum()
    union = (temp > 0).sum()

    print('bbx iou is: {:.4f}'.format(intercection/union))

    return sal


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
                temp3 = np.maximum(temp1, temp2)
                cues_float[cur_class[0]+1] = smart_comb(temp1, temp3)
                # cues_float[cur_class[0]+1] = smart_comb_test(temp1, temp3)

            # for background
            if labels[0]>0: # background not always exist
                cues_float[0] = resize(1.0-max_layer_mask_per_img, output_size, mode='constant')

    return cues_float


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
        save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_02.pickle'
    elif host_name == 'ram-lab-server01':
        args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
        sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
        save_cue_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_cue_02.pickle'


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

            cur_class = np.nonzero(labels.squeeze().numpy())[0]
            num_cur_class = len(cur_class)
            outputs, preds = net_decouple(inputs)

            # calculate confidence
            conf = preds[labels.cpu().detach().squeeze().numpy()>0]
            # print(conf)

            cues_float = generate_cues(outputs, img, mask_gt, flag_classify, labels.detach().squeeze().numpy(), output_size, num_class)
            cues_hard = np.zeros(cues_float.shape, dtype='int16')


            for idx, i_class in enumerate(cur_class):
                temp = np.copy(cues_float[i_class])
                if i_class > 0:
                    thr_temp = temp.max() * 0.25
                else:
                    thr_temp = temp.max() * 0.92

                temp[temp>thr_temp] = 1
                temp[temp<=thr_temp] = 0
                cues_hard[i_class] = temp.astype('int16')

            if flag_visualization:
                temp_img = img.squeeze().numpy()
                temp_gt_mask = mask_gt.squeeze().numpy()
                temp_gt_mask[temp_gt_mask==255] = 0
                plt.figure()
                plt.subplot(2,(2 + num_cur_class),1); plt.imshow(temp_img/255); plt.title('Input image'); plt.axis('off')
                plt.subplot(2,(2 + num_cur_class),2); plt.imshow(temp_gt_mask); plt.title('true mask'); plt.axis('off')

                for idx, i_class in enumerate(cur_class):
                    plt.subplot(2,(2 + num_cur_class),3+idx); plt.imshow(cues_float[i_class], cmap='gray'); plt.axis('off')
                    plt.subplot(2,(2 + num_cur_class),num_cur_class+5+idx); plt.imshow(cues_hard[i_class], cmap='gray'); plt.axis('off')

                fg_idx = cur_class.nonzero()[0]
                if len(fg_idx) == 1:
                    if flag_classify:
                        temp_mask = outputs[-2].squeeze()[cur_class[fg_idx[0]]-1]
                    else:
                        temp_mask = outputs[-1].squeeze()[cur_class[fg_idx[0]]-1]

                    thr_temp = temp_mask.max() * 0.25
                    temp_mask_hard = np.zeros(temp_mask.shape, dtype='int16')
                    temp_mask_hard[temp_mask>thr_temp] = 1

                    plt.subplot(2,(2 + num_cur_class),num_cur_class+3); plt.imshow(temp_mask, cmap='gray'); plt.axis('off')
                    plt.subplot(2,(2 + num_cur_class),num_cur_class+4); plt.imshow(temp_mask_hard, cmap='gray'); plt.axis('off')

                plt.close('all')

            img_id_sec = img_id_dic_SEC[img_name[0] + '.png']
            cue_name = '%i_cues' % img_id_sec
            conf_name = '%i_conf' % img_id_sec

            new_cues_dict[cue_name] = np.asarray(cues_hard.nonzero(), dtype='int16')
            new_cues_dict[conf_name] = conf

        pickling_on = open(save_cue_path,"wb")
        pickle.dump(new_cues_dict, pickling_on)
        pickling_on.close()







