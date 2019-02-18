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
import st_resnet.resnet_st_seg01_for_cue
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

            sum_att_temp = attention[cur_class,:,:].sum(axis=0)
            for i in range(num_cur_class):
                temp = np.maximum(attention[cur_class[i],:,:]*2-sum_att_temp,0.0)
                # temp = attention[cur_class[i],:,:]
                temp = resize(temp, output_size, mode='constant')
                cues_float[cur_class[i]+1] = temp

            # for background
            if labels[0]>0: # background not always exist
                cues_float[0] = resize(1.0-max_layer_mask_per_img, output_size, mode='constant')

    return cues_float

def spacial_norm_preds_only(mask, class_cur):
    temp = np.zeros(mask.shape)
    # spactial normalize
    num_class_cur = len(class_cur)
    temp_cur = mask[class_cur,:,:].reshape([num_class_cur, -1])
    # temp_min = np.min(temp_cur, axis=1, keepdims=True)
    # temp_cur = temp_cur - temp_min
    temp_cur[temp_cur<0] = 0    # manual relu
    temp_max = np.max(temp_cur, axis=1, keepdims=True)
    temp_max[temp_max == 0] = 1
    temp_cur = temp_cur / temp_max

    if class_cur[0] == 0 and num_class_cur > 1:
        # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
        temp_cur[0,np.sum(temp_cur[1:,:], axis=0)>0.1] = 0

    temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1], mask.shape[2]])
    temp = temp * 0.9 + 0.05

    return temp

# only used here
def net_outputs_to_list(layer4_feature_np, x_np, sm_mask_np, cur_class):
    # sm_mask_np need no process, already range in [0,1]
    # layer4_feature_np shape: (1,2048,41,41) range from [0, 17.xxxx], need normalization
    # x_np shape:(1,21,41,41) range from about [-56.38xxx, 130.51xxx], need normalization

    x_np_norm = np.expand_dims(spacial_norm_preds_only(x_np.squeeze(), cur_class), axis=0)
    x = layer4_feature_np.max(axis = 1, keepdims = True)
    ly4 = layer4_feature_np/x

    return [ly4, x_np_norm[:,1:,:,:]] # [ly4, sm_mask_np[:,1:,:,:]] [ly4, x_np_norm[:,1:,:,:]]



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
        save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01.pickle'
        # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
        # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
        # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_fc2conv_cpu.pth'
        model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_pure_gray_top_val_acc_my_resnet_15_15_cpu_rename_fc2conv.pth'
    elif host_name == 'ram-lab-server01':
        args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
        sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
        # save_cue_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scal_rand_gray.pickle'
        save_cue_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scal_pure_gray.pickle'
        # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
        # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
        # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_fc2conv_cpu.pth'
        model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_pure_gray_top_val_acc_my_resnet_15_15_cpu_rename_fc2conv.pth'


    output_size = [41,41]
    num_class = 21

    flag_use_cuda = torch.cuda.is_available()

    args.batch_size = 1
    net = st_resnet.resnet_st_seg01_for_cue.resnet50(pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(torch.load(model_path), strict = True)


    print(args)

    if flag_use_cuda:
        net.cuda()

    dataloader = VOCData(args)
    img_id_dic_SEC = {}
    with open(sec_id_img_name_list_dir) as f:
        for line in f:
            (key, val) = line.split()
            img_id_dic_SEC[key] = int(val)


    net.train(False)
    with torch.no_grad():
        new_cues_dict = {}
        for data in dataloader.dataloaders["train"]:
            inputs, labels, mask_gt, img, img_name = data
            if flag_use_cuda:
                inputs = inputs.cuda()

            cur_class = np.nonzero(labels.squeeze().numpy())[0]
            num_cur_class = len(cur_class)
            layer4_feature, x, sm_mask = net(inputs)

            outputs = net_outputs_to_list(layer4_feature.detach().cpu().numpy(), x.detach().cpu().numpy(), sm_mask.detach().cpu().numpy(), cur_class)
            cues_float = generate_cues(outputs, img, mask_gt, flag_classify, labels.detach().squeeze().numpy(), output_size, num_class)
            cues_hard = np.zeros(cues_float.shape, dtype='int16')


            for idx, i_class in enumerate(cur_class):
                temp = np.copy(cues_float[i_class])
                if i_class > 0:
                    thr_temp = temp.max() * 0.2
                else:
                    thr_temp = temp.max() * 0.8

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

                    thr_temp = temp_mask.max() * 0.2
                    temp_mask_hard = np.zeros(temp_mask.shape, dtype='int16')
                    temp_mask_hard[temp_mask>=thr_temp] = 1

                    plt.subplot(2,(2 + num_cur_class),num_cur_class+3); plt.imshow(temp_mask, cmap='gray'); plt.axis('off')
                    plt.subplot(2,(2 + num_cur_class),num_cur_class+4); plt.imshow(temp_mask_hard, cmap='gray'); plt.axis('off')

                plt.close('all')

            img_id_sec = img_id_dic_SEC[img_name[0] + '.png']
            cue_name = '%i_cues' % img_id_sec
            conf_name = '%i_conf' % img_id_sec

            new_cues_dict[cue_name] = np.asarray(cues_hard.nonzero(), dtype='int16')

        pickling_on = open(save_cue_path,"wb")
        pickle.dump(new_cues_dict, pickling_on)
        pickling_on.close()







