import torch
import torch.optim as optim
from sec.sec_data_loader_no_rand import VOCData
import sec.sec_org_net
import st_resnet.resnet_st_seg01
import time
import socket
from arguments import get_args
import common_function
import numpy as np
import datetime
from skimage.transform import resize
import matplotlib.pyplot as plt


def sm_mask_from_cues(cues_np, labels_np, flag_use_label):
    # assume inputs are binary cues

    tatal_val_for_cur_class = 0.99
    val_for_cue = 0.8
    val_for_cur_non_cue = 0.2
    val_for_non_class = 1e-4

    if flag_use_label:
        cur_class_idx = labels_np.nonzero()[0]
        non_cur_class = labels_np < 1

        # initialize with non cur class val
        sm_mask = np.ones(cues_np.shape) * val_for_non_class
        # for cur_class
        # initialize with average cur class no cue pixel
        val_avg_cur_class = tatal_val_for_cur_class/len(cur_class_idx)
        sm_mask[cur_class_idx,:,:] = val_for_cur_non_cue
        pixel_w_cue = np.sum(cues_np, axis=0)>0
        pixel_wo_cue = np.logical_not(pixel_w_cue)
        for i_class in cur_class_idx:
            idx_cur_cue = cues_np[i_class]>0
            temp = sm_mask[i_class]
            temp[pixel_wo_cue] = val_avg_cur_class
            temp[idx_cur_cue] = val_for_cue
            sm_mask[i_class] = temp

    return  sm_mask


if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.model = 'my_resnet'
    args.input_size = [321,321]
    args.output_size = [41, 41]
    flag_use_label = True
    flag_visualization = False

    host_name = socket.gethostname()
    flag_use_cuda = torch.cuda.is_available()
    date_str = str(datetime.datetime.now().day)

    if host_name == 'sunting':
        args.batch_size = 2
        args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
        args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
        # args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
        args.cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_all_hard_snapped_merge_0216.pickle"
        # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
        # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
    elif host_name == 'sunting-ThinkCentre-M90':
        args.batch_size = 2
        args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
        args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
        args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
        model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
    elif host_name == 'ram-lab-server01':
        args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
        args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
        model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
        # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_sec01_ws_top_val_iou_my_resnet.pth'
        # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
        # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
        # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/my_cues.pickle"
        args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_cue_02_hard_snapped.pickle"
        args.batch_size = 18


    # net = st_resnet.resnet_st_seg01.resnet50(pretrained=False, num_classes=args.num_classes)
    # net.load_state_dict(torch.load(model_path), strict = True)

    args.batch_size = 1
    st_crf_layer = sec.sec_org_net.STCRFLayer(True)

    print(args)

    # if flag_use_cuda:
    #     net.cuda()

    dataloader = VOCData(args)

    max_iou = 0
    iou_obj = common_function.iou_calculator()

    num_train_batch = len(dataloader.dataloaders["train"])

    # net.train(False)
    with torch.no_grad():

        train_iou = 0
        eval_iou = 0
        counter = 0
        for data in dataloader.dataloaders["train"]:
            inputs, labels, mask_gt, img, cues = data
            # if flag_use_cuda:
            #     inputs = inputs.cuda(); labels = labels.cuda(); cues = cues.cuda()
            #
            # sm_mask = net(inputs)
            sm_mask = sm_mask_from_cues(cues.squeeze().numpy(), labels.squeeze().numpy(), flag_use_label)
            result_big, result_small = st_crf_layer.run(np.expand_dims(sm_mask, axis=0), img.numpy())

            for i in range(labels.shape[0]):
                mask_pre = np.argmax(result_big[i], axis=0)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

            if flag_visualization:
                for i in range(labels.shape[0]):
                    result_pre = np.argmax(result_big[i], axis=0)
                    plt.subplot(1,4,1); plt.imshow(img[i]/255); plt.title('Input image'); plt.axis('off')
                    temp = mask_gt[i,:,:].numpy()
                    temp[temp==255] = 0
                    plt.subplot(1,4,2); plt.imshow(mask_gt[i,:,:].numpy()); plt.title('gt'); plt.axis('off')
                    plt.subplot(1,4,3); plt.imshow(np.argmax(sm_mask,axis=0)); plt.title('sm mask'); plt.axis('off')
                    plt.subplot(1,4,4); plt.imshow(result_pre); plt.title('sm mask crf'); plt.axis('off')
                    plt.close("all")

            counter = counter + 1

        train_iou = iou_obj.cal_cur_iou()
        iou_obj.iou_clear()

        print('cur train iou mean: ', train_iou.mean())

    print("done")



