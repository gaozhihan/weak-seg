import torch
import torch.optim as optim
import sec.sec_org_net
import multi_scale.voc_data_mul_scale_w_cues
import st_resnet.resnet_st_seg01
import time
import socket
from arguments import get_args
import common_function
import numpy as np
import datetime
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
import multi_scale.STCRF_adaptive01

args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]
max_size = [385, 385]
flag_eval_only = True

args.rand_gray = False
args.lr = 5e-06
# args.lr = 1.25e-06 # 3.125e-07 = 1e-5*(0.5**5)
# args.CRF_model = 'adaptive_CRF'
args.origin_size = True

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    # args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    args.cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_hard_snapped.pickle"
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_fc2conv_cpu.pth'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/res_from_mul_scale_resnet_cue_01_hard_snapped_my_resnet_cpu.pth'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/res_from_mul_scale_resnet_cue_01_w_STBCE_my_resnet_cpu.pth'

elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_sec01_ws_top_val_iou_my_resnet.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_fc2conv_cpu.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_wsc_0210_my_resnet.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_from_mul_scale_resnet_cue_01_hard_snapped_my_resnet.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_from_mul_scale_ws_top_val_iou_my_resnet.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_ws_gray_0217_my_resnet.pth'
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_wsc_ft_gray_color_0221_0222_my_resnet.pth'
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/my_cues.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_cue_01_hard_snapped.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_hard_snapped.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scal_rand_gray_hard_snapped.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_gray_01_hard_snapped_merge.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scale_gray_01_hard_snapped_merge.pickle"
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_all_hard_snapped_merge_0216.pickle"
    args.batch_size = 12

if args.origin_size:
    args.batch_size = 1

net = st_resnet.resnet_st_seg01.resnet50(pretrained=False, num_classes=args.num_classes)
net.load_state_dict(torch.load(model_path), strict = True)

if args.CRF_model == 'adaptive_CRF':
    st_crf_layer = multi_scale.STCRF_adaptive01.STCRFLayer(True)
else:
    st_crf_layer = multi_scale.voc_data_mul_scale_w_cues.STCRFLayer(False)

seed_loss_layer = multi_scale.voc_data_mul_scale_w_cues.SeedingLoss()
# expand_loss_layer = sec.sec_org_net.ExpandLossLayer(flag_use_cuda)
st_constrain_loss_layer = multi_scale.voc_data_mul_scale_w_cues.STConstrainLossLayer()
st_BCE_loss_layer = multi_scale.voc_data_mul_scale_w_cues.STBCE_loss()
st_half_BCE_loss_layer = multi_scale.voc_data_mul_scale_w_cues.ST_half_BCE_loss()

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = multi_scale.voc_data_mul_scale_w_cues.VOCData(args)

iou_obj = common_function.iou_calculator()

num_train_batch = len(dataloader.dataloaders["train"])

weight_STBCE = 0.1
weight_dec = 0.9

net.train(False)

with torch.no_grad():

    train_iou = 0
    eval_iou = 0

    start = time.time()
    if not flag_eval_only:
        for data in dataloader.dataloaders["train"]:
            inputs, labels, mask_gt, img, cues = data

            # ---- random resize ------------------------------
            rand_scale = random.uniform(0.67, 1.0) #random.uniform(0.67, 1.0)
            cur_size = [round(max_size[0] * rand_scale), round(max_size[1] * rand_scale)]
            inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')
            mask_gt_resize = np.zeros((mask_gt.shape[0], cur_size[0], cur_size[1]),dtype='float32')

            max_val = max(max(inputs.max(), -inputs.min()), 1.0).numpy()
            mask_gt_f_temp = mask_gt.detach().numpy().astype('float32')
            max_val_mask = max(mask_gt_f_temp.max(), 1.0)
            mask_gt_f_temp = mask_gt_f_temp/max_val_mask
            img_np = np.zeros((img.shape[0], cur_size[0], cur_size[1], 3))
            img_np_temp = img.detach().numpy()/255.0

            for i in range(inputs.shape[0]):
                inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))
                mask_gt_resize[i] = resize(mask_gt_f_temp[i], cur_size, order=0)
                img_np[i] = resize(img_np_temp[i], cur_size)
                # resize(mask_gt[0]/mask_gt[0].max(), cur_size, order=0)*mask_gt[0].max()

            mask_gt_resize = (mask_gt_resize*max_val_mask).astype('uint8')
            img_np = np.round(img_np*255.0)

            if flag_use_cuda:
                inputs = torch.from_numpy(inputs_resize).cuda(); labels = labels.cuda() #; cues = cues.cuda()
            else:
                inputs = torch.from_numpy(inputs_resize)

            sm_mask = net(inputs)

            # mask_mended = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(sm_mask.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(sm_mask.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # mask_mended = multi_scale.STCRF_adaptive01.min_mend_floor_mask_by_labels(sm_mask.detach().cpu().numpy(), labels.detach().cpu().numpy())
            mask_mended = sm_mask.detach().cpu().numpy()

            if args.CRF_model == 'adaptive_CRF':
                result_big, result_small = st_crf_layer.run(mask_mended, img_np, labels.detach().cpu().numpy())
            else:
                result_big, result_small = st_crf_layer.run(mask_mended, img_np)

            # result_small = multi_scale.STCRF_adaptive01.mend_mask_by_labels(result_small, labels.detach().cpu().numpy())
            # result_small = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(result_small, labels.detach().cpu().numpy())
            # result_small_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(result_small, labels.detach().cpu().numpy())

            # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(result_small, labels.detach().cpu().numpy())
            # plt.figure()
            # plt.imshow(np.argmax(mask_mended.squeeze(), axis=0))

            # calculate the SEC loss
            seed_loss = seed_loss_layer(sm_mask, cues, flag_use_cuda)
            constrain_loss = st_constrain_loss_layer(result_small, sm_mask, flag_use_cuda)
            st_BCE_loss = st_BCE_loss_layer(result_small, sm_mask, labels.detach().cpu().numpy(), flag_use_cuda)
            st_half_BCE_loss = st_half_BCE_loss_layer(result_small, sm_mask, labels.detach().cpu().numpy(), flag_use_cuda)
            # st_half_BCE_loss = st_half_BCE_loss_layer(result_small_mended, sm_mask, labels.detach().cpu().numpy(), flag_use_cuda)
            # expand_loss = expand_loss_layer(sm_mask, labels)

            for i in range(labels.shape[0]):
                mask_pre = np.argmax(result_big[i], axis=0)
                iou_obj.add_iou_mask_pair(mask_gt_resize[i,:,:], mask_pre)

        train_iou = iou_obj.cal_cur_iou()
        iou_obj.iou_clear()

        time_took = time.time() - start

        # print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())
        print('cur train iou mean: ', train_iou.mean())
        weight_STBCE = weight_STBCE * 2
        weight_dec = weight_dec * weight_dec

    # if (epoch % 5 == 0):  # evaluation
    for data in dataloader.dataloaders["val"]:
        inputs, labels, mask_gt, img = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()

        sm_mask = net(inputs)


        # mask_mended = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(sm_mask.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(sm_mask.detach().cpu().numpy(), labels.detach().cpu().numpy())
        mask_mended = multi_scale.STCRF_adaptive01.min_mend_floor_mask_by_labels(sm_mask.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # mask_mended = sm_mask.detach().cpu().numpy()

        if args.CRF_model == 'adaptive_CRF':
            result_big, result_small = st_crf_layer.run(mask_mended, img.numpy(), labels.detach().cpu().numpy())
        else:
            result_big, result_small = st_crf_layer.run(mask_mended, img.numpy())

        for i in range(labels.shape[0]):
            mask_pre = np.argmax(result_big[i], axis=0)
            iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

    eval_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    print('cur eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())
    # print('cur eval iou mean: ', eval_iou.mean())

print("done")



