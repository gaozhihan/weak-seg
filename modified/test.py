# originally located at ROOTT_DIR/test_no_gt/test_w_pred.py
import torch
import torch.optim as optim
# import sec.sec_org_net
from modified import data_loader
from modified.arguments import get_args
from modified.save_test import SaveTest
import st_resnet.resnet_st_seg01
import time
import socket
import common_function
import numpy as np
import datetime
from skimage.transform import resize
import random
# import matplotlib.pyplot as plt
import multi_scale.STCRF_adaptive01
import st_resnet.resnet_st_more_drp

args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]
max_size = [385, 385]
flag_train = False
flag_eval = False
flag_test = True

args.rand_gray = False
args.lr = 5e-06
# args.lr = 1.25e-06 # 3.125e-07 = 1e-5*(0.5**5)
# args.CRF_model = 'adaptive_CRF'
args.origin_size = False

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_hard_snapped.pickle"
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_fc2conv_cpu.pth'
    classifier_model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu.pth'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_wsc_ft_gray_color_0221_0222_my_resnet.pth'
    classifier_model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_acc_my_resnet_multi_scale_09_01.pth'
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_all_hard_snapped_merge_0216.pickle"
    args.batch_size = 12
elif host_name == 'dycpu2.cse.ust.hk':
    args.data_dir = '/home/data/gaozhihan/project/weak-seg/weak-seg-aux/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = '/home/data/gaozhihan/project/weak-seg/weak-seg/sec/input_list.txt'
    model_path = '/home/data/gaozhihan/project/weak-seg/weak-seg-aux/models/res_wsc_ft_gray_color_0221_0222_my_resnet_cpu.pth'
    classifier_model_path = '/home/data/gaozhihan/project/weak-seg/weak-seg-aux/models/st_top_val_acc_my_resnet_09_01_cpu.pth'
    args.cues_pickle_dir = "/home/data/gaozhihan/project/weak-seg/weak-seg-aux/models/st_resnet_cue_01_all_hard_snapped_merge_0216.pickle"
    args.batch_size = 8
else:
    raise ValueError('Incorrect host name')

if args.origin_size:
    args.batch_size = 1

net = st_resnet.resnet_st_seg01.resnet50(pretrained=False, num_classes=args.num_classes)
net.load_state_dict(torch.load(model_path), strict = True)

net_classifier = st_resnet.resnet_st_more_drp.resnet50(pretrained=False, num_classes=args.num_classes)
net_classifier.load_state_dict(torch.load(classifier_model_path), strict = True)

if args.CRF_model == 'adaptive_CRF':
    st_crf_layer = multi_scale.STCRF_adaptive01.STCRFLayer(True)
else:
    st_crf_layer = data_loader.STCRFLayer(False)

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()
    net_classifier.cuda()

dataloader = data_loader.VOCData(args)

iou_obj = common_function.iou_calculator()

num_train_batch = len(dataloader.dataloaders["train"])

weight_STBCE = 0.1
weight_dec = 0.9

net.train(False)
net_classifier.train(False)

with torch.no_grad():

    train_iou = 0
    eval_iou = 0

    start = time.time()
    if flag_train:
        for data in dataloader.dataloaders["train"]:
            inputs, img = data

            # ---- random resize ------------------------------
            rand_scale = random.uniform(0.67, 1.0) #random.uniform(0.67, 1.0)
            cur_size = [round(max_size[0] * rand_scale), round(max_size[1] * rand_scale)]
            inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')

            max_val = max(max(inputs.max(), -inputs.min()), 1.0).numpy()
            img_np = np.zeros((img.shape[0], cur_size[0], cur_size[1], 3))
            img_np_temp = img.detach().numpy()/255.0

            for i in range(inputs.shape[0]):
                inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))
                img_np[i] = resize(img_np_temp[i], cur_size)

            img_np = np.round(img_np*255.0)

            if flag_use_cuda:
                inputs = torch.from_numpy(inputs_resize).cuda()
            else:
                inputs = torch.from_numpy(inputs_resize)

            sm_mask = net(inputs)
            layer4_feature, fc = net_classifier(inputs)
            preds = torch.sigmoid(fc)
            preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy().astype('float32')

            # mask_mended = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            # mask_mended = multi_scale.STCRF_adaptive01.min_mend_floor_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            mask_mended = sm_mask.detach().cpu().numpy()

            if args.CRF_model == 'adaptive_CRF':
                result_big, result_small = st_crf_layer.run(mask_mended, img_np, preds_thr_numpy)
            else:
                result_big, result_small = st_crf_layer.run(mask_mended, img_np)

            # result_small = multi_scale.STCRF_adaptive01.mend_mask_by_labels(result_small, preds_thr_numpy)
            # result_small = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(result_small, preds_thr_numpy)
            # result_small_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(result_small, preds_thr_numpy)

            # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(result_small, preds_thr_numpy)
            # plt.figure()
            # plt.imshow(np.argmax(mask_mended.squeeze(), axis=0))

            for i in range(inputs.shape[0]):
                mask_pre = np.argmax(result_big[i], axis=0)

            # evaluate mask_pre


    if flag_eval:
        for data in dataloader.dataloaders["val"]:
            inputs, img = data
            if flag_use_cuda:
                inputs = inputs.cuda()

            sm_mask = net(inputs)
            layer4_feature, fc = net_classifier(inputs)
            preds = torch.sigmoid(fc)
            preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy().astype('float32')

            # mask_mended = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            mask_mended = multi_scale.STCRF_adaptive01.min_mend_floor_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            # mask_mended = sm_mask.detach().cpu().numpy()

            if args.CRF_model == 'adaptive_CRF':
                result_big, result_small = st_crf_layer.run(mask_mended, img.numpy(), preds_thr_numpy)
            else:
                result_big, result_small = st_crf_layer.run(mask_mended, img.numpy())

            for i in range(inputs.shape[0]):
                mask_pre = np.argmax(result_big[i], axis=0)

            # evaluate mask_pre
    if flag_test:
        Saver = SaveTest(data_dir=args.data_dir)
        for data in dataloader.dataloaders["test"]:
            inputs, img = data
            if flag_use_cuda:
                inputs = inputs.cuda()

            sm_mask = net(inputs)
            layer4_feature, fc = net_classifier(inputs)
            preds = torch.sigmoid(fc)
            preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy().astype('float32')

            # mask_mended = multi_scale.STCRF_adaptive01.min_mend_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            # mask_mended = multi_scale.STCRF_adaptive01.mend_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            mask_mended = multi_scale.STCRF_adaptive01.min_mend_floor_mask_by_labels(sm_mask.detach().cpu().numpy(), preds_thr_numpy)
            # mask_mended = sm_mask.detach().cpu().numpy()

            if args.CRF_model == 'adaptive_CRF':
                result_big, result_small = st_crf_layer.run(mask_mended, img.numpy(), preds_thr_numpy)
            else:
                result_big, result_small = st_crf_layer.run(mask_mended, img.numpy())

            for i in range(inputs.shape[0]):
                mask_pre = np.argmax(result_big[i], axis=0)
                Saver.save_test(mask=mask_pre)

            # evaluate mask_pre
        print('number of test images: {}'.format(Saver.counter))
print("done")