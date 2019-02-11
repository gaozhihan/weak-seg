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
import matplotlib.pyplot as plt
from skimage.transform import resize
import multi_scale.STCRF_adaptive01


args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]

args.CRF_model = 'adaptive_CRF'

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    args.batch_size = 24

if args.CRF_model == 'adaptive_CRF':
    st_crf_layer = multi_scale.STCRF_adaptive01.STCRFLayer(False)
else:
    st_crf_layer = multi_scale.voc_data_mul_scale_w_cues.STCRFLayer(True)

print(args)

dataloader = VOCData(args)

iou_obj = common_function.iou_calculator()

num_train_batch = len(dataloader.dataloaders["train"])

with torch.no_grad():

    train_iou = 0
    eval_iou = 0

    start = time.time()

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, cues = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda(); cues = cues.cuda()

        img_np = img.detach().numpy()
        mask_cue = cues.detach().numpy()*0.96
        mask_cue[mask_cue==0] = 0.001
        for i in range(labels.shape[0]):
            mask_cue[i,:,:,:]/mask_cue[i,:,:,:].sum(axis=0)

        if args.CRF_model == 'adaptive_CRF':
            result_big, result_small = st_crf_layer.run(mask_cue, img_np, labels.detach().cpu().numpy())
        else:
            result_big, result_small = st_crf_layer.run(mask_cue, img_np)

        for i in range(labels.shape[0]):
            mask_pre = np.argmax(result_big[i], axis=0)
            iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

            plt.figure()
            plt.subplot(1,5,1); plt.imshow(img_np[i]/255.0); plt.title('Input image'); plt.axis('off')
            temp = mask_gt[i,:,:].numpy()
            temp[temp==255] = 0
            plt.subplot(1,5,2); plt.imshow(mask_gt[i,:,:].numpy()); plt.title('gt'); plt.axis('off')
            plt.subplot(1,5,3); plt.imshow(np.argmax(mask_cue[i], axis=0)); plt.title('cues'); plt.axis('off')
            plt.subplot(1,5,4); plt.imshow(mask_cue[i,0,:,:]); plt.title('bg cues'); plt.axis('off')
            plt.subplot(1,5,5); plt.imshow(mask_pre); plt.title('af crf'); plt.axis('off')
            plt.close('all')

    train_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())


print("done")



