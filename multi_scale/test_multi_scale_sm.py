import torch
import torch.nn as nn
from sec.sec_data_loader_no_rand import VOCData
import socket
import st_resnet.resnet_st_seg01
from arguments import get_args
import datetime
import numpy as np
import random
from skimage.transform import resize
import matplotlib.pyplot as plt
import st_01.sec_net
import common_function

args = get_args()
args.need_mask_flag = False
args.model = 'my_resnet'
args.input_size = [385,385]
max_size = [385, 385]
uni_sm_mask_size = [41, 41]
args.output_size = [41, 41]
crf_size = [321, 321]
args.batch_size = 1
random.uniform(0.67, 1.0)
# rand_scale = [0.67, 0.8337, 1.0]
rand_scale = [0.6, 0.8, 1.0]
flag_crf = True
flag_avg = True

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
now = datetime.datetime.now()
date_str = str(now.day) + '_' + str(now.day)

args.batch_size = 1
if host_name == 'sunting':
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/resnet50_feat.pth'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/res_from_mul_scale_resnet_cue_01_hard_snapped_my_resnet_cpu.pth'
elif host_name == 'sunting-ThinkCentre-M90':
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_01_top_val_rec_SEC_31_31.pth'
    args.super_pixel_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_attention/'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_sec01_ws_top_val_iou_my_resnet.pth'

net = st_resnet.resnet_st_seg01.resnet50(pretrained=False, num_classes=args.num_classes)
net.load_state_dict(torch.load(model_path), strict = False)

if args.loss == 'BCELoss':
    criterion = nn.BCELoss()
elif args.loss == 'MultiLabelSoftMarginLoss':
    criterion = nn.MultiLabelSoftMarginLoss()

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)
num_scale = len(rand_scale)

crf_layer = st_01.sec_net.CRFLayer(False)
iou_obj = common_function.iou_calculator()

with torch.no_grad():
    net.train(False)
    for phase in ['train', 'val']:
        if phase == 'train':

            for data in dataloader.dataloaders["train"]:
                inputs, labels, mask_gt, img, cues = data

                labels_np = labels.squeeze().numpy()
                num_class = int(labels_np.sum())
                cur_class = labels_np.nonzero()[0]
                sm_mask_uni_size = np.zeros((num_scale, args.num_classes, uni_sm_mask_size[0], uni_sm_mask_size[1]), dtype='float32')

                if flag_use_cuda:
                    labels = labels.cuda()

                max_val = max(max(inputs.max(), -inputs.min()), 1.0)
                for idx_scale, i_scale in enumerate(rand_scale):
                    cur_size = [round(max_size[0] * i_scale), round(max_size[1] * i_scale)]
                    inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')

                    for i in range(inputs.shape[0]):
                        inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))

                    if flag_use_cuda:
                        inputs_tensor = torch.from_numpy(inputs_resize).cuda()
                    else:
                        inputs_tensor = torch.from_numpy(inputs_resize)

                    sm_mask = net(inputs_tensor)

                    mask_gt_np = mask_gt.squeeze().numpy()
                    # mask_gt_np[mask_gt_np==255] = 0

                    sm_mask_uni_size[idx_scale, :, :, :] = np.transpose(resize(np.transpose(sm_mask.cpu().squeeze().numpy(), [1,2,0]), uni_sm_mask_size, mode='constant'), [2,0,1])

                if flag_avg:
                    sm_mask_comb = np.mean(sm_mask_uni_size, axis=0, keepdims=True)
                else:
                    sm_mask_comb = np.max(sm_mask_uni_size, axis=0, keepdims=True)

                if flag_crf:
                    result_big, result_small = crf_layer.run(sm_mask_comb, np.expand_dims(resize(img.numpy().squeeze().astype('uint8'),crf_size).astype('float32'), axis=0))
                    for i in range(labels.shape[0]):
                        temp = resize(np.transpose(result_big[i], [1,2,0]), args.input_size)
                        mask_pre = np.argmax(temp, axis=2)
                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)
                else:
                    for i in range(labels.shape[0]):
                        temp = np.transpose(sm_mask_comb[i,:,:,:], [1,2,0])
                        temp = resize(temp, args.input_size, mode='constant')
                        mask_pre = np.argmax(temp, axis=2)
                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

            train_iou = iou_obj.cal_cur_iou()
            iou_obj.iou_clear()

            print('Train iou is : ', train_iou, ' mean: ', train_iou.mean())

        else:  # evaluation

            for data in dataloader.dataloaders["val"]:
                inputs, labels, mask_gt, img = data

                labels_np = labels.squeeze().numpy()
                num_class = int(labels_np.sum())
                cur_class = labels_np.nonzero()[0]
                sm_mask_uni_size = np.zeros((num_scale, args.num_classes, uni_sm_mask_size[0], uni_sm_mask_size[1]), dtype='float32')

                if flag_use_cuda:
                    labels = labels.cuda()

                max_val = max(max(inputs.max(), -inputs.min()), 1.0)
                for idx_scale, i_scale in enumerate(rand_scale):
                    cur_size = [round(max_size[0] * i_scale), round(max_size[1] * i_scale)]
                    inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')

                    for i in range(inputs.shape[0]):
                        inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))

                    if flag_use_cuda:
                        inputs_tensor = torch.from_numpy(inputs_resize).cuda()
                    else:
                        inputs_tensor = torch.from_numpy(inputs_resize)

                    sm_mask = net(inputs_tensor)

                    mask_gt_np = mask_gt.squeeze().numpy()
                    # mask_gt_np[mask_gt_np==255] = 0

                    sm_mask_uni_size[idx_scale, :, :, :] = np.transpose(resize(np.transpose(sm_mask.cpu().squeeze().numpy(), [1,2,0]), uni_sm_mask_size, mode='constant'), [2,0,1])

                if flag_avg:
                    sm_mask_comb = np.mean(sm_mask_uni_size, axis=0, keepdims=True)
                else:
                    sm_mask_comb = np.max(sm_mask_uni_size, axis=0, keepdims=True)

                if flag_crf:
                    result_big, result_small = crf_layer.run(sm_mask_comb, np.expand_dims(resize(img.numpy().squeeze().astype('uint8'),crf_size).astype('float32'), axis=0))
                    for i in range(labels.shape[0]):
                        temp = resize(np.transpose(result_big[i], [1,2,0]), args.input_size)
                        mask_pre = np.argmax(temp, axis=2)
                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)
                else:
                    for i in range(labels.shape[0]):
                        temp = np.transpose(sm_mask_comb[i,:,:,:], [1,2,0])
                        temp = resize(temp, args.input_size, mode='constant')
                        mask_pre = np.argmax(temp, axis=2)
                        iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

            eval_iou = iou_obj.cal_cur_iou()
            iou_obj.iou_clear()

            print('Eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())


print("done")
