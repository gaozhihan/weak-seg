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
from skimage.transform import resize


args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_rec_my_resnet_5_5.pth'
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    args.batch_size = 24

net = st_resnet.resnet_st_seg01.resnet50(pretrained=False, num_classes=args.num_classes)
net.load_state_dict(torch.load(model_path), strict = True)

st_crf_layer = sec.sec_org_net.STCRFLayer(True)
seed_loss_layer = sec.sec_org_net.SeedingLoss()
expand_loss_layer = sec.sec_org_net.ExpandLossLayer(flag_use_cuda)
st_constrain_loss_layer = sec.sec_org_net.STConstrainLossLayer()


print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)

iou_obj = common_function.iou_calculator()

num_train_batch = len(dataloader.dataloaders["train"])

with torch.no_grad():

    train_iou = 0
    eval_iou = 0

    start = time.time()

    net.train(False)

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, cues = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda(); cues = cues.cuda()

        sm_mask = net(inputs)

        for i in range(labels.shape[0]):
            temp = resize(sm_mask[i].permute([1,2,0]).numpy(), args.input_size, mode='constant')
            mask_pre = np.argmax(temp, axis=2)
            iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

    train_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())

    # if (epoch % 5 == 0):  # evaluation
    for data in dataloader.dataloaders["val"]:
        inputs, labels, mask_gt, img = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()

        with torch.no_grad():
            sm_mask = net(inputs)

            for i in range(labels.shape[0]):
                temp = resize(sm_mask[i].permute([1,2,0]).numpy(), args.input_size, mode='constant')
                mask_pre = np.argmax(temp, axis=2)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

    eval_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    print('cur eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())

print("done")



