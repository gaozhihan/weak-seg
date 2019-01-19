import torch
from sec.sec_data_loader_no_rand import VOCData
import sec.sec_org_net
import st_resnet.resnet_st_sal_att
import socket
from arguments import get_args
import common_function
import numpy as np
import datetime
import matplotlib.pyplot as plt
from skimage.transform import resize


args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]
args.batch_size = 1

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_sec01_ws_top_val_iou_my_resnet.pth'
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    args.batch_size = 24

net = st_resnet.resnet_st_sal_att.resnet50(pretrained=False, num_classes=args.num_classes)
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
    net.train(False)

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, cues = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda(); cues = cues.cuda()

        # obtain saliency and attention
        layer4_feature, sm_mask = net(inputs)
        f4_np = layer4_feature.squeeze().numpy()
        sm_mask_np = sm_mask.squeeze().numpy()
        img_np = img.squeeze().numpy().astype('uint8')
        mask_gt_np = mask_gt.squeeze().numpy()
        mask_gt_np[mask_gt_np==255] = 0

        sal_f4 = np.sum(f4_np, axis=0)
        if sal_f4.max() != 0:
            sal_f4 = (sal_f4 - sal_f4.min())/sal_f4.max()

        sal_sm_mask = np.sum(sm_mask_np[1:,:,:], axis=0) # important! do NOT include background
        if sal_sm_mask.max() != 0:
            sal_sm_mask = (sal_sm_mask - sal_sm_mask.min())/sal_sm_mask.max()

        cur_class = np.nonzero(labels.squeeze().numpy()[1:])[0]
        num_cur_class = len(cur_class)

        # visualization --------------------------
        # attention
        plt.figure(); plt.title('attention')

        if num_cur_class > 1: # more than one class present, can use subtract attention
            sal_cur_class = sm_mask_np[cur_class+1,:,:].sum(axis=0)

            for idx, i_class in enumerate(cur_class):
                plt.subplot(2,num_cur_class,idx+1); plt.imshow(sm_mask_np[i_class+1], cmap='gray'); plt.axis('off')
                sub_att_temp = np.maximum(sm_mask_np[i_class+1]* 2 - sal_cur_class, 0.0)
                plt.subplot(2,num_cur_class,num_cur_class+idx+1); plt.imshow(sub_att_temp, cmap='gray'); plt.axis('off')

        else:
            for idx, i_class in enumerate(cur_class):
                plt.subplot(1,num_cur_class,idx+1); plt.imshow(sm_mask_np[i_class+1], cmap='gray')

        # saliency
        plt.figure()
        plt.subplot(1,6,1); plt.imshow(img_np); plt.title('img'); plt.axis('off')
        plt.subplot(1,6,2); plt.imshow(mask_gt_np); plt.title('gt'); plt.axis('off')
        plt.subplot(1,6,3); plt.imshow(sal_f4, cmap='gray'); plt.title('sal f4'); plt.axis('off')
        plt.subplot(1,6,4); plt.imshow(sal_sm_mask, cmap='gray'); plt.title('sal sm'); plt.axis('off')
        plt.subplot(1,6,5); plt.imshow(np.maximum(sal_f4,sal_sm_mask), cmap='gray'); plt.title('sal max'); plt.axis('off')
        plt.subplot(1,6,6); plt.imshow((sal_f4+sal_sm_mask)/2, cmap='gray'); plt.title('sal sum'); plt.axis('off')

        plt.close('all')



    # if (epoch % 5 == 0):  # evaluation
    for data in dataloader.dataloaders["val"]:
        inputs, labels, mask_gt, img = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()

        layer4_feature, sm_mask = net(inputs)



print("done")



