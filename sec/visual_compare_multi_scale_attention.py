import torch
import torch.nn as nn
from sec.sec_data_loader_no_rand import VOCData
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np
import random
from skimage.transform import resize
import matplotlib.pyplot as plt

args = get_args()
args.need_mask_flag = False
args.model = 'SEC'
args.input_size = [385,385]
max_size = [385, 385]
uni_sm_mask_size = [41, 41]
args.output_size = [41, 41]
args.batch_size = 1
random.uniform(0.67, 1.0)
# rand_scale = [0.67, 0.8337, 1.0]
rand_scale = [0.5, 0.8, 1.0]

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
now = datetime.datetime.now()
date_str = str(now.day) + '_' + str(now.day)

args.batch_size = 1

if host_name == 'sunting':
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/sec_rename_CPU.pth' # 'vgg16'
elif host_name == 'sunting-ThinkCentre-M90':
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/snapped_attention/'
    model_path = '/home/sunting/Documents/program/weak-seg/models/sec_rename_CPU.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_01_top_val_rec_SEC_31_31.pth'
    args.super_pixel_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_attention/'

net = st_01.sec_net.SEC_NN()
net.load_state_dict(torch.load(model_path), strict = False)
crf_layer = st_01.sec_net.CRFLayer(False)

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

with torch.no_grad():
    net.train(False)
    for phase in ['train', 'val']:
        if phase == 'train':

            for data in dataloader.dataloaders["train"]:
                inputs, labels, mask_gt, img, cues = data

                labels_np = labels.squeeze().numpy()
                num_class = int(labels_np.sum())
                cur_class = labels_np.nonzero()[0]
                sm_mask_uni_size = np.zeros((num_scale, num_class, uni_sm_mask_size[0], uni_sm_mask_size[1]), dtype='float32')

                if flag_use_cuda:
                    labels = labels.cuda()

                max_val = max(max(inputs.max(), -inputs.min()), 1.0)
                for idx_scale, i_scale in enumerate(rand_scale):
                    cur_size = [round(max_size[0] * i_scale), round(max_size[1] * i_scale)]
                    inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')

                    for i in range(inputs.shape[0]):
                        inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))

                    # plt.imshow(np.transpose(inputs[0].detach().numpy(), (1,2,0)))

                    if flag_use_cuda:
                        inputs_tensor = torch.from_numpy(inputs_resize).cuda()
                    else:
                        inputs_tensor = torch.from_numpy(inputs_resize)

                    sm_mask, preds = net(inputs_tensor)

                    plt.subplot(num_scale+2, num_class+2, idx_scale*(num_class+2)+1); plt.imshow(img.squeeze().numpy().astype('uint8')); plt.title('input')
                    mask_gt_np = mask_gt.squeeze().numpy()
                    mask_gt_np[mask_gt_np==255] = 0
                    plt.subplot(num_scale+2, num_class+2, idx_scale*(num_class+2)+2); plt.imshow(mask_gt_np); plt.title('gt')
                    for idx_class, i_class in enumerate(cur_class):
                        plt.subplot(num_scale+2, num_class+2, idx_scale*(num_class+2)+3+idx_class); plt.imshow(sm_mask.squeeze()[i_class].numpy(), cmap='gray')
                        sm_mask_uni_size[idx_scale, idx_class, :, :] = resize(sm_mask.squeeze()[i_class].numpy(), uni_sm_mask_size, mode='constant')

                for i in range(num_class):
                    plt.subplot(num_scale+2, num_class+2, (num_class+2)*num_scale + 3+i); plt.imshow(np.mean(sm_mask_uni_size[:,i,:,:], axis=0), cmap='gray')
                    plt.subplot(num_scale+2, num_class+2, (num_class+2)*(num_scale+1) + 3+i); plt.imshow(np.max(sm_mask_uni_size[:,i,:,:], axis=0), cmap='gray')

                plt.close('all')


        else:  # evaluation

            for data in dataloader.dataloaders["val"]:
                inputs, labels, mask_gt, img = data
                if flag_use_cuda:
                    inputs = inputs.cuda(); labels = labels.cuda()


print("done")
