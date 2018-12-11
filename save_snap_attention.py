import torch
import torch.nn as nn
from voc_data_img_name import VOCData
#from voc_data_org_size_batch import VOCData
import time
import socket
from arguments import get_args
import matplotlib.pyplot as plt
from skimage.transform import resize
import common_function
import numpy as np
import json
import net_saliency
from skimage.segmentation import mark_boundaries



def snap_to_superpixel(saliency_mask, img, seg):

    if img.max() > 1:
        img = img / 255.0

    img_shape = img.shape[:2]
    saliency_mask_rez = resize(saliency_mask, img_shape, mode='constant')
    saliency_mask_snapped = np.zeros(img_shape)

    num_seg = int(seg.max()) + 1

    for i_seg in range(num_seg):
        cur_seg = (seg == i_seg)
        cur_saliency_region = saliency_mask_rez[cur_seg]
        saliency_mask_snapped[cur_seg] = cur_saliency_region.mean()

    return saliency_mask_snapped



# assume batch size = 1
def snap_anntention_to_superpixel(outputs, img, mask_gt, flag_classify, labels, super_pixel_path):
    with torch.no_grad():
        mask_gt[mask_gt==255] = 0
        mask_gt = mask_gt.squeeze()
        if img.max() > 1:
            img = img.squeeze() / 255.0

        if flag_classify:
            features = outputs[:-1]
            predictions = outputs[-1]

        else:
            features = outputs

        cur_class = np.nonzero(labels[1:])[0]
        # print(cur_class)
        num_cur_class = len(cur_class)
        attention = features[-1].squeeze()
        super_pixel = np.load(super_pixel_path)
        snapped_attention = np.zeros((num_cur_class, img.shape[0], img.shape[1]))

        # plt.subplot(2,(2 + num_cur_class),1); plt.imshow(img); plt.title('Input image'); plt.axis('off')
        # plt.subplot(2,(2 + num_cur_class),3 + num_cur_class); plt.imshow(mask_gt); plt.title('true mask'); plt.axis('off')
        # plt.subplot(2,(2 + num_cur_class),2); plt.imshow(super_pixel); plt.title('super pixel'); plt.axis('off')
        # plt.subplot(2,(2 + num_cur_class),4 + num_cur_class); plt.imshow(mark_boundaries(img,super_pixel)); plt.title('boundary'); plt.axis('off')

        for i in range(num_cur_class):
            # plt.subplot(2,(2 + num_cur_class),3+i); plt.imshow(attention[cur_class[i],:,:]); plt.title('raw attention {}'.format(cur_class[i])); plt.axis('off')
            snapped_attention[i,:,:] = snap_to_superpixel(attention[cur_class[i],:,:], img, super_pixel)
            # plt.subplot(2,(2 + num_cur_class),num_cur_class+5+i); plt.imshow(snapped_attention[i,:,:]); plt.title('snapped attention {}'.format(cur_class[i])); plt.axis('off')

        # plt.close('all')
        return  snapped_attention


if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.test_flag = True
    args.model = 'my_resnet' # resnet; my_resnet; SEC; my_resnet3; decoupled
    model_path = 'models/top_val_acc_my_resnet_25' # sec: sec_rename; resnet: top_val_acc_resnet; my_resnet: top_val_acc_my_resnet_25; my_resnet3: top_val_rec_my_resnet3_27; decoupled: top_val_acc_decoupled_28
    args.input_size = [321,321] #[224,224]
    args.output_size = [41, 41] #[29,29]
    args.origin_size = False
    args.color_vote = True
    args.fix_CRF_itr = False
    args.preds_only = True
    args.CRF_model = 'my_CRF' # SEC_CRF or my_CRF
    flag_classify = False

    host_name = socket.gethostname()
    flag_use_cuda = torch.cuda.is_available()

    if host_name == 'sunting':
        args.batch_size = 5
        args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
        model_path = model_path + '_CPU'
        super_pixel_path = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
        save_snapped_attention_path = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    elif host_name == 'sunting-ThinkCenter-M90':
        args.batch_size = 18
    elif host_name == 'ram-lab':
        args.data_dir = '/data_shared/Docker/ltai/ws/decoupled_net/data/VOC2012/VOC2012_SEG_AUG'
        if args.model == 'SEC':
            args.batch_size = 50
        elif args.model == 'resnet':
            args.batch_size = 100
        elif args.model == 'my_resnet':
            args.batch_size = 30
        elif args.model == 'decoupled':
            args.batch_size = 38

    model_path = model_path + '.pth'
    args.batch_size = 1
    net_decouple = net_saliency.decouple_net_saliency(flag_classify)

    print(args)

    if flag_use_cuda:
        net_decouple.cuda()

    dataloader = VOCData(args)

    net_decouple.train(False)
    counter = 0
    with torch.no_grad():

        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':

                for data in dataloader.dataloaders["train"]:
                    inputs, labels, mask_gt, img, img_name = data
                    if flag_use_cuda:
                        inputs = inputs.cuda(); labels = labels.cuda()

                    outputs = net_decouple(inputs)
                    snapped_attention = snap_anntention_to_superpixel(outputs, img.detach().squeeze().numpy(), mask_gt, flag_classify, labels.detach().squeeze().numpy(), super_pixel_path+img_name[0]+'.npy')
                    np.save(save_snapped_attention_path+img_name[0]+'.npy', snapped_attention.astype('float16'))
                    print(counter)
                    counter+=1

            else:  # evaluation
                start = time.time()
                for data in dataloader.dataloaders["val"]:
                    inputs, labels, mask_gt, img, img_name = data
                    if flag_use_cuda:
                        inputs = inputs.cuda(); labels = labels.cuda()

                    outputs = net_decouple(inputs)
                    snapped_attention = snap_anntention_to_superpixel(outputs, img.detach().squeeze().numpy(), mask_gt, flag_classify, labels.detach().squeeze().numpy(), super_pixel_path+img_name[0]+'.npy')
                    np.save(save_snapped_attention_path+img_name[0]+'.npy', snapped_attention.astype('float16'))
                    print(counter)
                    counter+=1








