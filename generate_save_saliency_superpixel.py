import torch
import torch.nn as nn
from voc_data_img_name import VOCData
#from voc_data_org_size_batch import VOCData
import time
import socket
from arguments import get_args
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
import numpy as np
import net_saliency

# ---------------- visualize the saliency mask generated from the given feature maps ------------------------
def generate_saliency(outputs, flag_classify, flag_sum):
    with torch.no_grad():

        if flag_classify:
            features = outputs[:-1]

        else:
            features = outputs

        num_maps = len(features)
        num_img = features[0].shape[0]

        shape_mask = features[0][0][0].shape

        for i_img in range(num_img):

            if flag_sum:
                sum_layer_mask_per_img = np.zeros(shape_mask)
            else:
                max_layer_mask_per_img = np.zeros(shape_mask)

            for i_map in range(num_maps):
                temp_feature = features[i_map][i_img].sum(axis=0)  #[1:] means no background
                # normalization
                if temp_feature.max() != 0:
                    temp_feature = (temp_feature - temp_feature.min())/temp_feature.max()

                temp = resize(temp_feature, shape_mask, mode='constant')
                if flag_sum:
                    sum_layer_mask_per_img = sum_layer_mask_per_img + temp
                    if sum_layer_mask_per_img.max() > 0:
                        sum_layer_mask_per_img = (sum_layer_mask_per_img - sum_layer_mask_per_img.min())/sum_layer_mask_per_img.max()
                    return sum_layer_mask_per_img
                else:
                    max_layer_mask_per_img = np.maximum(max_layer_mask_per_img, temp)
                    if max_layer_mask_per_img.max() > 0:
                        max_layer_mask_per_img = (max_layer_mask_per_img - max_layer_mask_per_img.min())/max_layer_mask_per_img.max()
                    return max_layer_mask_per_img


def snap_saliency_to_superpixel(saliency_mask, img, arg_super_pixel):

    if img.max() > 1:
        img = img / 255.0

    img_shape = img.shape[:2]
    saliency_mask_rez = resize(saliency_mask, img_shape, mode='constant')
    saliency_mask_snapped = np.zeros(img_shape)

    if arg_super_pixel == 'felzenszwalb':
        seg = felzenszwalb(img, scale=100, sigma=1.5, min_size=10)
    elif arg_super_pixel == 'slic':
        seg = slic(img, n_segments=100, compactness=20, sigma=0.8)
    elif arg_super_pixel == 'quickshift':
        seg = quickshift(img, kernel_size=5, max_dist=10, ratio=0.5, sigma=1)

    num_seg = int(seg.max()) + 1

    for i_seg in range(num_seg):
        cur_seg = (seg == i_seg)
        cur_saliency_region = saliency_mask_rez[cur_seg]
        saliency_mask_snapped[cur_seg] = cur_saliency_region.mean()

    # print(num_seg)
    # plt.subplot(2,3,1); plt.imshow(img); plt.title('Input image'); plt.axis('off')
    # plt.subplot(2,3,3); plt.imshow(saliency_mask_rez); plt.title('original saliency'); plt.axis('off')
    # plt.subplot(2,3,4); plt.imshow(seg); plt.title('super pixel'); plt.axis('off')
    # plt.subplot(2,3,5); plt.imshow(mark_boundaries(img,seg)); plt.title('super pixel'); plt.axis('off')
    # plt.subplot(2,3,6); plt.imshow(saliency_mask_snapped); plt.title('snapped saliency'); plt.axis('off')

    return saliency_mask_snapped, seg




if __name__ == '__main__':
    arg_model = 'decoupled_net' # vgg16, sec, resnet
    flag_sum = False
    arg_super_pixel = 'felzenszwalb' # 'felzenszwalb', 'slic', 'quickshift'

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
        save_saliency_path = '/home/sunting/Documents/program/VOC2012_SEG_AUG/raw_saliency/'
        save_superpixel_path = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
        save_snapped_saliency_path = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
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
    if arg_model == 'vgg16':
        net = net_saliency.Vgg16_for_saliency(flag_classify)
    elif arg_model == 'sec':
        net = net_saliency.SEC_for_saliency(flag_classify, args)
    elif arg_model == 'resnet':
        net = net_saliency.resnet_saliency(flag_classify)
    elif arg_model == 'decoupled_net':
        net = net_saliency.decouple_net_saliency(flag_classify)

    print(args)

    if flag_use_cuda:
        net.cuda()

    dataloader = VOCData(args)

    net.train(False)

    with torch.no_grad():

        start = time.time()
        counter = 0
        for phase in ['train', 'val']:
            if phase == 'train':

                for data in dataloader.dataloaders["train"]:
                    inputs, labels, mask_gt, img, img_name = data
                    if flag_use_cuda:
                        inputs = inputs.cuda(); labels = labels.cuda()

                    outputs = net(inputs)
                    saliency_mask = generate_saliency(outputs, flag_classify, flag_sum)
                    saliency_mask_snapped, super_pixel_seg = snap_saliency_to_superpixel(saliency_mask, img.detach().squeeze().numpy(), arg_super_pixel)

                    # mask_gt[mask_gt==255] = 0
                    # plt.subplot(2,3,2); plt.imshow(mask_gt.detach().squeeze().numpy()); plt.title('gt mask'); plt.axis('off')
                    # plt.close('all')

                    # save raw_saliency, super pixel seg, snapped saliency
                    np.save(save_saliency_path+img_name[0]+'.npy', saliency_mask.astype('float16'))
                    np.save(save_superpixel_path+img_name[0]+'.npy', super_pixel_seg.astype('uint16'))
                    np.save(save_snapped_saliency_path+img_name[0]+'.npy', saliency_mask_snapped.astype('float16'))
                    # d = np.load('test3.npy')
                    print(counter)
                    counter+=1

            else:  # evaluation
                start = time.time()
                for data in dataloader.dataloaders["val"]:
                    inputs, labels, mask_gt, img, img_name = data
                    if flag_use_cuda:
                        inputs = inputs.cuda(); labels = labels.cuda()

                    outputs = net(inputs)
                    saliency_mask = generate_saliency(outputs, flag_classify, flag_sum)
                    saliency_mask_snapped, super_pixel_seg = snap_saliency_to_superpixel(saliency_mask, img.detach().squeeze().numpy(), arg_super_pixel)

                    # mask_gt[mask_gt==255] = 0
                    # plt.subplot(2,3,2); plt.imshow(mask_gt.detach().squeeze().numpy()); plt.title('gt mask'); plt.axis('off')
                    # plt.close('all')

                    # save raw_saliency, super pixel seg, snapped saliency
                    np.save(save_saliency_path+img_name[0]+'.npy', saliency_mask.astype('float16'))
                    np.save(save_superpixel_path+img_name[0]+'.npy', super_pixel_seg.astype('uint16'))
                    np.save(save_snapped_saliency_path+img_name[0]+'.npy', saliency_mask_snapped.astype('float16'))
                    print(counter)
                    counter+=1







