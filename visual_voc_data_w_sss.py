import torch
from voc_data_w_superpixel_snapped_at_sal import VOCData
import time
import socket
from arguments import get_args
import matplotlib.pyplot as plt
from skimage.transform import resize
import common_function
import numpy as np
from skimage.segmentation import mark_boundaries

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
        args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
        args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
        args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
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
    print(args)
    dataloader = VOCData(args)

    with torch.no_grad():

        for phase in ['train', 'val']:
            if phase == 'train':
                for data in dataloader.dataloaders["train"]:
                    inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask_expand = data
                    temp_label = labels.squeeze()
                    temp_label[0] = 0
                    temp = temp_label.nonzero().tolist()
                    temp_label = [item for sublist in temp for item in sublist]
                    attention_mask = attention_mask_expand[:, temp_label, :,:]

                    # visualization
                    x = labels.detach().squeeze().numpy()
                    cur_class = np.nonzero(x[1:])[0]
                    num_cur_class = len(cur_class)
                    plt_col_num = max(4, num_cur_class)
                    plt.subplot(2, plt_col_num, 1); plt.imshow(img.detach().squeeze()/255.0); plt.title('Input image'); plt.axis('off')

                    mask_gt_temp = mask_gt.detach().squeeze().numpy()
                    mask_gt_temp[mask_gt_temp==255] = 0
                    plt.subplot(2, plt_col_num, 2); plt.imshow(mask_gt_temp); plt.title('mask gt'); plt.axis('off')

                    plt.subplot(2, plt_col_num, 3); plt.imshow(saliency_mask.squeeze()); plt.title('saliency'); plt.axis('off')
                    plt.subplot(2, plt_col_num, 4); plt.imshow(super_pixel.squeeze()); plt.title('super pixel'); plt.axis('off')

                    for i in range(num_cur_class):
                        plt.subplot(2, plt_col_num, plt_col_num+1+i); plt.imshow(attention_mask[:,i,:,:].squeeze()); plt.title('snapped attention {}'.format(cur_class[i])); plt.axis('off')

                    plt.close('all')

            else:  # evaluation
                for data in dataloader.dataloaders["val"]:
                    inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask_expand = data








