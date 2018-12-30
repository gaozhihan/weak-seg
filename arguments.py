import argparse

def get_args():
    parser = argparse.ArgumentParser(description='weak_supervise')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='threshold (default: 0.3)')
    parser.add_argument('--data_dir', type=str,
                        default="./data/VOC2012/VOC2012_SEG_AUG",
                        help='data loading directory')
    parser.add_argument('--saliency_dir', type=str,
                        default="./data/VOC2012/VOC2012_SEG_AUG/snapped_saliency/",
                        help='saliency loading directory')
    parser.add_argument('--attention_dir', type=str,
                        default="./data/VOC2012/VOC2012_SEG_AUG/snapped_attention/",
                        help='attention loading directory')
    parser.add_argument('--super_pixel_dir', type=str,
                        default="./data/VOC2012/VOC2012_SEG_AUG/super_pixel/",
                        help='super pixel loading directory')
    parser.add_argument('--sec_id_img_name_list_dir', type=str,
                        default="/home/sunting/Documents/program/SEC-master/training/input_list.txt",
                        help='sec id image mame input list')
    parser.add_argument('--cues_pickle_dir', type=str,
                        default="/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle",
                        help='cues pickle file dir for SEC')
    parser.add_argument('--model', type=str, default="my_resnet3",  # resnet SEC
                        help='model type resnet|SEC')
    parser.add_argument('--CRF_model', type=str, default="my_CRF",  # resnet SEC
                        help='CRF model my_CRF|SEC_CRF')
    parser.add_argument('--loss', type=str, default="BCELoss",  # resnet SEC
                        help='model type MultiLabelSoftMarginLoss|BCELoss')
    parser.add_argument('--batch_size', type=int, default=15,
                        help='training batch size')
    parser.add_argument('--origin_size', action='store_true', default=False,
                        help='when it is training')
    parser.add_argument('--relu_mask', action='store_true', default=False,
                        help='whether apply relu for not')
    parser.add_argument('--preds_only', action='store_true', default=True,
                        help='whether only use')
    parser.add_argument('--input_size', nargs='+', type=int, default=[256,256],
                        help='size of training images [224,224]|[321,321]')
    parser.add_argument('--output_size', nargs='+', type=int, default=[32,32], # 32 for 256 input; 28 for 224 input
                        help='size of output mask')
    parser.add_argument('--step_size', type=int, default=20,
                        help='optimizer scheduler step size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='training epoches (default: 100)')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='classificiation nums')
    parser.add_argument('--no_bg', action='store_true', default=False,
                        help='no background')
    parser.add_argument('--color_vote', action='store_true', default=True,
                        help='no background')
    parser.add_argument('--fix_CRF_itr', action='store_true', default=False,
                        help='fix CRF iteration')
    parser.add_argument('--test_flag', action='store_true', default=False,
                        help='when it is training')
    parser.add_argument('--SEC_loss_flag', action='store_true', default=False,
                        help='whether to use SEC loss for training')
    parser.add_argument('--cross_entropy_weight', action='store_true', default=True,
                        help='whether to weigh the cross entropy by confidence')
    parser.add_argument('--need_mask_flag', action='store_true', default=False,
                        help='need mask even training')

    args = parser.parse_args()
    if args.no_bg==True:
        args.num_classes=20
    return args



