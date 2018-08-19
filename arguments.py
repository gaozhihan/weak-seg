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
    parser.add_argument('--model', type=str, default="my_resnet",  # resnet SEC
                        help='model type resnet|SEC')
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
    parser.add_argument('--output_size', nargs='+', type=int, default=[29,29],
                        help='size of output mask')
    parser.add_argument('--step_size', type=int, default=20,
                        help='optimizer scheduler step size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoches (default: 100)')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='classificiation nums')
    parser.add_argument('--no_bg', action='store_true', default=False,
                        help='no background')
    parser.add_argument('--test_flag', action='store_true', default=False,
                        help='when it is training')
    parser.add_argument('--need_mask_flag', action='store_true', default=False,
                        help='need mask even training')

    args = parser.parse_args()
    if args.no_bg==True:
        args.num_classes=20
    return args



