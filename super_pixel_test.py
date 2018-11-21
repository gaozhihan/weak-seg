import torch
from voc_data import VOCData
#from voc_data_org_size_batch import VOCData
import time
import socket
from arguments import get_args
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
import common_function
import numpy as np

def generate_visualize_super_pixel(img, mask_gt):
    if img.max() > 1:
        img = img / 255.0

    segments_fz = felzenszwalb(img, scale=100, sigma=3, min_size=10)
    segments_slic = slic(img, n_segments=100, compactness=20, sigma=0.8)
    segments_quick = quickshift(img, kernel_size=5, max_dist=10, ratio=0.5, sigma=1)

    print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
    print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
    print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

    # visualization
    f, ax = plt.subplots(nrows=3, ncols=3)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title("orginal image")
    ax[0, 1].imshow(mask_gt)
    ax[0, 1].set_title("mask gt")
    ax[0, 2].imshow(mark_boundaries(img,mask_gt))
    ax[0, 2].set_title("mask boundary")
    ax[1, 0].imshow(segments_fz)
    ax[1, 0].set_title("Felzenszwalbs")
    ax[1, 1].imshow(segments_slic)
    ax[1, 1].set_title("slic")
    ax[1, 2].imshow(segments_quick)
    ax[1, 2].set_title("quickshift")
    ax[2, 0].imshow(mark_boundaries(img,segments_fz))
    ax[2, 0].set_title("Felzenszwalbs")
    ax[2, 1].imshow(mark_boundaries(img,segments_slic))
    ax[2, 1].set_title("slic")
    ax[2, 2].imshow(mark_boundaries(img,segments_quick))
    ax[2, 2].set_title("quickshift")

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def params_visualize_felzenszwalb(img, mask_gt):
    # image : (width, height, 3) or (width, height) ndarray Input image.
    #
    # scale : float
    #         Free parameter. Higher means larger clusters.
    #
    # sigma : float
    #         Width (standard deviation) of Gaussian kernel used in preprocessing.
    #
    # min_size : int
    #           Minimum component size. Enforced using postprocessing.
    #
    # multichannel : bool, optional (default: True)
    #               Whether the last axis of the image is to be interpreted as multiple channels. A value of False, for a 3D image, is not currently supported.

    if img.max() > 1:
        img = img / 255.0

    scale_def = 100
    sigma_def = 1.5 #3
    min_size_def = 10

    scale_grid = [10, 100, 200]
    sigma_grid = [0.8, 1.5, 2, 3]
    min_size_grid = [1, 10, 50]

    f, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img)
    ax[0].set_title("orginal image")
    ax[1].imshow(mask_gt)
    ax[1].set_title("mask gt")
    ax[2].imshow(mark_boundaries(img,mask_gt))
    ax[2].set_title("mask boundary")
    for a in ax.ravel():
        a.set_axis_off()

    # for scale
    f, ax = plt.subplots(nrows=1, ncols=len(scale_grid))
    for i in range(len(scale_grid)):
        segments_fz = felzenszwalb(img, scale=scale_grid[i], sigma=sigma_def, min_size=min_size_def)
        ax[i].imshow(mark_boundaries(img,segments_fz))
        ax[i].set_title("scale {}".format(scale_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for sigma
    f, ax = plt.subplots(nrows=1, ncols=len(sigma_grid))
    for i in range(len(sigma_grid)):
        segments_fz = felzenszwalb(img, scale=scale_def, sigma=sigma_grid[i], min_size=min_size_def)
        ax[i].imshow(mark_boundaries(img,segments_fz))
        ax[i].set_title("sigma {}".format(sigma_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for min_size
    f, ax = plt.subplots(nrows=1, ncols=len(min_size_grid))
    for i in range(len(min_size_grid)):
        segments_fz = felzenszwalb(img, scale=scale_def, sigma=sigma_def, min_size=min_size_grid[i])
        ax[i].imshow(mark_boundaries(img,segments_fz))
        ax[i].set_title("min size {}".format(min_size_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    plt.tight_layout()
    plt.show()


def params_visualize_slic(img, mask_gt):
    # image : (width, height, 3) or (width, height) ndarray Input image.
    #
    # n_segments : int, optional
    #           The (approximate) number of labels in the segmented output image.
    #
    # compactness : float, optional
    #          Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. In SLICO mode, this is the initial compactness. This parameter depends strongly on image contrast and on the shapes of objects in the image. We recommend exploring possible values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.

    if img.max() > 1:
        img = img / 255.0

    n_segments_def = 200
    compactness_def = 20
    sigma_def = 0.8

    n_segments_grid = [10, 100, 200]
    compactness_grid = [0.5, 5, 20, 30]
    sigma_grid = [0.5, 1, 10]

    f, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img)
    ax[0].set_title("orginal image")
    ax[1].imshow(mask_gt)
    ax[1].set_title("mask gt")
    ax[2].imshow(mark_boundaries(img,mask_gt))
    ax[2].set_title("mask boundary")
    for a in ax.ravel():
        a.set_axis_off()

    # for n_segments
    f, ax = plt.subplots(nrows=1, ncols=len(n_segments_grid))
    for i in range(len(n_segments_grid)):
        segments_slic = slic(img, n_segments=n_segments_grid[i], compactness=compactness_def, sigma=sigma_def)
        ax[i].imshow(mark_boundaries(img,segments_slic))
        ax[i].set_title("n segment {}".format(n_segments_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for compactness
    f, ax = plt.subplots(nrows=1, ncols=len(compactness_grid))
    for i in range(len(compactness_grid)):
        segments_slic = slic(img, n_segments=n_segments_def, compactness=compactness_grid[i], sigma=sigma_def)
        ax[i].imshow(mark_boundaries(img,segments_slic))
        ax[i].set_title("compactness {}".format(compactness_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for sigma
    f, ax = plt.subplots(nrows=1, ncols=len(sigma_grid))
    for i in range(len(sigma_grid)):
        segments_slic = slic(img, n_segments=n_segments_def, compactness=compactness_def, sigma=sigma_grid[i])
        ax[i].imshow(mark_boundaries(img,segments_slic))
        ax[i].set_title("sigma {}".format(sigma_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    plt.tight_layout()
    plt.show()


def params_visualize_quickshift(img, mask_gt):
    # image : (width, height, channels) ndarray Input image.
    #
    # ratio : float, optional, between 0 and 1
    #       Balances color-space proximity and image-space proximity. Higher values give more weight to color-space.
    #
    # kernel_size : float, optional
    #        Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
    #
    # max_dist : float, optional
    #       Cut-off point for data distances. Higher means fewer clusters.
    #
    # sigma : float, optional
    #        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    #

    if img.max() > 1:
        img = img / 255.0

    kenel_size_def = 5
    max_dist_def = 10
    ratio_def = 0.5 # between 0 and 1
    sigma_def = 1

    kenel_size_grid = [3, 5, 9]
    max_dist_grid = [6, 10, 30]
    ratio_grid = [0.2, 0.5, 0.8]
    sigma_grid = [0, 0.1, 1, 5]

    f, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img)
    ax[0].set_title("orginal image")
    ax[1].imshow(mask_gt)
    ax[1].set_title("mask gt")
    ax[2].imshow(mark_boundaries(img,mask_gt))
    ax[2].set_title("mask boundary")
    for a in ax.ravel():
        a.set_axis_off()

    # for kernel_size
    f, ax = plt.subplots(nrows=1, ncols=len(kenel_size_grid))
    for i in range(len(kenel_size_grid)):
        segments_quick = quickshift(img, kernel_size=kenel_size_grid[i], max_dist=max_dist_def, ratio=ratio_def, sigma=sigma_def)
        ax[i].imshow(mark_boundaries(img,segments_quick))
        ax[i].set_title("kernel size {}".format(kenel_size_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for max_dist
    f, ax = plt.subplots(nrows=1, ncols=len(max_dist_grid))
    for i in range(len(max_dist_grid)):
        segments_quick = quickshift(img, kernel_size=kenel_size_def, max_dist=max_dist_grid[i], ratio=ratio_def, sigma=sigma_def)
        ax[i].imshow(mark_boundaries(img,segments_quick))
        ax[i].set_title("max dist {}".format(max_dist_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for ratio
    f, ax = plt.subplots(nrows=1, ncols=len(ratio_grid))
    for i in range(len(ratio_grid)):
        segments_quick = quickshift(img, kernel_size=kenel_size_def, max_dist=max_dist_def, ratio=ratio_grid[i], sigma=sigma_def)
        ax[i].imshow(mark_boundaries(img,segments_quick))
        ax[i].set_title("ratio {}".format(ratio_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    # for sigma
    f, ax = plt.subplots(nrows=1, ncols=len(sigma_grid))
    for i in range(len(sigma_grid)):
        segments_quick = quickshift(img, kernel_size=kenel_size_def, max_dist=max_dist_def, ratio=ratio_def, sigma=sigma_grid[i])
        ax[i].imshow(mark_boundaries(img,segments_quick))
        ax[i].set_title("sigma {}".format(sigma_grid[i]))
        for a in ax.ravel():
            a.set_axis_off()

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.test_flag = True
    args.model = 'my_resnet' # resnet; my_resnet; SEC; my_resnet3; decoupled
    model_path = 'models/top_val_acc_my_resnet_25' # sec: sec_rename; resnet: top_val_acc_resnet; my_resnet: top_val_acc_my_resnet_25; my_resnet3: top_val_rec_my_resnet3_27; decoupled: top_val_acc_decoupled_28
    args.input_size = [224,224] #[224,224] [321,321]
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

        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':

                for data in dataloader.dataloaders["train"]:
                    inputs, labels, mask_gt, img = data
                    # generate_visualize_super_pixel(img.detach().squeeze().numpy(), mask_gt.detach().squeeze().numpy())
                    params_visualize_felzenszwalb(img.detach().squeeze().numpy(), mask_gt.detach().squeeze().numpy())
                    # params_visualize_slic(img.detach().squeeze().numpy(), mask_gt.detach().squeeze().numpy())
                    # params_visualize_quickshift(img.detach().squeeze().numpy(), mask_gt.detach().squeeze().numpy())

                    plt.close('all')

            else:  # evaluation
                start = time.time()
                for data in dataloader.dataloaders["val"]:
                    inputs, labels, mask_gt, img = data
                    generate_visualize_super_pixel(img.detach().numpy(), mask_gt.detach().numpy())

                    plt.close('all')








