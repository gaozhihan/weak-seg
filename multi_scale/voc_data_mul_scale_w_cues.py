import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import pickle
from PIL import Image
import random
from torchvision.transforms import functional as F

from arguments import get_args
import socket
from skimage.transform import resize
import matplotlib.pyplot as plt

from multiprocessing import Pool
import os

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


class VOCData():
    def __init__(self, args):
        self.max_size = [385, 385]
        if args.model ==  "SEC":
            if args.test_flag == False:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(self.max_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                ]),
                }
            else:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(self.max_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                ]),
                }

        else:
            if args.test_flag == False:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(self.max_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                }
            else:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(self.max_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                }


        self.image_datasets = {x:VOCDataset(x+".txt", args, self.data_transforms[x])
                          for x in ['train', 'val']}

        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
            self.image_datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4),
            "val": torch.utils.data.DataLoader(
            self.image_datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4)}


        self.dataset_sizes = {
                "train": len(self.image_datasets["train"]),
                "val": len(self.image_datasets["val"])}
        #self.class_names = self.image_datasets['train'].classes

class VOCDataset(Dataset):

    def __init__(self, list_file,  args, transform=None):
        self.args = args
        self.list_file = list_file
        self.no_bg = args.no_bg
        self.data_dir = args.data_dir
        self.train_flag = not args.test_flag
        self.size = args.input_size
        self.max_size = [385, 385] # max size for 312 * (0.8, 1.2)
        self.num_classes = args.num_classes
        self.transform = transform
        self.need_mask = args.need_mask_flag
        self.file_list = self._get_file_list()
        self.label_dict = self._get_lable_dict()
        self.img_id_dic_SEC = self._get_img_id_list_SEC()
        self.cues_data_SEC = pickle.load(open(self.args.cues_pickle_dir,'rb'))

    def _get_img_id_list_SEC(self):
        img_id_dic = {}
        with open(self.args.sec_id_img_name_list_dir) as f:
            for line in f:
                (key, val) = line.split()
                img_id_dic[key] = int(val)

        return  img_id_dic

    def _get_file_list(self):
        with open(os.path.join(self.data_dir, "ImageSets", self.list_file),"r") as f:
            all_files = f.readlines()
            return [item.split()[0] for item in all_files]

    def _get_lable_dict(self):
        if (self.no_bg==False or self.train_flag==False):
            with open(
                    os.path.join(
                        self.data_dir, "ImageSets", "class_label.pickle"
                        ),
                    "rb") as f:
                return pickle.load(f)
        else:
            with open(
                    os.path.join(
                        self.data_dir, "ImageSets", "class_label_without_bg.pickle"
                        ),
                    "rb") as f:
                return pickle.load(f)


    def __len__(self):
        return len(self.file_list)


    def __get_cues_from_img_name(self, img_name):
        img_id_sec = self.img_id_dic_SEC[img_name]
        cues = self.cues_data_SEC['%i_cues' % img_id_sec]
        cues_numpy = np.zeros([self.args.num_classes, self.args.output_size[0], self.args.output_size[1]])
        cues_numpy[cues[0], cues[1], cues[2]] = 1.0
        return  cues_numpy.astype('float32')


    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "images", self.file_list[idx]+".png")
        mask_name = os.path.join(self.data_dir, "segmentations", self.file_list[idx]+".png")

        img = Image.open(img_name)
        mask_temp = Image.open(mask_name)
        if self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
            cues_numpy = self.__get_cues_from_img_name(self.file_list[idx]+".png")

        # apply rand flip -------------------------------------------------
        if random.random() > 0.5:
            img =F.hflip(img)
            mask_temp = F.hflip(mask_temp)
            if self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
                cues_numpy = np.flip(cues_numpy,2).copy()

        if self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
            cues = torch.from_numpy(cues_numpy)


        if self.args.origin_size:
            img_array = np.array(img).astype(np.float32)
            mask = np.array(mask_temp)
        else:
            img_array = np.array(img.resize(self.max_size)).astype(np.float32)
            mask = np.array(mask_temp.resize(self.max_size))

        img_ts = self.transform(img)
        if self.args.model=="SEC":
            img_ts = img_ts.float()*255.0
            img_ts= torch.index_select(img_ts, 0, torch.LongTensor([2,1,0]))

        if self.train_flag == False:
            label = np.zeros(21, dtype=np.float32)
        else:
            label = np.zeros(self.num_classes, dtype=np.float32)
        for item in self.label_dict[self.file_list[idx]]:
            label[item] = 1
        label_ts = torch.from_numpy(label)

        if self.train_flag and self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
            return img_ts, label_ts, mask, img_array, cues # img_name,
        else:
            # return img_ts, label_ts, mask, img_name, img_array
            return img_ts, label_ts, mask, img_array


#--modify crf, constrain loss for multi scale training from sec/sec_org_net ------------------------------
class STConstrainLossLayer(nn.Module):
    def __init__(self):
        super(STConstrainLossLayer, self).__init__()

    def forward(self, crf_sm_mask, sm_mask, flag_use_cuda):
        num_pixel = sm_mask.shape[-2]*sm_mask.shape[-1]
        temp = torch.from_numpy(crf_sm_mask.astype('float32'))
        if flag_use_cuda:
            temp = temp.cuda()
        return ((temp * (temp/sm_mask).log()).sum()/num_pixel)/sm_mask.shape[0]


class SeedingLoss(nn.Module):

    def __init__(self):
        super(SeedingLoss, self).__init__()

    def forward(self, sm_mask, cues, flag_use_cuda):
        cur_size = [sm_mask.shape[-2], sm_mask.shape[-1]]
        cues_np_temp = cues.numpy()
        cues_np_resize = np.zeros((cues.shape[0], cues.shape[1], cur_size[0], cur_size[1]),dtype='float32')
        for i in range(sm_mask.shape[0]):
            cues_np_resize[i] = np.round(np.transpose(resize(np.transpose(cues_np_temp[i], (1,2,0)), cur_size), (2,0,1)))

        if flag_use_cuda:
            cues = torch.from_numpy(cues_np_resize).cuda()
        else:
            cues = torch.from_numpy(cues_np_resize)

        count = cues.sum()
        loss = -(cues * sm_mask.log()).sum()/count
        return loss



def crf(sm_mask_one, img_one, num_class, input_size, mask_size, num_iter):
    sm_u = np.transpose(resize(np.transpose(sm_mask_one, [1,2,0]), input_size, mode='constant'),[2,0,1])

    U = unary_from_softmax(sm_u)

    d = dcrf.DenseCRF2D(input_size[0], input_size[1], num_class)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(num_iter)
    return np.array(Q).reshape((num_class, input_size[0], input_size[1]))


class STCRFLayer():
    def __init__(self, flag_multi_process = False):
        self.num_class = 21
        self.min_prob = 0.0001
        self.mask_size = [41, 41]
        self.input_size = [321, 321]
        self.num_iter = 5
        self.flag_multi_process = flag_multi_process
        if flag_multi_process:
            num_cores = os.cpu_count()
            self.pool = Pool(processes=num_cores)

    def run(self, sm_mask, img):
        self.mask_size = [sm_mask.shape[-2], sm_mask.shape[-1]]
        self.input_size = [img.shape[-3], img.shape[-2]]

        if self.flag_multi_process:
            return self.run_parallel(sm_mask, img)
        else:
            return self.run_single(sm_mask, img)

    def run_single(self, sm_mask, img): # the input array are detached numpy already
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)
        for i in range(batch_size):

            sm_u = np.transpose(resize(np.transpose(sm_mask[i], [1,2,0]), self.input_size, mode='constant'),[2,0,1])

            U = unary_from_softmax(sm_u)

            d = dcrf.DenseCRF2D(self.input_size[0], self.input_size[1], self.num_class)
            d.setUnaryEnergy(U)

            d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

            Q = d.inference(self.num_iter)
            result_big[i] = np.array(Q).reshape((self.num_class, self.input_size[0], self.input_size[1]))
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small

    def run_parallel(self, sm_mask, img): # flag_train is for the strange dif between train & test in org SEC
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)

        # temp = self.pool.starmap(self.crf,[(sm_mask[i], img[i]) for i in range(batch_size)])
        temp = self.pool.starmap(crf,[(sm_mask[i], img[i], self.num_class, self.input_size, self.mask_size, self.num_iter) for i in range(batch_size)])
        for i in range(batch_size):
            result_big[i] = temp[i]
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small



# visualize to check if the data loader works as expected.  mostly copied from './multi_scale/train_classifier' ---------------------------
if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = False
    args.model = 'my_resnet'
    args.input_size = [321,321]
    max_size = [385, 385]
    # max_size = [321, 321]
    args.output_size = [41, 41]
    args.need_mask_flag = True
    flag_visualization = True

    host_name = socket.gethostname()
    flag_use_cuda = torch.cuda.is_available()

    if host_name == 'sunting':
        args.batch_size = 1
        args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
        model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/resnet50_feat.pth'
        # args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
        args.cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_02_hard_snapped.pickle"
    elif host_name == 'sunting-ThinkCentre-M90':
        args.batch_size = 2
        args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    elif host_name == 'ram-lab-server01':
        args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
        # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/resnet50_feat.pth'
        model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_rec_my_resnet_9_9.pth'
        args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_cue_02_hard_snapped.pickle"
        args.batch_size = 10

    print(args)

    dataloader = VOCData(args)

    max_acc = 0
    max_recall = 0

    for epoch in range(args.epochs):

        for data in dataloader.dataloaders["train"]:
            inputs, labels, mask_gt, img, cues = data

            # visualization
            if flag_visualization:
                img_np = img.squeeze().numpy().astype('uint8')
                mask_gt_np = mask_gt.squeeze().numpy()
                mask_gt_np[mask_gt_np==255] = 0
                cues_mask_np = np.argmax(cues.squeeze().numpy() ,axis=0)

                plt.figure()
                plt.subplot(1,3,1); plt.imshow(img_np); plt.title('Input image'); plt.axis('off')
                plt.subplot(1,3,2); plt.imshow(mask_gt_np); plt.title('gt'); plt.axis('off')
                plt.subplot(1,3,3); plt.imshow(cues_mask_np); plt.title('cues'); plt.axis('off')
                plt.close('all')

