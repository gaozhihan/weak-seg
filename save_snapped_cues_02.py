# compare with 'save_snapped_cues.py', use thr_mask_conflict_resolve(snapped_cues_float, labels)
import torch
import torch.nn as nn
import time
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
import pickle
from torch.utils.data import Dataset
import os


class VOCData():
    def __init__(self, args):
        if args.model ==  "SEC":
            if args.test_flag == False:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(args.input_size),
                    # transforms.CenterCrop(224),
                    #transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
                ]),
                }
            else:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(args.input_size),
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
                    transforms.Resize(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(args.input_size),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                }
            else:
                self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(args.input_size),
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
        self.num_classes = args.num_classes
        self.transform = transform
        self.need_mask = args.need_mask_flag
        self.file_list = self._get_file_list()
        self.label_dict = self._get_lable_dict()
        self.saliency_path = args.saliency_dir
        self.attention_path = args.attention_dir
        self.super_pixel_path = args.super_pixel_dir
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
        cue_name = '%i_cues' % img_id_sec
        cues = self.cues_data_SEC[cue_name]
        cues_numpy = np.zeros([self.args.num_classes, self.args.output_size[0], self.args.output_size[1]])
        cues_numpy[cues[0], cues[1], cues[2]] = 1.0
        return  cues_numpy.astype('float32'), cue_name

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "images", self.file_list[idx]+".png")
        mask_name = os.path.join(self.data_dir, "segmentations", self.file_list[idx]+".png")
        super_pixel_name = self.super_pixel_path + self.file_list[idx] + '.npy'
        super_pixel = np.load(super_pixel_name)
        img = Image.open(img_name)
        if self.args.origin_size:
            img_array = np.array(img).astype(np.float32)
            mask = np.array(Image.open(mask_name))
        else:
            img_array = np.array(img.resize(self.size)).astype(np.float32)
            mask = np.array(Image.open(mask_name).resize(self.size))

        if self.train_flag == False:
            label = np.zeros(21, dtype=np.float32)
        else:
            label = np.zeros(self.num_classes, dtype=np.float32)
        for item in self.label_dict[self.file_list[idx]]:
            label[item] = 1
        label_ts = torch.from_numpy(label)


        if self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
            cues_numpy, cue_name = self.__get_cues_from_img_name(self.file_list[idx]+".png")
            cues = torch.from_numpy(cues_numpy)
            return label_ts, mask, img_array, super_pixel.astype('int32'), cues, cue_name
        else:
            return label_ts, mask, img_array, super_pixel.astype('int32')


def snap_to_superpixel(saliency_mask, img, seg):

    if img.max() > 1:
        img = img / 255.0

    img_shape = img.shape[:2]
    saliency_mask_rez = resize(saliency_mask, img_shape, mode='constant')
    saliency_mask_snapped = np.zeros(img_shape)

    num_seg = int(seg.max()) + 1

    if num_seg > 50:
        for i_seg in range(num_seg):
            cur_seg = (seg == i_seg)
            cur_saliency_region = saliency_mask_rez[cur_seg]
            saliency_mask_snapped[cur_seg] = cur_saliency_region.mean()
    else:
        saliency_mask_snapped = saliency_mask_rez

    return saliency_mask_snapped


def snap_cues_to_superpixel(img_np, labels_np, super_pixel_np, cues_np):

    if img_np.max() > 1:
            img_np = img_np / 255.0

    num_class = len(labels_np)
    cur_class = np.nonzero(labels_np)[0]
    num_cur_class = len(cur_class)

    snapped_cues = np.zeros((num_class, img_np.shape[0], img_np.shape[1]))

    for i_class in cur_class:
        snapped_cues[i_class,:,:] = snap_to_superpixel(cues_np[i_class,:,:], img_np.squeeze(), super_pixel_np)

    return snapped_cues


def resize_resolve_conflict(mask, target_size):
    if mask.shape[0] > 21: # only one class present, do nothing
        return resize(mask, target_size, mode='constant')

    else:
        mask = np.transpose(resize(np.transpose(mask, [1,2,0]), target_size, mode='constant'),[2,0,1])
        thr = np.max(mask, axis=0)
        for i in range(mask.shape[0]):
            temp = mask[i, :, :]
            temp[temp<thr] = 0
            mask[i, :, :] = temp

        return mask

def thr_mask_conflict_resolve(snapped_cues_float, labels):

    thr_fg_ratio = 0.3
    thr_bg_ratio = 0.3
    snapped_cues_hard = np.zeros(snapped_cues_float.shape, dtype='int16')
    cur_class = np.nonzero(labels)[0]
    for i_class in cur_class:
        if i_class > 0:
            thr_val = snapped_cues_float[i_class].max() * thr_fg_ratio
        else:
            thr_val = snapped_cues_float[i_class].max() * thr_bg_ratio

        temp = snapped_cues_float[i_class]
        temp[temp < thr_val] = 0
        temp[temp >= thr_val] = 1
        snapped_cues_hard[i_class] = temp.astype('int16')

    # resolve conflict, just background give way for foreground
    temp = snapped_cues_hard[1:,:,:].sum(axis=0)
    temp_idx = temp > 0
    temp = snapped_cues_hard[0]
    temp[temp_idx] = 0
    snapped_cues_hard[0] = temp

    return snapped_cues_hard


if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.test_flag = True
    args = get_args()
    args.origin_size = False
    args.model = 'SEC'
    args.input_size = [321,321]
    args.output_size = [41, 41]
    args.need_mask_flag = True
    thr_ratio = 0.3
    flag_visualization = False

    flag_use_cuda = torch.cuda.is_available()

    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    # args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    args.cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_02_w_conf_thr03.pickle"
    args.batch_size = 1

    # save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/my_cues.pickle'
    # save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_02_hard_snapped.pickle'
    save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/st_cue_02_thr03_hard_snapped.pickle'

    print(args)

    dataloader = VOCData(args)

    with torch.no_grad():
        new_cues_dict = {}

        for data in dataloader.dataloaders["train"]:

            labels, mask_gt, img, super_pixel, cues, cue_name = data

            mask_gt_np = mask_gt.squeeze().numpy()
            img_np = img.squeeze().numpy().astype('uint8')
            cues_np = cues.squeeze().numpy()
            cues_snapped_temp = np.zeros(cues_np.shape)
            labels_np = labels.squeeze().numpy()
            super_pixel_np = super_pixel.squeeze().numpy()

            snapped_cues = snap_cues_to_superpixel(img_np.squeeze(), labels_np, super_pixel_np, cues_np)
            snapped_cues = np.transpose(resize(np.transpose(snapped_cues, [1,2,0]), args.output_size, mode='constant'),[2,0,1])
            snapped_cues_hard = thr_mask_conflict_resolve(snapped_cues, labels.cpu().detach().squeeze().numpy())

            cur_class = np.nonzero(labels_np)[0]
            num_cur_class = len(cur_class)

            temp = mask_gt.squeeze().numpy()
            temp[temp == 255] = 0

            if flag_visualization:
                plt.subplot(2, num_cur_class+1, 1); plt.imshow(img_np); plt.title('img'); plt.axis('off')
                plt.subplot(2, num_cur_class+1, num_cur_class+2); plt.imshow(temp); plt.title('gt'); plt.axis('off')

            for idx, i_class in enumerate(cur_class):
                if flag_visualization:
                    plt.subplot(2, num_cur_class+1, 2+idx); plt.imshow(cues_np[i_class]); plt.title('org cues'); plt.axis('off')

                    if flag_visualization:
                        plt.subplot(2, num_cur_class+1, num_cur_class+3+idx); plt.imshow(snapped_cues_hard[i_class]); plt.title('snapped cues'); plt.axis('off')
                else:
                    if flag_visualization:
                        plt.subplot(2, num_cur_class+1, num_cur_class+3+idx); plt.imshow(snapped_cues[idx]); plt.title('snapped cues'); plt.axis('off') # view

            if flag_visualization:
                plt.close('all')

            new_cues_dict[cue_name[0]] = np.asarray(snapped_cues_hard.nonzero(), dtype='int16')

        pickling_on = open(save_cue_path,"wb")
        pickle.dump(new_cues_dict, pickling_on)
        pickling_on.close()

