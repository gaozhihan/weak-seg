import torch
import torch.optim as optim
import sec.sec_org_net
import multi_scale.voc_data_mul_scale_w_cues
import time
import socket
from arguments import get_args
import common_function
import numpy as np
import datetime
from skimage.transform import resize
import random
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import pickle
from PIL import Image

class VOCData():
    def __init__(self, args, cues_pickle_dir1, cues_pickle_dir2):
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



        self.image_datasets = {x:VOCDataset(x+".txt", args, cues_pickle_dir1, cues_pickle_dir2, self.data_transforms[x])
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

    def __init__(self, list_file,  args, cues_pickle_dir1, cues_pickle_dir2, transform=None):
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
        self.img_id_dic_SEC = self._get_img_id_list_SEC()

        self.cues_data1 = pickle.load(open(cues_pickle_dir1,'rb'))
        self.cues_data2 = pickle.load(open(cues_pickle_dir2,'rb'))

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
        cue_key = '%i_cues' % img_id_sec
        cues1 = self.cues_data1[cue_key]
        cues_numpy1 = np.zeros([self.args.num_classes, self.args.output_size[0], self.args.output_size[1]])
        cues_numpy1[cues1[0], cues1[1], cues1[2]] = 1.0

        cues2 = self.cues_data2[cue_key]
        cues_numpy2 = np.zeros([self.args.num_classes, self.args.output_size[0], self.args.output_size[1]])
        cues_numpy2[cues2[0], cues2[1], cues2[2]] = 1.0

        return  cues_numpy1.astype('float32'), cues_numpy2.astype('float32'), cue_key


    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "images", self.file_list[idx]+".png")
        mask_name = os.path.join(self.data_dir, "segmentations", self.file_list[idx]+".png")
        img = Image.open(img_name)

        if self.args.origin_size:
            img_array = np.array(img).astype(np.float32)
            mask = np.array(Image.open(mask_name))
        else:
            img_array = np.array(img.resize(self.size)).astype(np.float32)
            mask = np.array(Image.open(mask_name).resize(self.size))

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
        if self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
            cues_numpy1, cues_numpy2, cue_key = self.__get_cues_from_img_name(self.file_list[idx]+".png")
            cues1 = torch.from_numpy(cues_numpy1)
            cues2 = torch.from_numpy(cues_numpy2)
            return img_ts, label_ts, mask, img_array, cues1, cues2, cue_key #self.file_list[idx]  # img_name,

        else:
            # return img_ts, label_ts, mask, img_name, img_array
            return img_ts, label_ts, mask, img_array #self.file_list[idx]



#------------------ the main function -------------------------
if __name__ == '__main__':
    args = get_args()
    args.need_mask_flag = True
    args.model = 'my_resnet'
    args.input_size = [321,321]
    args.output_size = [41, 41]
    max_size = [385, 385]
    flag_visual = True


    host_name = socket.gethostname()
    date_str = str(datetime.datetime.now().day)

    if host_name == 'sunting':
        args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
        args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
        cues_pickle_dir1 = "/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_hard_snapped.pickle"
        # cues_pickle_dir2 = "/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_mul_scal_rand_gray_hard_snapped.pickle"
        cues_pickle_dir2 = "/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_mul_scal_pure_gray_hard_snapped.pickle"
        save_cue_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_resnet_cue_01_gray_01_hard_snapped_merge.pickle'
    elif host_name == 'sunting-ThinkCentre-M90':
        args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
        args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
        args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    elif host_name == 'ram-lab-server01':
        args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
        args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
        # cues_pickle_dir1 = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_hard_snapped.pickle"
        # cues_pickle_dir1 = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scal_hard_snapped.pickle"
        cues_pickle_dir1 = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_gray_01_hard_snapped_merge.pickle"
        cues_pickle_dir2 = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scale_pure_gray_01_hard_snapped_merge.pickle"
        # cues_pickle_dir2 = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scal_pure_gray_hard_snapped.pickle"
        # save_cue_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_mul_scale_pure_gray_01_hard_snapped_merge.pickle'
        save_cue_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01_all_hard_snapped_merge_0216.pickle'


    args.batch_size = 1

    print(args)


    dataloader = VOCData(args, cues_pickle_dir1, cues_pickle_dir2)

    with torch.no_grad():

        start = time.time()
        new_cues_dict = {}
        for data in dataloader.dataloaders["train"]:
            inputs, labels, mask_gt, img, cues1, cues2, cue_key = data

            cues1_np = cues1.detach().squeeze().numpy()
            cues2_np = cues2.detach().squeeze().numpy()
            cues_temp = np.maximum(cues1_np, cues2_np)

            if flag_visual:
                plt.subplot(2,4,1); plt.imshow(img[0]/255); plt.title('Input image'); plt.axis('off')
                temp = mask_gt[0,:,:].numpy()
                temp[temp==255] = 0
                plt.subplot(2,4,5); plt.imshow(temp); plt.title('gt'); plt.axis('off')

                plt.subplot(2,4,2); plt.imshow(np.argmax(cues1_np,axis=0)); plt.title('cues1'); plt.axis('off')
                plt.subplot(2,4,6); plt.imshow(cues1_np[0,:,:]); plt.title('bk cues1'); plt.axis('off')

                plt.subplot(2,4,3); plt.imshow(np.argmax(cues2_np,axis=0)); plt.title('cues2'); plt.axis('off')
                plt.subplot(2,4,7); plt.imshow(cues2_np[0,:,:]); plt.title('bk cues2'); plt.axis('off')

                plt.subplot(2,4,4); plt.imshow(np.argmax(cues_temp,axis=0)); plt.title('cues merge'); plt.axis('off')
                plt.subplot(2,4,8); plt.imshow(cues_temp[0,:,:]); plt.title('bk cues merge'); plt.axis('off')

                plt.close('all')

            new_cues_dict[cue_key[0]] = np.asarray(cues_temp.astype('int16').nonzero(), dtype='int16')

        pickling_on = open(save_cue_path,"wb")
        pickle.dump(new_cues_dict, pickling_on)
        pickling_on.close()




