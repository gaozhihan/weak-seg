# assume can not use rand (eg. randomflip, randomresize etc)
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import pickle
from PIL import Image
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize
import matplotlib.pyplot as plt

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
        cues = self.cues_data_SEC['%i_cues' % img_id_sec]
        cues_numpy = np.zeros([self.args.num_classes, self.args.output_size[0], self.args.output_size[1]])
        cues_numpy[cues[0], cues[1], cues[2]] = 1.0
        return  cues_numpy.astype('float32')

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "images", self.file_list[idx]+".png")
        mask_name = os.path.join(self.data_dir, "segmentations", self.file_list[idx]+".png")
        saliency_name = self.saliency_path + self.file_list[idx] + '.npy'
        saliency_mask = np.load(saliency_name)
        attention_name = self.attention_path + self.file_list[idx] + '.npy'
        attention_mask = np.load(attention_name)
        super_pixel_name = self.super_pixel_path + self.file_list[idx] + '.npy'
        super_pixel = np.load(super_pixel_name)
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

        attention_mask_expand = np.zeros((self.num_classes, attention_mask.shape[1], attention_mask.shape[2]))
        temp = label_ts[1:].nonzero()
        for i, it in enumerate(temp):
            attention_mask_expand[it+1, :,:] = attention_mask[i]


        if self.file_list[idx]+".png" in self.img_id_dic_SEC.keys():
            cues_numpy = self.__get_cues_from_img_name(self.file_list[idx]+".png")
            cues = torch.from_numpy(cues_numpy)
            return img_ts, label_ts, mask, img_array, super_pixel.astype('int32'), saliency_mask.astype('float32'), attention_mask_expand.astype('float32'), cues
        else:
            return img_ts, label_ts, mask, img_array, super_pixel.astype('int32'), saliency_mask.astype('float32'), attention_mask_expand.astype('float32')


# ----------------------------------------------
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

    flag_use_cuda = torch.cuda.is_available()

    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    # args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    args.cues_pickle_dir = "/home/sunting/Documents/program/pyTorch/weak_seg/st_01/models/my_cues.pickle"
    args.batch_size = 1

    print(args)

    dataloader = VOCData(args)

    with torch.no_grad():

        for phase in ['train', 'val']:
            if phase == 'train':

                for data in dataloader.dataloaders["train"]:
                    inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask, cues = data

                    mask_gt_np = mask_gt.squeeze().numpy()
                    img_np = img.squeeze().numpy().astype('uint8')
                    temp_np = attention_mask.squeeze().numpy()
                    thr_value = temp_np.max()*0.3
                    temp_np[temp_np < thr_value] = 0
                    attention_mask_np = np.argmax(temp_np, axis=0)

                    temp_np = cues.squeeze().numpy()
                    cues_np = np.argmax(temp_np, axis=0)
                    temp_mask_gt = mask_gt_np
                    temp_mask_gt[temp_mask_gt==255] = 0

                    plt.subplot(1,4,1); plt.imshow(img_np); plt.title('Input image')
                    plt.subplot(1,4,2); plt.imshow(temp_mask_gt); plt.title('gt')
                    plt.subplot(1,4,3); plt.imshow(attention_mask_np); plt.title('at mask')
                    plt.subplot(1,4,4); plt.imshow(cues_np); plt.title('cues')

                    plt.close('all')

            else:  # evaluation
                for data in dataloader.dataloaders["val"]:
                    inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data



                    plt.close('all')
