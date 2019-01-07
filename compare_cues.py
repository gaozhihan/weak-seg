# assume can not use rand (eg. randomflip, randomresize etc)
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

        if self.train_flag:
            if self.need_mask:
                return img_ts, label_ts, mask, img_array, super_pixel.astype('int32'), saliency_mask.astype('float32'), attention_mask_expand.astype('float32') # img_name,
            else:
                return img_ts, label_ts
        else:
            # return img_ts, label_ts, mask, img_name, img_array
            return img_ts, label_ts, mask, img_array, super_pixel.astype('int32'), saliency_mask.astype('float32'), attention_mask_expand.astype('float32')


# ----------------------------------------------
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
