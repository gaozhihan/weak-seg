from torchvision.models import vgg16
import torch
import torch.nn as nn
from voc_data import VOCData
#from voc_data_org_size_batch import VOCData
import time
import socket
from arguments import get_args
import matplotlib.pyplot as plt
from skimage.transform import resize
import common_function
import numpy as np
import json
import sec

class Vgg16(nn.Module):
    def __init__(self, flag_classify):
        super(Vgg16, self).__init__()
        vgg16_org = vgg16(pretrained = True)
        self.classify = flag_classify
        self.features = vgg16_org.features

        if self.classify:
            self.classifier = vgg16_org.classifier

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {27,29}:
                results.append(x)

        if self.classify:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            results.append(x)

        return results


class sec_for_saliency(nn.Module):
    def __init__(self, flag_classify, args):
        super(sec_for_saliency, self).__init__()
        net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)
        #net.load_state_dict(torch.load("models/sec_rename_CPU.pth"), strict = True)
        net.load_state_dict(torch.load("models/01/top_val_acc_SEC_CPU.pth"), strict = True)
        net.train(False)
        self.classify = flag_classify
        self.classifier =  net.mask2label_pool
        self.features = net.features

        if self.classify:
            self.classifier =  net.mask2label_pool

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {27,29}: #{36,39}:
                results.append(x)

        if self.classify:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            results.append(x)

        return results


class SEC_for_saliency(nn.Module):
    def __init__(self, flag_classify, args):
        super(SEC_for_saliency, self).__init__()
        self.net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, flag_use_cuda)
        self.net.load_state_dict(torch.load("models/sec_rename_CPU.pth"), strict = False)
        self.net.train(False)
        self.classify = flag_classify

    def forward(self, x):
        with torch.no_grad():
            results = []
            mask, outputs = self.net(x)
            mask_np = mask.numpy()

            # for i in range(mask_np.shape[0]):
            #     temp = mask_np[i,:,:,:]
            #     temp_exp = np.exp(temp - np.max(temp, axis=0, keepdims=True))
            #     temp = temp_exp / np.sum(temp_exp, axis=0, keepdims=True)
            #     mask_np[i,:,:,:] = temp

            mask_np_exp = np.exp(mask_np - np.max(mask_np, axis=1, keepdims=True))
            mask = torch.from_numpy(mask_np_exp / np.sum(mask_np_exp, axis=1, keepdims=True))

        results.append(mask)
        if self.classify:
            results.append(outputs)

        return results



# ---------------- visualize the saliency mask generated from the given feature maps ------------------------
def visual_saliency(outputs, img, mask_gt, flag_classify):
    with torch.no_grad():
        if flag_classify:
            features = outputs[:-1]
            predictions = outputs[-1]
            pred_hard = torch.argmax(predictions, dim =1).numpy()
            class_idx = json.load(open("models/imagenet_class_index.json"))
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

        else:
            features = outputs

        num_maps = len(features)
        num_img = features[0].shape[0]

        shape_mask = features[0][0][0].numpy().shape

        #plt.figure(figsize=((3 + self.num_maps)*5,5))
        for i_img in range(num_img):
            temp_img = img[i_img]
            temp_gt_mask = mask_gt[i_img]
            plt.subplot(1,(4 + num_maps),1); plt.imshow(temp_img/255); plt.title('Input image'); plt.axis('off')
            plt.subplot(1,(4 + num_maps),2); plt.imshow(temp_gt_mask); plt.title('true mask'); plt.axis('off')
            sum_layer_mask_per_img = np.zeros(shape_mask)
            max_layer_mask_per_img = np.zeros(shape_mask)
            if flag_classify:
                print(idx2label[pred_hard[i_img]])

            for i_map in range(num_maps):
                temp_feature = features[i_map][i_img].sum(dim=0).numpy()  #[1:] means no background
                # normalization
                if temp_feature.max() != 0:
                    temp_feature = (temp_feature - temp_feature.min())/temp_feature.max()

                temp = resize(temp_feature, shape_mask, mode='constant')
                sum_layer_mask_per_img = sum_layer_mask_per_img + temp
                max_layer_mask_per_img = np.maximum(max_layer_mask_per_img, temp)
                plt.subplot(1,(4 + num_maps),5+i_map); plt.imshow(temp); plt.axis('off')

            if sum_layer_mask_per_img.max() > 0:
                sum_layer_mask_per_img = (sum_layer_mask_per_img - sum_layer_mask_per_img.min())/sum_layer_mask_per_img.max()

            if max_layer_mask_per_img.max() > 0:
                max_layer_mask_per_img = (max_layer_mask_per_img - max_layer_mask_per_img.min())/max_layer_mask_per_img.max()

            plt.subplot(1,(4 + num_maps),3); plt.imshow(sum_layer_mask_per_img); plt.title('sum'); plt.axis('off')
            plt.subplot(1,(4 + num_maps),4); plt.imshow(max_layer_mask_per_img); plt.title('max'); plt.axis('off')
            print("done")


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

    net = Vgg16(flag_classify)
    #net = SEC_for_saliency(flag_classify, args)

    print(args)

    if flag_use_cuda:
        net.cuda()

    dataloader = VOCData(args)

    net.train(False)
    with torch.no_grad():

        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':

                for data in dataloader.dataloaders["train"]:
                    inputs, labels, mask_gt, img = data
                    if flag_use_cuda:
                        inputs = inputs.cuda(); labels = labels.cuda()

                    outputs = net(inputs)
                    visual_saliency(outputs, img, mask_gt, flag_classify)

            else:  # evaluation
                start = time.time()
                for data in dataloader.dataloaders["val"]:
                    inputs, labels, mask_gt, img = data
                    if flag_use_cuda:
                        inputs = inputs.cuda(); labels = labels.cuda()

                    outputs = outputs.squeeze()
                    visual_saliency(outputs, img, mask_gt, flag_classify)



