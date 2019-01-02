import torch
import torch.nn as nn
import torch.optim as optim
from voc_data_w_superpixel_snapped_at_sal import VOCData
import time
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize

args = get_args()
args.origin_size = False
args.model = 'SEC'
args.input_size = [321,321]
args.output_size = [41, 41]
args.need_mask_flag = True

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
now = datetime.datetime.now()
date_str = str(now.day) + '_' + str(now.month)

if host_name == 'sunting':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/snapped_attention/'
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.super_pixel_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG/snapped_attention/'
    model_path = '/home/sunting/Documents/program/weak-seg/models/sec_rename_CPU.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_01_top_val_rec_SEC_31_31.pth'
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/sec_rename_CPU.pth'
    args.super_pixel_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/super_pixel/'
    args.saliency_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_saliency/'
    args.attention_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG/snapped_attention/'
    args.batch_size = 24

net = st_01.sec_net.SEC_NN()
net.load_state_dict(torch.load(model_path), strict = False)

criterion_BCE = nn.BCELoss()
criterion_seed = st_01.sec_net.SeedingLoss()

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)

optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

max_acc = 0
max_recall = 0

max_iou = 0
iou_obj = common_function.iou_calculator()

for epoch in range(args.epochs):

    train_iou = 0
    eval_iou = 0

    train_seed_loss = 0.0
    train_BCE_loss = 0.0
    eval_seed_loss = 0.0
    eval_BCE_loss = 0.0
    TP_train = 0; TP_eval = 0
    T_train = 0;  T_eval = 0
    P_train = 0;  P_eval = 0
    main_scheduler.step()

    start = time.time()
    net.train(True)

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()  # saliency_mask = saliency_mask.cuda(); attention_mask = attention_mask.cuda()

        optimizer.zero_grad()

        sm_mask, preds = net(inputs)

        for i in range(labels.shape[0]):
            temp = np.transpose(sm_mask[i,:,:,:].detach().cpu().numpy(), [1,2,0])
            temp = resize(temp, args.input_size, mode='constant')
            mask_pre = np.argmax(temp, axis=2)
            iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)


        loss_BCE = criterion_BCE(preds.squeeze(), labels)
        loss_seed = criterion_seed(sm_mask, attention_mask, labels, super_pixel, flag_use_cuda)

        (loss_BCE + loss_seed).backward()
        optimizer.step()

        train_BCE_loss += loss_BCE.item() * inputs.size(0)
        train_seed_loss += loss_seed.item()

        preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy()
        labels_numpy = labels.detach().cpu().numpy()

        TP_train += np.logical_and(preds_thr_numpy.squeeze(),labels_numpy).sum()
        T_train += labels_numpy.sum()
        P_train += preds_thr_numpy.sum()

    train_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    recall_train = TP_train / T_train if T_train!=0 else 0
    acc_train = TP_train / P_train if P_train!=0 else 0
    time_took = time.time() - start
    epoch_train_BCE_loss = train_BCE_loss / dataloader.dataset_sizes["train"]
    epoch_train_seed_loss = train_seed_loss / dataloader.dataset_sizes["train"]

    print('Epoch: {} took {:.2f}, Train seed Loss: {:.4f}, BCE loss: {:.4f}, acc: {:.4f}, rec: {:.4f}'.format(epoch, time_took, epoch_train_seed_loss, epoch_train_BCE_loss, acc_train, recall_train))
    print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())


    # evaluation every 50 epoch ---------------------------------------------------------------
    if (epoch % 50 == 0):  # evaluation
        net.train(False)
        start = time.time()
        for data in dataloader.dataloaders["val"]:
            inputs, labels, mask_gt, img, super_pixel, saliency_mask, attention_mask = data
            if flag_use_cuda:
                inputs = inputs.cuda(); labels = labels.cuda()  # saliency_mask = saliency_mask.cuda(); attention_mask = attention_mask.cuda()

            with torch.no_grad():
                sm_mask, preds = net(inputs)

            for i in range(labels.shape[0]):
                temp = np.transpose(sm_mask[i,:,:,:].detach().cpu().numpy(), [1,2,0])
                temp = resize(temp, args.input_size, mode='constant')
                mask_pre = np.argmax(temp, axis=2)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

            loss_BCE = criterion_BCE(preds.squeeze(), labels)
            loss_seed = criterion_seed(sm_mask, attention_mask, labels, super_pixel, flag_use_cuda)

            eval_BCE_loss += loss_BCE.item() * inputs.size(0)
            eval_seed_loss += loss_seed.item()


            preds_thr_numpy = (preds.data>args.threshold).detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()

            TP_eval += np.logical_and(preds_thr_numpy.squeeze(),labels_numpy).sum()
            T_eval += labels_numpy.sum()
            P_eval += preds_thr_numpy.sum()

        eval_iou = iou_obj.cal_cur_iou()
        iou_obj.iou_clear()

        time_took = time.time() - start
        epoch_eval_BCE_loss = eval_BCE_loss / dataloader.dataset_sizes["val"]
        epoch_eval_seed_loss = eval_seed_loss / dataloader.dataset_sizes["val"]

        recall_eval = TP_eval / T_eval if T_eval!=0 else 0
        acc_eval = TP_eval / P_eval if P_eval!=0 else 0

        if eval_iou.mean() > max_iou:
            print('save model ' + args.model + ' with val mean iou: {}'.format(eval_iou.mean()))
            torch.save(net.state_dict(), './st_01/models/top_val_iou_ws'+ args.model + '.pth')
            max_iou = eval_iou.mean()

        print('Epoch: {} took {:.2f}, eval seed Loss: {:.4f}, BCE loss: {:.4f}, acc: {:.4f}, rec: {:.4f}'.format(epoch, time_took, epoch_eval_seed_loss, epoch_eval_BCE_loss, acc_eval, recall_eval))
        print('cur eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())

print("done")
