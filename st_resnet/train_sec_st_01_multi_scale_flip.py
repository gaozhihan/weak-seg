import torch
import torch.optim as optim
import sec.sec_org_net
import multi_scale.voc_data_mul_scale_w_cues
import st_resnet.resnet_st_seg01
import time
import socket
from arguments import get_args
import common_function
import numpy as np
import datetime
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
import multi_scale.STCRF_adaptive01

args = get_args()
args.need_mask_flag = True
args.model = 'my_resnet'
args.input_size = [321,321]
args.output_size = [41, 41]
max_size = [385, 385]
# args.lr = 5e-06
args.CRF_model = 'adaptive_CRF'

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 1
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
    # model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/st_resnet/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_top_val_acc_my_resnet_5_cpu_rename_fc2conv.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_sec01_ws_top_val_iou_my_resnet.pth'
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/multi_scale/models/st_top_val_acc_my_resnet_multi_scale_09_01_cpu_rename_fc2conv.pth'
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/res_from_mul_scale_ws_top_val_iou_my_resnet.pth'
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/my_cues.pickle"
    # args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_01/models/st_cue_01_hard_snapped.pickle"
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/st_resnet/models/st_resnet_cue_01.pickle"
    args.batch_size = 12


net = st_resnet.resnet_st_seg01.resnet50(pretrained=False, num_classes=args.num_classes)
net.load_state_dict(torch.load(model_path), strict = True)

if args.CRF_model == 'adaptive_CRF':
    st_crf_layer = multi_scale.STCRF_adaptive01.STCRFLayer(True)
else:
    st_crf_layer = multi_scale.voc_data_mul_scale_w_cues.STCRFLayer(True)

seed_loss_layer = multi_scale.voc_data_mul_scale_w_cues.SeedingLoss()
# expand_loss_layer = sec.sec_org_net.ExpandLossLayer(flag_use_cuda)
st_constrain_loss_layer = multi_scale.voc_data_mul_scale_w_cues.STConstrainLossLayer()

print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = multi_scale.voc_data_mul_scale_w_cues.VOCData(args)

optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
# main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

max_iou = 0
iou_obj = common_function.iou_calculator()

num_train_batch = len(dataloader.dataloaders["train"])

for epoch in range(args.epochs):
    train_seed_loss = 0.0
    train_expand_loss = 0.0
    train_constraint_loss = 0.0

    train_iou = 0
    eval_iou = 0

    main_scheduler.step()
    start = time.time()

    net.train(True)

    for data in dataloader.dataloaders["train"]:
        inputs, labels, mask_gt, img, cues = data

        # ---- random resize ------------------------------
        rand_scale = random.uniform(0.8, 1.0) #random.uniform(0.67, 1.0)
        cur_size = [round(max_size[0] * rand_scale), round(max_size[1] * rand_scale)]
        inputs_resize = np.zeros((inputs.shape[0], inputs.shape[1], cur_size[0], cur_size[1]),dtype='float32')
        mask_gt_resize = np.zeros((mask_gt.shape[0], cur_size[0], cur_size[1]),dtype='float32')

        max_val = max(max(inputs.max(), -inputs.min()), 1.0).numpy()
        mask_gt_f_temp = mask_gt.detach().numpy().astype('float32')
        max_val_mask = max(mask_gt_f_temp.max(), 1.0)
        mask_gt_f_temp = mask_gt_f_temp/max_val_mask
        img_np = np.zeros((img.shape[0], cur_size[0], cur_size[1], 3))
        img_np_temp = img.detach().numpy()/255.0

        for i in range(inputs.shape[0]):
            inputs_resize[i] = np.transpose(resize(np.transpose(inputs[i].detach().numpy(), (1,2,0))/max_val, cur_size)*max_val, (2,0,1))
            mask_gt_resize[i] = resize(mask_gt_f_temp[i], cur_size, order=0)
            img_np[i] = resize(img_np_temp[i], cur_size)
            # resize(mask_gt[0]/mask_gt[0].max(), cur_size, order=0)*mask_gt[0].max()

        mask_gt_resize = (mask_gt_resize*max_val_mask).astype('uint8')
        img_np = np.round(img_np*255.0)

        if flag_use_cuda:
            inputs = torch.from_numpy(inputs_resize).cuda(); labels = labels.cuda() #; cues = cues.cuda()
        else:
            inputs = torch.from_numpy(inputs_resize)

        optimizer.zero_grad()

        sm_mask = net(inputs)

        if args.CRF_model == 'adaptive_CRF':
            result_big, result_small = st_crf_layer.run(sm_mask.detach().cpu().numpy(), img_np, labels.detach().cpu().numpy())
        else:
            result_big, result_small = st_crf_layer.run(sm_mask.detach().cpu().numpy(), img_np)

        # calculate the SEC loss
        seed_loss = seed_loss_layer(sm_mask, cues, flag_use_cuda)
        constrain_loss = st_constrain_loss_layer(result_small, sm_mask, flag_use_cuda)
        # expand_loss = expand_loss_layer(sm_mask, labels)

        for i in range(labels.shape[0]):
            mask_pre = np.argmax(result_big[i], axis=0)
            iou_obj.add_iou_mask_pair(mask_gt_resize[i,:,:], mask_pre)

            plt.figure()
            plt.subplot(1,3,1); plt.imshow(img[i]/255); plt.title('Input image'); plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(mask_gt[i,:,:].numpy()); plt.title('gt'); plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(mask_pre); plt.title('prediction'); plt.axis('off')
            plt.close('all')

        # for i in range(labels.shape[0]):
        #     temp = np.argmax(result_big[i], axis=0)
        #     plt.subplot(1,4,1); plt.imshow(img[i]/255); plt.title('Input image')
        #     plt.subplot(1,4,2); plt.imshow(mask_gt[i,:,:].numpy()); plt.title('gt')
        #     plt.subplot(1,4,3); plt.imshow(np.argmax(sm_mask[i].detach().cpu().numpy(),axis=0)); plt.title('sm mask')
        #     plt.subplot(1,4,4); plt.imshow(temp); plt.title('sm mask crf')
        #     plt.close("all")



        # (seed_loss + constrain_loss + expand_loss).backward()  # independent backward would cause Error: Trying to backward through the graph a second time ...
        # seed_loss.backward()
        (seed_loss + constrain_loss/8).backward()
        optimizer.step()

        train_seed_loss += seed_loss.item()
        train_constraint_loss += constrain_loss.item()
        # train_expand_loss += expand_loss.item()

    train_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    time_took = time.time() - start
    epoch_train_seed_loss = train_seed_loss / num_train_batch
    # epoch_train_expand_loss = train_expand_loss / num_train_batch
    epoch_train_constraint_loss = train_constraint_loss / num_train_batch

    print('Epoch: {} took {:.2f}, Train seed Loss: {:.4f},  constraint loss: {:.4f}'.format(epoch, time_took, epoch_train_seed_loss, epoch_train_constraint_loss))
    # print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())
    print('cur train iou mean: ', train_iou.mean())

    # if (epoch % 5 == 0):  # evaluation
    net.train(False)
    for data in dataloader.dataloaders["val"]:
        inputs, labels, mask_gt, img = data
        if flag_use_cuda:
            inputs = inputs.cuda(); labels = labels.cuda()

        with torch.no_grad():
            sm_mask = net(inputs)
            result_big, result_small = st_crf_layer.run(sm_mask.detach().cpu().numpy(), img.numpy())

            for i in range(labels.shape[0]):
                mask_pre = np.argmax(result_big[i], axis=0)
                iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)

    eval_iou = iou_obj.cal_cur_iou()
    iou_obj.iou_clear()

    if eval_iou.mean() > max_iou:
        print('save model ' + args.model + ' with val mean iou: {}'.format(eval_iou.mean()))
        torch.save(net.state_dict(), './st_resnet/models/res_from_mul_scale_resnet_cue_01_'+ args.model + '.pth')
        max_iou = eval_iou.mean()

    # print('cur eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())
    print('cur eval iou mean: ', eval_iou.mean())

print("done")



