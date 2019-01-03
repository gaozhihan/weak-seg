import torch
import torch.optim as optim
from sec.sec_data_loader import VOCData
import sec.sec_org_net
import time
import socket
from arguments import get_args
import common_function
import numpy as np
import datetime
from skimage.transform import resize
import matplotlib.pyplot as plt

args = get_args()
args.need_mask_flag = True
args.model = 'SEC'
args.input_size = [321,321]
args.output_size = [41, 41]

host_name = socket.gethostname()
flag_use_cuda = torch.cuda.is_available()
date_str = str(datetime.datetime.now().day)

if host_name == 'sunting':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/pyTorch/weak_seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'sunting-ThinkCentre-M90':
    args.batch_size = 2
    args.data_dir = '/home/sunting/Documents/data/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/home/sunting/Documents/program/weak-seg/sec/input_list.txt"
    args.cues_pickle_dir = "/home/sunting/Documents/program/weak-seg/models/sec_localization_cues/localization_cues.pickle"
    model_path = '/home/sunting/Documents/program/weak-seg/models/vgg16-397923af.pth' # 'vgg16'
elif host_name == 'ram-lab-server01':
    args.data_dir = '/data_shared/Docker/tsun/data/VOC2012/VOC2012_SEG_AUG'
    args.sec_id_img_name_list_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/sec/input_list.txt"
    # model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/vgg16-397923af.pth'
    model_path = '/data_shared/Docker/tsun/docker/program/weak-seg/models/sec_rename_CPU.pth'
    args.cues_pickle_dir = "/data_shared/Docker/tsun/docker/program/weak-seg/models/localization_cues.pickle"
    args.batch_size = 20


# model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # 'vgg16'
net = sec.sec_org_net.SEC_NN()
#net.load_state_dict(model_zoo.load_url(model_url), strict = False)
net.load_state_dict(torch.load(model_path), strict = False)

crf_sec_layer = sec.sec_org_net.CRFLayer()
seed_loss_layer = sec.sec_org_net.SeedingLoss()
expand_loss_layer = sec.sec_org_net.ExpandLossLayer(flag_use_cuda)
constrain_loss_layer = sec.sec_org_net.ConstrainLossLayer()


print(args)
print(model_path)

if flag_use_cuda:
    net.cuda()

dataloader = VOCData(args)

optimizer = optim.Adam(net.parameters(), lr=args.lr)  # L2 penalty: norm weight_decay=0.0001
main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

max_iou = 0
iou_obj = common_function.iou_calculator()

num_train_batch = len(dataloader.dataloaders["train"])

for epoch in range(args.epochs):
    train_seed_loss = 0.0
    train_expand_loss = 0.0
    train_constraint_loss = 0.0

    train_iou = 0
    eval_iou = 0

    # main_scheduler.step()
    # start = time.time()
    #
    # net.train(True)
    #
    # for data in dataloader.dataloaders["train"]:
    #     inputs, labels, mask_gt, img, cues = data
    #     if flag_use_cuda:
    #         inputs = inputs.cuda(); labels = labels.cuda(); cues = cues.cuda()
    #
    #     optimizer.zero_grad()
    #
    #     fc_mask, sm_mask = net(inputs)
    #
    #     for i in range(labels.shape[0]):
    #         temp = np.transpose(sm_mask[i,:,:,:].detach().cpu().numpy(), [1,2,0])
    #         temp = resize(temp, args.input_size, mode='constant')
    #         mask_pre = np.argmax(temp, axis=2)
    #         iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre)
    #
    #     fc_crf = crf_sec_layer.run(fc_mask.detach().cpu().numpy(), img.numpy(), True)
    #     # calculate the SEC loss
    #     seed_loss = seed_loss_layer(sm_mask, cues)
    #     constrain_loss = constrain_loss_layer(fc_crf, sm_mask, flag_use_cuda)
    #     expand_loss = expand_loss_layer(sm_mask, labels)
    #
    #     # for i in range(labels.shape[0]):
    #     #     temp = np.argmax(np.exp(fc_crf_log[i].astype('float32')), axis=0)
    #     #     plt.subplot(1,3,1); plt.imshow(img[i]/255); plt.title('Input image')
    #     #     plt.subplot(1,3,2); plt.imshow(np.argmax(sm_mask[i].detach().cpu().numpy(),axis=0)); plt.title('sm mask')
    #     #     plt.subplot(1,3,3); plt.imshow(temp); plt.title('fc crf log')
    #
    #     (seed_loss + constrain_loss + expand_loss).backward()  # independent backward would cause Error: Trying to backward through the graph a second time ...
    #     optimizer.step()
    #
    #     train_seed_loss += seed_loss.item()
    #     train_constraint_loss += constrain_loss.item()
    #     train_expand_loss += expand_loss.item()
    #
    # train_iou = iou_obj.cal_cur_iou()
    # iou_obj.iou_clear()
    #
    # time_took = time.time() - start
    # epoch_train_seed_loss = train_seed_loss / num_train_batch
    # epoch_train_expand_loss = train_expand_loss / num_train_batch
    # epoch_train_constraint_loss = train_constraint_loss / num_train_batch
    #
    # print('Epoch: {} took {:.2f}, Train seed Loss: {:.4f}, expand loss: {:.4f}, constraint loss: {:.4f}'.format(epoch, time_took, epoch_train_seed_loss, epoch_train_expand_loss, epoch_train_constraint_loss))
    # print('cur train iou is : ', train_iou, ' mean: ', train_iou.mean())

    if (epoch % 50 == 0):  # evaluation
        net.train(False)
        for data in dataloader.dataloaders["val"]:
            inputs, labels, mask_gt, img = data
            if flag_use_cuda:
                inputs = inputs.cuda(); labels = labels.cuda()

            with torch.no_grad():
                fc_mask, sm_mask = net(inputs)
                mask_pre = crf_sec_layer.run(sm_mask.detach().cpu().numpy(), img.numpy(), False)

                for i in range(labels.shape[0]):
                    iou_obj.add_iou_mask_pair(mask_gt[i,:,:].numpy(), mask_pre[i])

        eval_iou = iou_obj.cal_cur_iou()
        iou_obj.iou_clear()

        if eval_iou.mean() > max_iou:
            print('save model ' + args.model + ' with val mean iou: {}'.format(eval_iou.mean()))
            torch.save(net.state_dict(), './sec/models/SEC_top_val_iou_'+ args.model + '.pth')
            max_iou = eval_iou.mean()

        print('cur eval iou is : ', eval_iou, ' mean: ', eval_iou.mean())

print("done")



