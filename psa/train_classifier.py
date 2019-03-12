import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from arguments import get_args
import datetime
import numpy as np
import random


if __name__ == "__main__":
    args = get_args()
    sys.path.append(args.root_dir)

    args.need_mask_flag = False
    args.input_size = [321, 321]
    max_size = [385, 385]
    # max_size = [321, 321]
    args.output_size = [41, 41]
    args.rand_gray = True

    # host_name = socket.gethostname()
    flag_use_cuda = torch.cuda.is_available()
    now = datetime.datetime.now()
    date_str = str(now.day) + '_' + str(now.day)

    print(args)

    if args.model == 'resnet38':
        import psa.network.resnet38_cls as resnet38_cls
        import psa.network.resnet38d as resnet38d
        net = resnet38_cls.Net()
        weights_dict = resnet38d.convert_mxnet_to_torch(args.weights)
        net.load_state_dict(weights_dict, strict=False)
    elif args.model == 'vgg16':
        import psa.network.vgg16 as vgg16
        net = vgg16.Net()
        net.load_state_dict(torch.load(args.weights), strict=False)
    else:
        raise("wrong model settings")

    if args.loss == 'BCELoss':
        criterion = nn.BCELoss()
    elif args.loss == 'MultiLabelSoftMarginLoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    else:
        raise("wrong loss settings")

    if flag_use_cuda:
        net.cuda()

    if args.colorgray == "color":
        from multi_scale.voc_data_mul_scale import VOCData
    elif args.colorgray == "gray":
        from multi_scale.voc_data_mul_scale_all_gray import VOCData
    else:
        raise("wrong color settings")

    dataloader = VOCData(args)

    # L2 penalty: norm weight_decay=0.0001
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    main_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                               step_size=args.step_size)

    max_acc = 0
    max_recall = 0


    for epoch in range(args.epochs):
        train_loss = 0.0
        eval_loss = 0.0
        TP_train = 0
        TP_eval = 0
        T_train = 0
        T_eval = 0
        P_train = 0
        P_eval = 0
        main_scheduler.step()
        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
                for data in dataloader.dataloaders["train"]:
                    inputs, labels = data

                    rand_scale = random.uniform(0.67, 1.0)
                    cur_size = [round(max_size[0] * rand_scale),
                                round(max_size[1] * rand_scale)]

                    max_val = max(max(inputs.max(),
                                      -inputs.min()), 1.0).numpy()

                    # inputs_resize = np.zeros((inputs.shape[0],
                    #                           inputs.shape[1],
                    #                           cur_size[0],
                    #                           cur_size[1]),
                    #                          dtype='float32')

                    # for i in range(inputs.shape[0]):
                    #     inputs_resize[i] = np.transpose(
                    #         resize(np.transpose(
                    #             inputs[i].detach().numpy(),
                    #             (1,2,0))/max_val, cur_size)*max_val, (2,0,1))
                    inputs = F.interpolate(inputs, (cur_size[0],
                                                    cur_size[1]),
                                           mode='bilinear')
                    # plt.imshow(np.transpose(inputs[0].detach().numpy(),
                    #                         (1,2,0)))

                    if flag_use_cuda:
                        # inputs = torch.from_numpy(inputs_resize).cuda()
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # else:
                        # inputs = torch.from_numpy(inputs_resize)

                    optimizer.zero_grad()

                    layer4_feature, fc = net(inputs)
                    preds = torch.sigmoid(fc)

                    loss = criterion(preds.squeeze(), labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)

                    preds_thr_numpy = (preds > args.threshold).cpu().numpy()
                    labels_numpy = labels.detach().cpu().numpy()

                    TP_train += np.logical_and(preds_thr_numpy.squeeze(),
                                               labels_numpy).sum()
                    T_train += labels_numpy.sum()
                    P_train += preds_thr_numpy.sum()

            else:  # evaluation
                net.train(False)
                start = time.time()
                for data in dataloader.dataloaders["val"]:
                    inputs, labels = data
                    if flag_use_cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    with torch.no_grad():
                        layer4_feature, fc = net(inputs)
                        preds = torch.sigmoid(fc)

                    loss = criterion(preds.squeeze(), labels)

                    eval_loss += loss.item() * inputs.size(0)

                    preds_thr_numpy = (preds > args.threshold).cpu().numpy()
                    labels_numpy = labels.detach().cpu().numpy()

                    TP_eval += np.logical_and(preds_thr_numpy.squeeze(),
                                              labels_numpy).sum()
                    T_eval += labels_numpy.sum()
                    P_eval += preds_thr_numpy.sum()

        time_took = time.time() - start
        epoch_train_loss = train_loss / dataloader.dataset_sizes["train"]
        epoch_eval_loss = eval_loss / dataloader.dataset_sizes["val"]

        if flag_use_cuda:
            recall_train = TP_train / T_train if T_train != 0 else 0
            acc_train = TP_train / P_train if P_train != 0 else 0
            recall_eval = TP_eval / T_eval if T_eval != 0 else 0
            acc_eval = TP_eval / P_eval if P_eval != 0 else 0
        else:
            recall_train = TP_train / T_train if T_train != 0 else 0
            acc_train = TP_train / P_train if P_train != 0 else 0
            recall_eval = TP_eval / T_eval if T_eval != 0 else 0
            acc_eval = TP_eval / P_eval if P_eval != 0 else 0

        # print('TP_train: {};   T_train: {};   P_train: {};   acc_train: {};   recall_train: {} '.format(TP_train, T_train, P_train, acc_train, recall_train))
        # print('TP_eval: {};   T_eval: {};   P_eval: {};   acc_eval: {};   recall__eval: {} '.format(TP_eval, T_eval, P_eval, acc_eval, recall_eval))

        if acc_eval > max_acc:
            print('save model ' + args.model + ' with val acc: {}'
                  .format(acc_eval))
            torch.save(net.state_dict(),
                       '{}/psa/weights/psa_{}_top_val_acc_{}_{}.pth'.format(
                        args.root_dir, args.colorgray,
                        args.model, date_str))
            max_acc = acc_eval

        if recall_eval > max_recall:
            print('save model ' + args.model + ' with val recall: {}'
                  .format(recall_eval))
            torch.save(net.state_dict(),
                       '{}/psa/weights/psa_{}_top_val_acc_{}_{}.pth'.format(
                        args.root_dir, args.colorgray, args.model, date_str))
            max_recall = recall_eval

        print('Epoch: {} took {:.2f}, Train Loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}; eval loss: {:.4f}, Acc: {:.4f}, Recall: {:.4f}'.format(epoch, time_took, epoch_train_loss, acc_train, recall_train, epoch_eval_loss, acc_eval, recall_eval))

    print("done")
