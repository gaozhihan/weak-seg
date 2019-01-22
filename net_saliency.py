from torchvision.models import vgg16
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
import sec
import decoupled_net

class Vgg16_for_saliency(nn.Module):
    def __init__(self, flag_classify):
        super(Vgg16_for_saliency, self).__init__()
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
                results.append(x.detach().numpy())

        if self.classify:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            results.append(x.detach().numpy())

        return results


class sec_for_saliency(nn.Module):
    def __init__(self, flag_classify, args):
        super(sec_for_saliency, self).__init__()
        net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, False)
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
                results.append(x.detach().numpy())

        if self.classify:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            results.append(x.detach().numpy())

        return results


class SEC_for_saliency(nn.Module):
    def __init__(self, flag_classify, args):
        super(SEC_for_saliency, self).__init__()
        self.net = sec.SEC_NN(args.batch_size, args.num_classes, args.output_size, args.no_bg, False)
        self.net.load_state_dict(torch.load("models/sec_rename_CPU.pth"), strict = False)
        self.net.train(False)
        self.classify = flag_classify

    def forward(self, x):
        with torch.no_grad():
            results = []
            mask, outputs = self.net(x)
            mask_np = mask.detach().numpy()

            mask_np_exp = np.exp(mask_np - np.max(mask_np, axis=1, keepdims=True))
            mask = mask_np_exp / np.sum(mask_np_exp, axis=1, keepdims=True)

        results.append(mask[:,1:,:,:])
        if self.classify:
            results.append(outputs.detach().numpy())

        return results


class decouple_net_saliency(nn.Module):
    # for this model, max combine is better
    def __init__(self, flag_classify):
        super(decouple_net_saliency, self).__init__()
        self.decoupled_net = decoupled_net.DecoupleNet(21)
        self.decoupled_net.load_state_dict(torch.load("models/top_val_acc_decoupled_28_CPU.pth"), strict = False)
        self.decoupled_net.train(False)
        self.relu = nn.ReLU()
        self.classify = flag_classify

    def forward(self, x):
        with torch.no_grad():
            results = []

            x = self.decoupled_net.features(x)
            x_np = self.relu(x).cpu().detach().numpy()
            if x_np.max() > 0:
                x_np = x_np/x_np.max()
            results.append(x_np)

            #E-A
            ea_x = self.decoupled_net.drop1(x)
            ea_x = self.decoupled_net.ea_conv(ea_x)
            ea_x = self.decoupled_net.drop2(ea_x)
            temp = ea_x[:,1:,:,:]
            ea_x_np = self.relu(temp).cpu().detach().numpy()
            if ea_x_np.max() > 0:
                ea_x_np = ea_x_np/ea_x_np.max()

            results.append(ea_x_np)

            if self.classify:
                outputs = self.decoupled_net.avg_pool(ea_x)
                results.append(outputs.cpu().detach().numpy())

        return results


class decouple_net_saliency_with_pred(nn.Module):
    # for this model, max combine is better
    def __init__(self, flag_classify):
        super(decouple_net_saliency_with_pred, self).__init__()
        self.decoupled_net = decoupled_net.DecoupleNet(21)
        self.decoupled_net.load_state_dict(torch.load("models/top_val_acc_decoupled_28_CPU.pth"), strict = False)
        self.decoupled_net.train(False)
        self.relu = nn.ReLU()
        self.classify = flag_classify
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        with torch.no_grad():
            results = []

            x = self.decoupled_net.features(x)
            x_np = self.relu(x).cpu().detach().numpy()
            if x_np.max() > 0:
                x_np = x_np/x_np.max()
            results.append(x_np)

            #E-A
            ea_x = self.decoupled_net.drop1(x)
            ea_x = self.decoupled_net.ea_conv(ea_x)
            ea_x = self.decoupled_net.drop2(ea_x)
            temp = ea_x[:,1:,:,:]
            ea_x_np = self.relu(temp).cpu().detach().numpy()
            if ea_x_np.max() > 0:
                ea_x_np = ea_x_np/ea_x_np.max()

            results.append(ea_x_np)

            outputs = self.decoupled_net.avg_pool(ea_x)
            outputs = self.softmax(outputs)

            if self.classify:
                results.append(outputs.cpu().detach().numpy())

        return results, outputs.cpu().detach().squeeze().numpy()


class resnet_saliency(nn.Module):
    def __init__(self, flag_classify):
        super(resnet_saliency, self).__init__()
        self.resnet50 = resnet50(pretrained = True)
        self.classify = flag_classify

    def forward(self, x):
        with torch.no_grad():
            results = []
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)

            results.append(x.detach().numpy())

            if self.classify:
                x = self.resnet50.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.resnet50.fc(x)
                results.append(x.detach().numpy())

        return results

