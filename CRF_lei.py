import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
from PIL import Image
from functional import generate_u, crf_processing, fast_hist
from skimage.transform import resize

class CAM_iou():
    def __init__(self, labels, mask_gt, mask, img, preds):
        self.inputs_ts = img
        self.cam_maps_ts = mask  #1x21x7x7
        self.num_classes = mask.shape[0]
        self.labels_ts = labels
        self.preds_ts = preds
        self.img_arrays = img # H, W, 3
        self.H = self.img_arrays.shape[0]
        self.W = self.img_arrays.shape[1]
        self.HW =self.H*self.W
        self.masks = mask_gt

        self.present_class_ts = np.nonzero(self.preds_ts.squeeze())

    def get_prob_mask(self):

        selected_cams_ts = self.cam_maps_ts[self.present_class_ts[0], :,:]
        scores_ts = selected_cams_ts * (selected_cams_ts>0)
        scores_ts = np.exp(scores_ts)  # 1*21*41*41
        scores_ts = scores_ts / np.sum(scores_ts, axis=0, keepdims=True)
        # scores_ts = F.upsample(scores_ts.float(), (self.H, self.W), mode='bilinear').squeeze(0).data
        # probs = scores_ts.cpu().numpy()
        eps = 0.00001
        scores_ts[scores_ts < eps] = eps
        probs = np.zeros((len(self.present_class_ts[0]), self.H, self.W))

        for i in range(len(self.present_class_ts[0])):
            probs[i,:,:] = resize(scores_ts[i,:,:], (self.H, self.W))

        return probs.reshape(-1, self.HW), np.argmax(probs, axis=0).reshape(self.HW)


    def run(self):
        probs, showing_mask = self.get_prob_mask()
        showing_mask=crf_processing(probs, self.img_arrays)
        predict_mask = np.zeros(self.HW)
        for i,j in enumerate(self.present_class_ts[0]):
            predict_mask+=(showing_mask==i)*j
        return fast_hist(predict_mask, self.masks, self.num_classes)

