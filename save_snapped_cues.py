# only test iou of the sm_mask
import torch
import torch.nn as nn
from compare_cues import VOCData
import time
import socket
import st_01.sec_net
from arguments import get_args
import datetime
import numpy as np
import common_function
from skimage.transform import resize
import matplotlib.pyplot as plt




input_size = [321,321]
output_size = [41, 41]

data_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG'
super_pixel_dir = '/home/sunting/Documents/program/VOC2012_SEG_AUG/super_pixel/'
sec_id_img_name_list_dir = "/home/sunting/Documents/program/SEC-master/training/input_list.txt"
cues_pickle_dir = "/home/sunting/Documents/program/SEC-master/training/localization_cues/localization_cues.pickle"

file_list = self._get_file_list()
def _get_file_list(self):
        with open(os.path.join(self.data_dir, "ImageSets", self.list_file),"r") as f:
            all_files = f.readlines()
            return [item.split()[0] for item in all_files]
