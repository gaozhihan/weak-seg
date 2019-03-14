# the folder of input images /VOC2012_SEG_AUG/images/ is not complete and lacks test input images
# download the latest test dataset from http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
# the folder /VOC2012/JPEGImages/ contains all test images but in .jpg format
# convert .jpg images to .png format

from PIL import Image
import os

source_dir = '/home/data/gaozhihan/project/weak-seg/weak-seg-aux/VOC2012_SEG_AUG/JPEGImages/'
target_dir = '/home/data/gaozhihan/project/weak-seg/weak-seg-aux/VOC2012_SEG_AUG/images/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

jpg_file_list = os.listdir(source_dir)
for jpg_file in jpg_file_list:
    jpg_file_path = os.path.join(source_dir, jpg_file)
    im = Image.open(jpg_file_path)
    save_path = os.path.join(target_dir, jpg_file[0: -3] + 'png')
    im.save(save_path)