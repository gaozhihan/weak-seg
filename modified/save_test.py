# helper that helps save the test output with specified format and PIL.Image.palette manipulation
import os
import numpy as np
from PIL import Image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def toRGBarray(arr,classes):
    cmap = color_map(classes)
    rows = arr.shape[0]
    cols = arr.shape[1]
    r = np.zeros(arr.size*3, dtype=np.uint8).reshape(rows,cols,3)
    for i in range(rows):
        for j in range(cols):
            r[i,j] = cmap[arr[i,j]]
    return r

def color_map_viz():
    import matplotlib.pyplot as plt
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

    plt.imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()

class SaveTest(object):
    def __init__(self,  save_dir=None, data_dir=None, file_list='test.txt'):
        if data_dir is None:
            data_dir = "/home/data/gaozhihan/project/weak-seg/weak-seg-aux/VOC2012_SEG_AUG"
        if save_dir is None:
            save_dir = '/home/data/gaozhihan/project/weak-seg/weak-seg-aux/results/VOC2012/Segmentation/comp6_test_cls'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.file_list_path = os.path.join(data_dir, "ImageSets", file_list)
        self.counter = 0
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        with open(self.file_list_path, "r") as f:
            all_files = f.readlines()
            return [item.split()[0] for item in all_files]

    @property
    def _counter_max(self):
        return len(self.file_list)

    def _get_orig_img_path(self):
        ret = os.path.join(self.data_dir, "images", self.file_list[self.counter]+".png")
        return ret

    def _get_save_info(self):
        save_path = os.path.join(self.save_dir, self.file_list[self.counter] + '.png')
        orig_img_path = self._get_orig_img_path()
        orig_img = Image.open(orig_img_path)
        return save_path, orig_img.size

    def save_test(self, mask):
        '''
        :param mask:
            mask_pre in test script
            np.array
            dtype = np.int64
            shape: VOCDataset.max_size, i.e., (385, 385)
        '''
        if self.counter >= self._counter_max:
            raise ValueError('counter larger than the number of test files')
        save_path, save_shape = \
            self._get_save_info()

        img = mask.astype(np.uint8)
        img = toRGBarray(img, 21)
        img = Image.fromarray(img, mode='RGB')
        img = img.resize(size=save_shape)
        palette_img = Image.new('P', (16, 16))
        palette_img.putpalette(color_map(N=21))
        # please refer to link https://stackoverflow.com/questions/29433243/convert-image-to-specific-palette-using-pil-without-dithering/29438149#29438149?newreg=5f7fd6d385d1455baefe9a1d3134cba2
        # im.convert receive no keyword args. 'P' for mode and 0 for dither
        # 3 args to create a new ImagingCore object
        # mode (required): mode string (e.g. "P")
        # dither (optional, default 0): PIL passes 0 or 1
        # paletteimage (optional): An ImagingCore with a palette
        img_im = img.im.convert('P', 0, palette_img.im)
        img = img._new(img_im)
        img.save(save_path)
        self.counter += 1
        return self.counter

if __name__ == '__main__':
    color_map_viz()