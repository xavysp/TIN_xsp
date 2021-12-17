# borrowed from https://github.com/meteorshowers/RCF-pytorch
import os
import json

from torch.utils import data
from os.path import join
from PIL import Image
import numpy as np
import cv2
#from matplotlib import pyplot as plt

def prepare_image_cv2(im):
    im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class Data_Loader(data.Dataset):
    def __init__(self, split='train', scale=None, arg=None):
        self.root = join(arg.datadir,arg.train_data)
        self.split = split
        self.scale = scale
        if self.split == 'train':
            data_name = arg.train_data
            list_file = join(self.root, arg.train_list)
        elif self.split == 'test':
            data_name = arg.test_data
            #self.filelist = join(self.bsds_root, 'image-test.lst')
            if not arg.test_data=="CLASSIC":
                list_file = join(self.bsds_root, arg.test_list)
        else:
            raise ValueError("Invalid split type!")
        if data_name == "CLASSIC":
            # please in your main project dir create a  "data" dir, then paste
            # the images you need in edge-maps
            self.filelist = os.listdir("data")
        else:
            self.filelist =[]
            if data_name in ["BIPED", 'BRIND','MDBD']:
                with open(list_file) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    self.filelist.append(
                        (os.path.join(self.root, tmp_img),
                         os.path.join(self.root, tmp_gt),))
            else:

                with open(list_file, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    self.filelist.append(
                        (os.path.join(self.root, tmp_img),
                         os.path.join(self.root, tmp_gt),))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        r = np.random.randint(0, 100000)
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)

            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = cv2.resize(lb, (256, 256), interpolation=cv2.INTER_LINEAR)

            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < 64)] = 2
            lb[lb >= 64] = 1
            # lb[lb >= 128] = 1
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb
        else:
            img_file, lb_file = self.filelist[index].split()
            data = []
            data_name = []

            original_img = np.array(cv2.imread(join(self.bsds_root, img_file)), dtype=np.float32)
            img = cv2.resize(original_img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

            if self.scale is not None:
                for scl in self.scale:
                    img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
                    data.append(img_scale.transpose(2, 0, 1))
                    data_name.append(img_file)
                return data, img, data_name

            img = prepare_image_cv2(img)

            lb = np.array(Image.open(join(self.bsds_root, lb_file)), dtype=np.float32)

            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = cv2.resize(lb, (256, 256), interpolation=cv2.INTER_LINEAR)
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < 64)] = 2
            lb[lb >= 64] = 1

            return img, lb, img_file

