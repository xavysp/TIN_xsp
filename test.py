import torch
import os, platform
from os.path import join
import numpy as np
from PIL import Image
import cv2

import argparse
import scipy.io as io
import time
from model import TIN

# ------------------ TIN Setting data -----------------------
IS_LINUX = True if platform.system()=="Linux" else False
dataset_base_dir = '/opt/dataset'if IS_LINUX else 'C:/Users/xavysp/dataset'
parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='results',
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default=dataset_base_dir,
        help='dir to the dataset')
parser.add_argument('--test_data', type=str, default='BIPED',
        help='test data')
parser.add_argument('--train_data', type=str, default='BIPED',
        help='data settings for BSDS, Multicue and NYUD datasets')
parser.add_argument('--train_list', type=str, default='train_pair.lst',
        help='training data list')
parser.add_argument('--test_list', type=str, default='test_pair.lst',
        help='testing data list')

parser.add_argument('--model', type=str, default='tin',
        help='model to train the dataset') # check later
parser.add_argument('--eta', type=float, default=0.3,
        help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--print-freq', type=int, default=10,
        help='print frequency')


args = parser.parse_args()


test_img = 'img/mri_brain.jpg'
## READ IMAGE
im = np.array(cv2.imread(test_img), dtype=np.float32)
## Multiscale
scales = [0.5,1.0,1.5]
images = []
for scl in scales:
    img_scale = cv2.resize(im, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
    images.append(img_scale.transpose(2, 0, 1)) # (H x W x C) to (C x H x W)

## CREATE MODEL
weight_file = 'weights/TIN2.pth'
model = TIN(False,2)
model.cuda()
model.eval()
#load weight
checkpoint = torch.load(weight_file)
model.load_state_dict(checkpoint)

## FEED FORWARD
h, w, _ = im.shape
ms_fuse = np.zeros((h, w))

with torch.no_grad():
    for img in images:
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        img = img.cuda()
        out = model(img)
        fuse = out[-1].squeeze().detach().cpu().numpy()
        fuse = cv2.resize(fuse, (w, h), interpolation=cv2.INTER_LINEAR)
        ms_fuse += fuse
    ms_fuse /= len(scales)

    filename = 'mri_brain'
    result = Image.fromarray(255-(ms_fuse * 255).astype(np.uint8))
    result.save( "img/result_%s.png" % filename)
print('finished.')

