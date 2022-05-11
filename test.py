import torch
import os, platform
from os.path import join
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from datasets import Data_Loader

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
parser.add_argument('--test_data', type=str, default='BSDS',
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

parser.add_argument('--chckpnt', type=str, default='weight-0.01-iter-20.pth',
        help='name of checkpoint') # check later


args = parser.parse_args()


# test_img = 'img/mri_brain.jpg'
# ## READ IMAGE
# im = np.array(cv2.imread(test_img), dtype=np.float32)
# ## Multiscale
# scales = [0.5,1.0,1.5]
# images = []
# for scl in scales:
#     img_scale = cv2.resize(im, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
#     images.append(img_scale.transpose(2, 0, 1)) # (H x W x C) to (C x H x W)
def main(args):

    """
    data loader
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = 1
    test_dataset = Data_Loader(split="test", arg=args)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=False)

    ## CREATE MODEL
    weight_file = join('weights',args.train_data,args.chckpnt)

    device = torch.device('cpu' if torch.cuda.device_count() == 0
                              else 'cuda')

    model = TIN(False,2).to(device)
    model.eval()
    #load weight

    checkpoint = torch.load(weight_file,map_location=device)
    model.load_state_dict(checkpoint)

    ## FEED FORWARD
    # h, w, _ = im.shape
    # ms_fuse = np.zeros((h, w))
    save_dir = os.path.join('results',args.train_data+str(2)+args.test_data)
    os.makedirs(save_dir,exist_ok=True)
    print("Data will be saved in> ",save_dir)
    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(test_loader):
            img = sample_batched['images'].to(device)
            img_shape = sample_batched['image_shape']
            file_name = sample_batched['file_names']
            file_name = os.path.basename(file_name[0])
            filename,_= os.path.splitext(file_name)
            h,w,c = img_shape
            h = h.numpy()[0]
            w = w.numpy()[0]

            print("Image size ", img.shape)
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            out = model(img)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)

            fuse = out[-1].squeeze().detach().cpu().numpy()
            # print('w', w[0],'h',h[0])
            fuse = cv2.resize(fuse, (w, h), interpolation=cv2.INTER_LINEAR)

            result = Image.fromarray(255-(fuse * 255).astype(np.uint8))
            filename = filename+'.png'
            result.save(os.path.join(save_dir,filename))
            print(batch_id, ": ",img_shape,os.path.join(save_dir,filename))
    print('finished.')
    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", len(test_loader), "images. *****")
    print("FPS: %f.4" % (len(test_loader)/total_duration))



if __name__ == '__main__':
    main(args=args)
