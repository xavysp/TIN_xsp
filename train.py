import os, sys, platform
from os.path import join
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from datasets import Data_Loader
from model import TIN
from logger import Logger
from utils import init_model
from torch.utils.tensorboard import SummaryWriter

"""
learning rate adjustment div by 10 every 10 epochs
"""
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-2 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return  lr

def make_optimizer(model, lr):
    optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optim

def save_img_progress(results, filename):
    if not os.path.isdir('img_log'):
        os.mkdir('img_log')

    results_all = torch.zeros((len(results), 1, 256, 256))
    # print(results[0].shape)
    for i in range(len(results)):
        results_all[i, 0, :, :] = results[i][0]
    torchvision.utils.save_image(1 - results_all, join('img_log', "%s.jpg" % filename))

def save_ckpt(model, name, chkpnt_dir=None):
    print('saving checkpoint ... {}'.format(name), flush=True)
    torch.save(model.state_dict(), os.path.join(chkpnt_dir, '{}.pth'.format(name)))

"""
balance cross entropy 
"""
def balanced_cross_entropy_loss(prediction, label):

    if not prediction.shape==label.shape:
        a = torch.nn.UpsamplingBilinear2d([label.size()[2], label.size()[3]])
        prediction = a(prediction)
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)

def batch_bce_loss(prediction, label):

    if not prediction.shape==label.shape:
        a = torch.nn.UpsamplingBilinear2d([label.size()[2], label.size()[3]])
        prediction = a(prediction)
    label = label.long()
    mask = label.float()
    # b, c, h, w = mask.shape
    weight = torch.zeros_like(mask)
    num_positive = torch.sum((mask == 1).float(),dim=[1,2,3], keepdim=True).float()
    num_negative = torch.sum((mask == 0).float(),dim=[1,2,3], keepdim=True).float()
    # num_negative = c*h*w-num_positive

    # weight[label==1]= 1.0 * num_negative / (num_positive + num_negative)
    # weight[label==0]= 1.1 * num_positive / (num_positive + num_negative)
    # weight[label==2]= 0.

    weight.masked_scatter_(label==1,
                           torch.ones_like(label)*(1.0 * num_negative/ (num_positive + num_negative)))
    weight.masked_scatter_(label==0,
        torch.ones_like(label)*(1.1 * num_positive / (num_positive + num_negative)))
    weight[label==2]== 0.

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)
    # return torch.sum(cost) / (num_negative + num_positive)

# ------------------ TIN Setting data -----------------------
IS_LINUX = True if platform.system()=="Linux" else False
dataset_base_dir = '/opt/dataset'if IS_LINUX else 'C:/Users/xavysp/dataset'
parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='results',
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default=dataset_base_dir,
        help='dir to the dataset')
parser.add_argument('--test_data', type=str, default='BRIND',
        help='test data')
parser.add_argument('--train_data', type=str, default='BRIND',
        help='data settings for BSDS, Multicue and NYUD datasets')
parser.add_argument('--train_list', type=str, default='train_pair2.lst',
        help='training data list')
parser.add_argument('--test_list', type=str, default='test_pair.lst',
        help='testing data list')

parser.add_argument('--model', type=str, default='tin',
        help='model to train the dataset') # check later
parser.add_argument('--eta', type=float, default=0.3,
        help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--print-freq', type=int, default=10,
        help='print frequency')
parser.add_argument('--chkpnt', type=str, default='TIN2.pth',
        help='name of checkpoint') # check later


args = parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    """
    data loader
    """

    batch_size = 10
    train_dataset = Data_Loader(split="train", arg=args)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=True)

    log = Logger('log.txt')
    sys.stdout = log

    """
    create model
    """
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    model = TIN(False,2).to(device)
    #conv1_w = model.conv1_1.weight.data
    init_model(model)
    model= model.to(device)
    #conv1_ww = model.conv1_1.weight.data
    #print(conv1_w==conv1_ww)
    # model.cuda()
    model.train()

    """
    PARAMS
    """
    init_lr = 1e-2
    total_epoch = 120
    #####
    each_epoch_iter = len(train_loader)
    total_iter = len(train_loader)# each_epoch_iter//batch_size
    #####
    print('Total inter in 1 batch> ',each_epoch_iter)
    print('batch',batch_size)
    print('Iteration with batch >1', total_iter)


    print_cnt = 10
    ckpt_cnt = 500
    cnt = 0
    avg_loss = 0.

    writer = SummaryWriter()
    optim = make_optimizer(model, init_lr)

    print('*' * 60)
    print('train images in all are %d ' % each_epoch_iter)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total params in all are %d ' % pytorch_total_params)
    print('*' * 60)

    checkpoint_dir = os.path.join("weights", args.train_data)
    os.makedirs(checkpoint_dir, exist_ok=True)

    """
    START TRAINING
    """
    for epoch in range(total_epoch):
        for i, (image, label) in enumerate(train_loader):
            cnt += 1
            if epoch % 10 == 0:
                cur_lr= adjust_learning_rate(optim, epoch)
            if device.type=='cpu':
                image, label = image, label
            else:
                image, label = image.cuda(), label.cuda()
            outs = model(image)
            total_loss = 0

            for each in outs:
                total_loss += balanced_cross_entropy_loss(each, label)/batch_size
                # total_loss += batch_bce_loss(each, label)/batch_size
                # print("Noo")
            # total_loss=total_loss.sum()# just for batch>1
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            avg_loss += float(total_loss)
            if cnt % print_cnt == 0:
                writer.add_scalar('Loss/train', avg_loss / print_cnt, cnt)
                print('[{}/{}] loss:{} avg_loss: {}'.format(i, total_iter, float(total_loss), avg_loss / print_cnt),
                      flush=True)
                avg_loss = 0
                if cnt%100==0:
                    save_img_progress(outs, 'iter-{}'.format(cnt))

        # if cnt % ckpt_cnt == 0:
        save_ckpt(
            model, 'weight-{}-iter-{}'.format(init_lr, epoch+1),
                  chkpnt_dir=checkpoint_dir)

    save_ckpt(model, 'final-model')

if __name__ == '__main__':
    main(args=args)
