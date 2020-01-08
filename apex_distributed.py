# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py

import csv

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.distributed
from tqdm import tqdm

from apex import amp
from apex.parallel import DistributedDataParallel

# ATTENTION: 这个没有验证 因为pip安装apex出了问题 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b',
                    '--batch-size',
                    default=6400,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 6400), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

def main():
    args = parser.parse_args()
    main_worker(args.local_rank, 2, args)


def main_worker(gpu, ngpus_per_node, args):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(gpu)

    model = MyModel()
    model.cuda()

    args.batch_size = int(args.batch_size / ngpus_per_node)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    model, optimizer = amp.initialize(model,
                                      optimizer)
    model = DistributedDataParallel(model)
    cudnn.benchmark = True

    train_dataset = MyDataset()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    train_loader2 = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)
    
    for epoch in range(5):
        train_sampler.set_epoch(epoch)
        model.train()

        for i, (data, label) in enumerate(train_loader):
            
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            output = model(data)
            loss = criterion(output, label)

            # print('epoch', epoch, 'gpu', gpu)
            # params = list(model.named_parameters())
            # for i in range(len(params)):
            #     (name, param) = params[i]
            #     print(name)
            #     print(param.grad)

            print('epoch', epoch, 'iter', i, 'gpu', gpu)
            print(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5个epoch 2个gpu 不加控制这个会写10次哦
            # 如果不像每个gpu都做 那么就
            # if gpu == 0:
            #     # with open('./hehe.txt', 'a') as f:
            #     #     f.write(str(gpu)+'\n')
            #     time.sleep(5)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        # print(x.size())
        # print(x)
        return self.net2(self.relu(self.net1(x)))

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(10,10)
        self.data[:,0] = torch.arange(10)
        self.labels = torch.ones(10).long()

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])
 
    def __len__(self):
        return 10

if __name__ == '__main__':
    main()