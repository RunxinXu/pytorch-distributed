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
from torch.utils.data import Dataset
import torch.utils.data.distributed

from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b',
                    '--batch-size',
                    default=3200,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
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

# 这个local_rank 默认成-1非常重要!!!

def main():
    args = parser.parse_args()
    print(args.local_rank) # 在这边其实就有两个输出了 一个是0 一个是1 （双卡情况下）
    main_worker(args.local_rank, 2, args)

def main_worker(gpu, ngpus_per_node, args):

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(gpu) 
    model = MyModel()
    model.cuda(gpu)

    args.batch_size = int(args.batch_size / ngpus_per_node)  # distributeddataparallel的batch_size是单卡的！
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    cudnn.benchmark = True
    
    train_dataset = MyDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # 不要以为这个就没有shuffle了!
    # 有了sampler，shuffle参数没有用
    # 而在sampler中已经默认shuffle了
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)
    # 使用这个的话，就是每个GPU自己要刷完一整个epoch
    train_loader2 = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)

    for epoch in range(5):
        train_sampler.set_epoch(epoch)
        
        model.train()
        for i, (data, label) in tqdm(enumerate(train_loader)):
            data = data.cuda(gpu, non_blocking=True)
            label = label.cuda(gpu, non_blocking=True)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        print('data', self.data)
        print('labels', self.labels)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])
 
    def __len__(self):
        return 10

if __name__ == '__main__':
    main()