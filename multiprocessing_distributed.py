# https://github.com/pytorch/examples/blob/master/imagenet/main.py

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
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b',
                    '--batch-size',
                    default=2,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')

def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=2, args=(2, args))

def main_worker(gpu, ngpus_per_node, args):

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=2, rank=gpu)
    torch.cuda.set_device(gpu)

    model = MyModel()
    model.cuda(gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node) # batch size 是单卡的
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

            # output = model.module.hehe(data)
            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            params = list(model.named_parameters())
            (name, param) = params[0]
            print('epoch', epoch, 'gpu', gpu, 'param', param.view(-1)[0].item())

            # 5个epoch 2个gpu 不加控制这个会写10次哦
            # 如果不像每个gpu都做 那么就
            if gpu == 0:
                # with open('./hehe.txt', 'a') as f:
                #     f.write(str(gpu)+'\n')
                time.sleep(5)

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
    
    # 不可以用这个来！！老实用forward的！！
    def hehe(self, x):
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