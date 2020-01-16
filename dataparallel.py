import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b',
                    '--batch-size',
                    default=20,
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

def main():
    args = parser.parse_args()
    gpus = [0, 1]
    main_worker(gpus=gpus, args=args)

def main_worker(gpus, args):
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model = MyModel()
    model.cuda()
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    cudnn.benchmark = True

    train_dataset = MyDataset()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)

    for epoch in range(5):
        for i, (data, label) in enumerate(train_loader):

            data = torch.randn(args.batch_size, 10).cuda()
            label = torch.ones(args.batch_size).long().cuda()
        
            output = model(data)
            loss = criterion(output, label)

            print('hehe', model.module.count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

        self.count = 0

    def forward(self, x):
        # print(x.size())
        # print(x)
        print('count', self.count)
        self.count += 1
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