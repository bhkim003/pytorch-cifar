'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import time
import timeit

from apex.parallel import DistributedDataParallel as DDP


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3"  # Set the GPU 2 to use


""" DDP Setting Start """
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# local_rank는 command line에서 따로 줄 필요는 없지만, 선언은 필요
parser.add_argument("--local_rank", default=0, type=int)
#4개 30초

# User's argument
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args() # 이거 적어줘야됨. parser argument선언하고

args.gpu = args.local_rank
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
args.world_size = torch.distributed.get_world_size()




best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch





# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'





# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/data/cifar10', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
trainloader = torch.utils.data.DataLoader( #여기는 shuffle 옵션 넣지마셈
    trainset, batch_size=128, num_workers=2, sampler=train_sampler)

testset = torchvision.datasets.CIFAR10(
    root='/data/cifar10', train=False, download=True, transform=transform_test)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)
testloader = torch.utils.data.DataLoader( #여기는 shuffle 옵션 넣지마셈
    testset, batch_size=100, num_workers=2, sampler=test_sampler)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck') 


device = args.gpu
net = net.to(args.gpu)
net = DDP(net, delay_allreduce=True)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)







# Training
def train(epoch):
    if torch.distributed.get_rank() == 0:
        start = time.time()
        start_time = timeit.default_timer()

    if torch.distributed.get_rank() == 0:
        print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if torch.distributed.get_rank() == 0:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    if torch.distributed.get_rank() == 0:
        print('epoch', epoch, "time :", time.time() - start,'\n')
        print('Elapsed time for epoch {}: {:.2f}s'.format(epoch, timeit.default_timer() - start_time),'\n')


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if torch.distributed.get_rank() == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()








###################################################################################################
### OJY's DDP setup ###############################################################################
###################################################################################################




# git clone https://github.com/NVIDIA/apex
# cd apex
# # if pip >= 23.1 (근데 나는 23.1보다 높은데 안되던데. 그래서 밑에 있는 걸로함.) refer: https://github.com/NVIDIA/apex?tab=readme-ov-file#quick-start
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# # otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./






# import argparse
# from apex.parallel import DistributedDataParallel as DDP
# ...

# if __name__ == "__main__":
#     """ DDP Setting Start """
#     parser.argparse.ArgumentParser()
#     # local_rank는 command line에서 따로 줄 필요는 없지만, 선언은 필요
#     parser.add_argument("--local_rank", default=0, type=int)
#     # User's argument here
#     ...


#     args = parser.parse_args() # 이거 적어줘야됨. parser argument선언하고

    
#     args.gpu = args.local_rank
#     torch.cuda.set_device(args.gpu)
#     torch.distributed.init_process_group(backend="nccl", init_method="env://")
#     args.world_size = torch.distributed.get_world_size()
    
#     # User's Dataset here
#     train_ds = Dataset(~)
#     val_ds = Dataset(~)
    
#     # shuffle 여부만 sampler에서 설정하고, 나머지는 DataLoader에서 설정
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
#     val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    
#     # DataLoader에 sampler 옵션 추가
#     train_loader = torch.utils.data.DataLoader(train_ds, ~, sampler=train_sampler)
#     val_loader = torch.utils.data.DataLoader(val_ds, ~, sampler=val_sampler)
#     """ DDP Setting End """
#     ...
    
#     model = Net().to(args.gpu)
#     model = DDP(model, delay_allreduce=True)
    
#     ...


# #프린트 주의
#     if torch.distributed.get_rank() == 0:
#         프린트 이 if문에안에서 해야 여러개 안뜸


'''
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py
'''


# --master_port=${PORT}
# 2063인가

# # Example
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
# --nproc_per_node=4 \
# --master_port=1202 \
# main.py