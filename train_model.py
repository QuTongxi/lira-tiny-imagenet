# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)
def select_best_gpu():
    import pynvml
    pynvml.nvmlInit()  # 初始化
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count == 0:
        device = "cpu"
    else:
        gpu_id, max_free_mem = 0, 0.
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_free = round(pynvml.nvmlDeviceGetMemoryInfo(handle).free/(1024*1024*1024), 3)  # 单位GB
            if memory_free > max_free_mem:
                gpu_id = i
                max_free_mem = memory_free
        device = f"cuda:{gpu_id}"
        print(f"total have {gpu_count} gpus, max gpu free memory is {max_free_mem}, which gpu id is {gpu_id}")
    return device


available_device = select_best_gpu()

# 方法1：直接通过os全局设置GPU    
import os
if available_device.startswith("cuda"):
    os.environ['CUDA_VISIBLE_DEVICES'] = available_device.split(":")[1]

import argparse
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from wide_resnet import WideResNet
import matplotlib.pyplot as plt

import sys; sys.path.append('./tiny_imagenet')
from TinyImagenet import * # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--n_shadows", default=None, type=int)
parser.add_argument("--shadow_id", default=0, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="exp/cifar10", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--seed", default=43, type=int)
parser.add_argument("--dpath",default='',type=str)
parser.add_argument("--dataset",default='tiny-imagenet',type=str)
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

def get_dataset(dataset, root):
    if dataset == 'cifar10':
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ]
        )
        datadir = root
        train_ds = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
        return train_ds,test_ds
    elif dataset == 'tiny-imagenet':
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
            ]
        )
        train_ds = TinyImageNet(root, train=True, transform=train_transform) # type: ignore
        test_ds = TinyImageNet(root, train=False, transform=test_transform) # type: ignore
        return train_ds,test_ds
    else:
        raise NotImplementedError


def run():
    pl.seed_everything(args.seed)

    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    # Dataset


    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.
 
    train_ds, test_ds = get_dataset('tiny-imagenet', root=args.dpath)
    size = len(train_ds)

    np.random.seed(args.seed)
    if args.n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)

    _, val_ds = random_split(train_ds, [0.8, 0.2])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model
    if args.model == "wresnet28-2":
        m = WideResNet(28, 2, 0.0, 10)
    elif args.model == "wresnet28-10":
        m = WideResNet(28, 10, 0.3, 10)
    elif args.model == "resnet18":
        nclasses = 0
        if args.dataset == 'tiny-imagenet':
            nclasses = 200
        elif args.dataset == 'cifar10':
            nclasses = 10
        else:
            raise NotImplementedError
        m = models.resnet18(weights=None, num_classes=nclasses)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    else:
        raise NotImplementedError
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    val_acc_list = []
    test_acc_list = []

    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        # 计算并记录验证集和测试集的准确率
        val_acc = get_acc(m, val_dl)
        test_acc = get_acc(m, test_dl)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), val_acc_list, label="Train Accuracy")
    plt.plot(range(1, args.epochs + 1), test_acc_list, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_plot.png")
    plt.show()

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")
    with open(savedir+'/accu.txt', 'a') as f:
        content = f'epochs[{args.epochs}] train accuracy: {get_acc(m, train_dl)} test accuracy: {get_acc(m, test_dl):.4f}\n'
        f.write(content) 


@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    run()
