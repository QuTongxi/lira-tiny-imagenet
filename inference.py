# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from wide_resnet import WideResNet

import sys; sys.path.append('./tiny_imagenet')
from TinyImagenet import * # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--savedir", default="exp/cifar10", type=str)
parser.add_argument("--dataset",default='tiny-imagenet',type=str)
parser.add_argument("--dpath",default='',type=str)
args = parser.parse_args()

_IMAGENET_RGB_MEANS = [0.4914, 0.4822, 0.4465]
_IMAGENET_RGB_STDS = [0.2470, 0.2435, 0.2616]

def get_dataset(dataset, root):
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_RGB_MEANS, _IMAGENET_RGB_STDS),
            ]
        )
        datadir = root
        train_ds = CIFAR10(root=datadir, train=True, download=True, transform=transform)
        return train_ds
    elif dataset == 'tiny-imagenet':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
            ]
        )  
        train_ds = TinyImageNet(root, train=True, transform=transform) # type: ignore
        return train_ds
    else:
        raise NotImplementedError

@torch.no_grad()
def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    # Dataset
    train_ds = get_dataset(args.dataset, args.dpath)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    for path in os.listdir(args.savedir):
        if args.model == "wresnet28-2":
            m = WideResNet(28, 2, 0.0, 10)
        elif args.model == "wresnet28-10":
            m = WideResNet(28, 10, 0.3, 10)
        elif args.model == "resnet18":
            if args.dataset == 'cifar10':
                m = models.resnet18(weights=None, num_classes=10)
                m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                m.maxpool = nn.Identity()
            elif args.dataset == 'tiny-imagenet':
                m = models.resnet18(weights=None, num_classes=200)
                m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                m.maxpool = nn.Identity()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        m.load_state_dict(torch.load(os.path.join(args.savedir, path, "model.pt")))
        m.to(DEVICE)
        m.eval()

        logits_n = []
        for i in range(args.n_queries):
            logits = []
            for x, _ in tqdm(train_dl):
                x = x.to(DEVICE)
                outputs = m(x)
                logits.append(outputs.cpu().numpy())
            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)
        print(logits_n.shape)

        np.save(os.path.join(args.savedir, path, "logits.npy"), logits_n)


if __name__ == "__main__":
    run()
