import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
import numpy as np

def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False, keep_file: str = ''):
    print('==> Using Pytorch Dataset')
    _IMAGENET_RGB_MEANS = [0.4914, 0.4822, 0.4465]
    _IMAGENET_RGB_STDS = [0.2470, 0.2435, 0.2616]
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_RGB_MEANS, _IMAGENET_RGB_STDS),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_RGB_MEANS, _IMAGENET_RGB_STDS),
        ]
    )

    datadir = data_path
    train_dataset = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    if keep_file is not None:        
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
        
    val_dataset = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)

    if dist_sample:
        cali_sampler = torch.utils.data.distributed.DistributedSampler(cali_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        cali_sampler = None
        val_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    cali_loader = torch.utils.data.DataLoader(
        cali_dataset, batch_size=batch_size, shuffle=(cali_sampler is None),
        num_workers=workers, pin_memory=True, sampler=cali_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, cali_loader, val_loader
