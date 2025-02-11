import torch.nn as nn
from quant import *
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os
import random
import numpy as np
import pytorch_lightning as pl

def seed_all(seed:int): 
    pl.seed_everything(seed)
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.cuda(), y.cuda()
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item() * 100.0



def get_loaders(data:str, nsamples:int, batchsize:int, keep_file:str):
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

    train_dataset = CIFAR10(root=data, train=True, download=True, transform=train_transform)
    assert keep_file is not None
    
    if keep_file is not None:        
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
        
      
    val_dataset = CIFAR10(root=data, train=False, download=True, transform=test_transform)
    def random_subset(data, nsamples):
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        return Subset(data, idx[:nsamples])
    cali_dataset = random_subset(cali_dataset, nsamples)

    cali_loader = DataLoader(cali_dataset, batch_size=batchsize, shuffle=True,num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)   
    return train_loader, cali_loader, val_loader



def replace_with_quant_modules(model, weight_quant_params, act_quant_params):    
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if 'fc' in name:
                wq_params = {'n_bits': 4, 'channel_wise': True, 'scale_method': 'mse', "symmetric": False}
                quant_module = QuantModule(
                    org_module=module,
                    weight_quant_params=wq_params,
                    act_quant_params=act_quant_params
                )
                setattr(model, name, quant_module)
                continue
            quant_module = QuantModule(
                org_module=module,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params
            )
            setattr(model, name, quant_module)
        elif isinstance(module, nn.Sequential):
            replace_with_quant_modules(module, weight_quant_params, act_quant_params)
        elif isinstance(module, nn.Module):
            replace_with_quant_modules(module, weight_quant_params, act_quant_params)
            
def recalibrate_batchnorm(model, dataloader):

    loader = dataloader
    samples = len(dataloader)
    model.cuda()
    model.eval()
    loss = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss += nn.functional.cross_entropy(out, y).item() * x.shape[0]
    
    loss = loss / samples
    print(f'loss: {loss:.4f}')
    
    model.train()
    with torch.no_grad():
        for x,y in loader:
            x = x.cuda()
            model(x)
            
    model.eval() 

    loss = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss += nn.functional.cross_entropy(out, y).item() * x.shape[0]
    
    loss = loss / samples
    print(f'loss: {loss:.4f}')