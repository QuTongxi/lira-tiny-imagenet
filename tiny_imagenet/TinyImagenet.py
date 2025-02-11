import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
from torchvision.datasets import CIFAR10

_IMAGENET_RGB_MEANS = [0.4914, 0.4822, 0.4465]
_IMAGENET_RGB_STDS = [0.2470, 0.2435, 0.2616]

tiny_train_trans = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ]
)

tiny_test_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ]
)

cifar10_train_trans = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_RGB_MEANS, _IMAGENET_RGB_STDS),
    ]
)

cifar10_test_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_RGB_MEANS, _IMAGENET_RGB_STDS),
    ]
)

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

def get_cifar10_loaders(dpath:str, nsamples:int, batchsize:int, keep_file:str):
    train_dataset = CIFAR10(root=dpath, train=True, download=True, transform=cifar10_train_trans)
    assert keep_file is not None
    
    if keep_file is not None:        
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
        
      
    val_dataset = CIFAR10(root=dpath, train=False, download=True, transform=cifar10_test_trans)
    def random_subset(data, nsamples):
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        return Subset(data, idx[:nsamples])
    if nsamples != -1:
        cali_dataset = random_subset(cali_dataset, nsamples)

    cali_loader = DataLoader(cali_dataset, batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)   
    return cali_loader, val_loader

def get_tiny_imagenet_loaders(dpath:str, nsamples:int, batchsize:int, keep_file:str):
    train_dataset = TinyImageNet(dpath, train=True, transform=tiny_train_trans)
    test_dataset = TinyImageNet(dpath, train=False, transform=tiny_test_trans)
    # assert keep_file is not None
    
    if keep_file is not None:        
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
    else:
        cali_dataset = train_dataset
        
    def random_subset(data, nsamples):
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        return Subset(data, idx[:nsamples])
    if nsamples != -1:
        cali_dataset = random_subset(cali_dataset, nsamples)

    cali_loader = DataLoader(cali_dataset, batch_size=batchsize, shuffle=True,num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)   
    return cali_loader, test_loader

def get_dataloaders(dataset:str, dpath:str, nsamples:int, batchsize:int, keep_file:str):
    if dataset == 'cifar10':
        return get_cifar10_loaders(dpath=dpath,nsamples=nsamples,batchsize=batchsize,keep_file=keep_file)
    elif dataset == 'tiny-imagenet':
        return get_tiny_imagenet_loaders(dpath=dpath,nsamples=nsamples,batchsize=batchsize,keep_file=keep_file)
    else:
        raise NotImplementedError

def get_train_dataloader(dataset:str, dpath:str, batchsize:int):
    if dataset == 'cifar10':
        train_set = CIFAR10(root=dpath, train=True,transform=cifar10_train_trans)
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)
        return train_loader
    elif dataset == 'tiny-imagenet':
        train_set = TinyImageNet(dpath, train=True, transform=tiny_train_trans)
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True)
        return train_loader
    else:
        raise NotImplementedError    

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
    