import torch
import torch.nn as nn
import argparse
from torchvision import models
from utils import *
from adaround import layer_reconstruction
import os
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
if available_device.startswith("cuda"):
    os.environ['CUDA_VISIBLE_DEVICES'] = available_device.split(":")[1]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--load',type=str,default='')
parser.add_argument('--datapath',type=str,default='')
parser.add_argument('--wbits',type=int,default=32)
parser.add_argument('--save',type=str,default='')

parser.add_argument('--keep',type=str,default='')
parser.add_argument('--logit_save_path',type=str,default='../dat/')
parser.add_argument('--nqueries',type=int,default=2)

parser.add_argument('--dataset',type=str,default='tiny-imagenet')

args = parser.parse_args()

seed_all(args.seed)

nclasses = 0
if args.dataset == 'tiny-imagenet':
    nclasses = 200
elif args.dataset == 'cifar10':
    nclasses = 10
else:
    raise NotImplementedError
model = models.resnet18(weights=None, num_classes=nclasses)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.load_state_dict(torch.load(args.load,weights_only=True))
model.cuda()
model.eval()

# trainloader, caliloader, testloader = get_loaders(args.datapath, nsamples=1024, batchsize=32,keep_file = args.keep)
import sys; sys.path.append('../tiny_imagenet')
from TinyImagenet import *

caliloader, testloader = get_dataloaders(args.dataset, args.datapath, nsamples=-1, batchsize=32, keep_file = args.keep)
trainloader = get_train_dataloader(args.dataset, dpath=args.datapath, batchsize=32)

print(len(caliloader), len(trainloader))

print(f'accuracy: {get_acc(model, testloader):.4f}')

# build quantization parameters
wq_params = {'n_bits': args.wbits, 'channel_wise': True, 'scale_method': 'mse', "symmetric": False}
aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': False}
replace_with_quant_modules(model, wq_params, aq_params)

# start quantization
for module in model.modules():
    if isinstance(module, QuantModule):
        module.set_quant_state(weight_quant=True, act_quant=False)


print(f'evaluating before AdaRound: {get_acc(model, testloader):.2f}')
# apply adaround
def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]

cali_data = get_train_samples(caliloader, num_samples=1024)

with torch.no_grad():
    for inputs, _ in caliloader:
        inputs = inputs.cuda()
        _ = model(inputs)

for name, module in model.named_modules():
    if isinstance(module, QuantModule):
        if module.ignore_reconstruction is True:
            print('Ignore reconstruction of layer {}'.format(name))
            continue
        else:
            print('Reconstruction for layer {}'.format(name))
            layer_reconstruction(model, module, cali_data)    

model.eval()

# recalibrate_batchnorm(model, caliloader)
accu = get_acc(model, testloader)
print(f'evaluating: {accu:.2f}')

def print_fc_stats(fc_layer):
    # 获取全连接层的权重
    weights = fc_layer.weight.data

    # 计算最大值、最小值、均值和方差
    max_value = torch.max(weights).item()
    min_value = torch.min(weights).item()
    mean_value = torch.mean(weights).item()
    var_value = torch.var(weights).item()

    # 打印结果
    print(f"全连接层统计信息:")
    print(f"  - 最大值: {max_value}")
    print(f"  - 最小值: {min_value}")
    print(f"  - 均值: {mean_value}")
    print(f"  - 方差: {var_value}")

print_fc_stats(model.fc)

import matplotlib.pyplot as plt

def plot_fc_distribution(fc_layer, save_path=None):
    """
    以图片形式展示全连接层权重的数据分布。

    参数:
        fc_layer (nn.Linear): 全连接层对象。
        save_path (str, optional): 图片保存路径。如果为 None，则直接显示图片。
    """
    # 获取全连接层的权重
    weights = fc_layer.weight.data.cpu().numpy().flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title("full connection", fontsize=16)
    plt.xlabel("weights", fontsize=14)
    plt.ylabel("frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()

plot_fc_distribution(model.fc, '../quant_fc.png')

def save_train_test_accuracy(model, trainloader, testloader):

    def get_acc(model, dl):
        model.eval()
        acc = []
        for x, y in dl:
            x, y = x.cuda(), y.cuda()
            acc.append(torch.argmax(model(x), dim=1) == y)
        acc = torch.cat(acc)
        acc = torch.sum(acc) / len(acc)
        return acc.item()
    train_acc = get_acc(model, trainloader)
    test_acc = get_acc(model, testloader)
    with open('helper.txt', 'a') as f:
        f.writelines(f'Model_W{args.wbits}A32 train accuracy: {train_acc:.4f} test accuracy: {test_acc:.4f}\n') 


if args.save:
    torch.save(model.state_dict(), args.save)
    save_train_test_accuracy(model, caliloader, testloader)
        
def run_inference(queries, train_dl, model, save_dir):
    model.eval()
    logits_n = []
    for i in range(queries):
        logits = []
        for x, _ in train_dl:
            x = x.cuda()
            outputs = model(x)
            logits.append(outputs.cpu().detach().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)
    
    dir = os.path.join(save_dir, f"AdaRound/W{args.wbits}A32")
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, 'logits.npy')
    np.save(path, logits_n)
    
run_inference(args.nqueries, trainloader, model, args.logit_save_path)