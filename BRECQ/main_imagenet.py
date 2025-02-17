import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import hubconf
from quant import *
from data.imagenet import build_imagenet_data
import pytorch_lightning as pl

import sys; sys.path.append('../tiny_imagenet')
from TinyImagenet import *

available_device = select_best_gpu()
if available_device.startswith("cuda"):
    os.environ['CUDA_VISIBLE_DEVICES'] = available_device.split(":")[1]

def seed_all(seed=1029):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def get_train_samples(cali_loader, num_samples):
    train_data = []
    for batch in cali_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)
    parser.add_argument('--save', default='q_model.pt', type=str)
    parser.add_argument('--load',type=str,default='')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=2048, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')
    
    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    
    parser.add_argument('--keep',type=str,default='')
    parser.add_argument('--logit_save_path',type=str,default='../dat/')
    parser.add_argument('--nqueries',type=int,default=2)

    parser.add_argument('--dataset',type=str,default='tiny-imagenet')

    args = parser.parse_args()

    seed_all(args.seed)
    # build imagenet data loader
    cali_loader, test_loader = get_dataloaders(args.dataset, args.data_path, -1, args.batch_size, args.keep)
    train_loader = get_train_dataloader(args.dataset, args.data_path, args.batch_size)

    nclasses = 0
    if args.dataset == 'tiny-imagenet':
        nclasses = 200
    elif args.dataset == 'cifar10':
        nclasses = 10
    else:
        raise NotImplementedError

    # load model
    cnn = hubconf.resnet18(pretrained=False, num_classes=nclasses)
    cnn.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    cnn.maxpool = nn.Identity()
    cnn.cuda()
    cnn.load_state_dict(torch.load(args.load, weights_only=True))
    cnn.eval()
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(cali_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))

    if args.test_before_calibration:
        print('Quantized accuracy before brecq: {}'.format(validate_model(test_loader, qnn)))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse')

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)

    # Start calibration
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    accu = validate_model(test_loader, qnn)
    print('Weight quantization accuracy: {}'.format(accu))

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p)
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
                                                               validate_model(test_loader, qnn)))
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
            f.writelines(f'Model_W{args.n_bits_w}A32 train accuracy: {train_acc:.4f} test accuracy: {test_acc:.4f}\n')        
        
    
    if args.save is not None:
        torch.save(qnn.state_dict(), args.save)
        save_train_test_accuracy(qnn, cali_loader, test_loader)
            
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
        
        dir = os.path.join(save_dir, f"BRECQ/W{args.n_bits_w}A32")
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, f'logits.npy')
        np.save(path, logits_n)
        
    run_inference(args.nqueries, train_loader, qnn, args.logit_save_path)