import torch
import torch.nn as nn
import argparse
from torchvision import models
from utils import *
from adaround import layer_reconstruction
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--load',type=str,default='')
parser.add_argument('--datapath',type=str,default='')
parser.add_argument('--wbits',type=int,default=32)
parser.add_argument('--save',type=str,default='')

parser.add_argument('--keep',type=str,default='')
parser.add_argument('--logit_save_path',type=str,default='../dat/')
parser.add_argument('--nqueries',type=int,default=2)

args = parser.parse_args()

seed_all(args.seed)

model= models.resnet18(num_classes = 10, weights = None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.load_state_dict(torch.load(args.load,weights_only=True))
model.cuda()
model.eval()

trainloader, caliloader, testloader = get_loaders(args.datapath, nsamples=1024, batchsize=32,keep_file = args.keep)

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
    save_train_test_accuracy(model, trainloader, testloader)
        
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