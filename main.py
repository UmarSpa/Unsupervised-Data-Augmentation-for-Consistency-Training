# -*- coding: utf-8 -*-
"""
Unsupervised Data Augmentation
=====================

We load CIFAR10 dataset and train a classification model in semi-supervised
(or supervised) setting.


Input data
----------------
CIFAR10 dataset has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.


Features
----------------
--mod:          default='semisup':          Supervised (sup) or semi-supervised training (semisup)
--sup_num:      default=4000:               Number of samples in supervised training set (out of 50K)
--val_num:      default=1000:               Number of samples in validation set (out of 50K)
--rand_seed:    default=89:                 Random seed for dataset shuffle
--sup_aug:      default=['crop', 'hflip']:  Data augmentation for supervised and unsupervised samples (crop, hflip, cutout, randaug)
--unsup_aug:    default=['randaug']:        Data augmentation (Noise) for unsupervised noisy samples (crop, hflip, cutout, randaug)
--bsz_sup:      default=64:                 Batch size for supervised training
--bsz_unsup:    default=448:                Batch size for unsupervised training
--softmax_temp: default=0.4:                Softmax temperature for target distribution (unsup)
--conf_thresh:  default=0.8:                Confidence threshold for target distribution (unsup)
--unsup_loss_w: default=1.0:                Unsupervised loss weight
--max_iter:     default=500000:             Total training iterations
--vis_idx:      default=10:                 Output visualization index
--eval_idx:     default=1000:               Validation index
--out_dir:      default='./output/':        Output directory


Examples runs
----------------
For semi supervised training:
>> python main.py --mod 'semisup' --sup_num 4000 --sup_aug 'crop' 'hflip' --unsup_aug 'randaug' --bsz_sup 64 --bsz_sup 448

For supervised training:
>> python main.py --mod 'sup' --sup_num 49000 --sup_aug 'randaug' --bsz_sup 64

Notes
----------------
Some of the code for this implementation was borrowed from online sources, as detailed below:

- Wide_ResNet in model.py: https://github.com/wang3702/EnAET/blob/73fd514c74de18c4f7c091012e5cff3a79e1ddbf/Model/Wide_Resnet.py
    - VanillaNet (initially present in guideline code) also works fine. [substitute Wide_ResNet(28, 2, 0.3, 10) with VanillaNet()]

- RandAugment in randAugment.py: https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
    - my own simpler implementation of myRandAugment also works fine. [substitute RandAugment with myRandAugment]

- EMA in ema.py: https://github.com/chrischute/squad/blob/master/util.py#L174-L220

"""
############################ Imports ###################################
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
import argparse
import torchvision
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from ema import EMA
from model import Wide_ResNet, VanillaNet
from randAugment import RandAugment, myRandAugment
from data import CIFAR10Sup, CIFAR10Unsup, CIFAR10Val

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

########################## Global setting ##############################
'''
Global setting is initialized here:
    - Hyper-prameters (dumped in output directory)
    - Ouput directory
    - Tensorboard writer
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mod', default='semisup', type=str,
                    help='Supervised (sup) or semi-supervised training (semisup)')
parser.add_argument('--sup_num', default=4000, type=int,
                    help='Number of samples in supervised training set (out of 50K)')
parser.add_argument('--val_num', default=1000, type=int,
                    help='Number of samples in validation set (out of 50K)')
parser.add_argument('--rand_seed', default=89, type=int,
                    help='Random seed for dataset shuffle')
parser.add_argument('--sup_aug', default=['crop', 'hflip'],  nargs='+',
                    type=str, help='Valid values: crop, hflip, cutout, randaug')
parser.add_argument('--unsup_aug', default=['randaug'],  nargs='+',
                    type=str, help='Valid values: crop, hflip, cutout, randaug')
parser.add_argument('--bsz_sup', default=64, type=int,
                    help='Batch size for supervised training')
parser.add_argument('--bsz_unsup', default=448, type=int,
                    help='Batch size for unsupervised training')
parser.add_argument('--softmax_temp', default=0.4, type=float,
                    help='Softmax temperature for target distribution (unsup)')
parser.add_argument('--conf_thresh', default=0.8, type=float,
                    help='Confidence threshold for target distribution (unsup)')
parser.add_argument('--unsup_loss_w', default=1.0,
                    type=float, help='Unsupervised loss weight')
parser.add_argument('--max_iter', default=500000, type=int,
                    help='Total training iterations')
parser.add_argument('--vis_idx', default=10, type=int,
                    help='Output visualization index')
parser.add_argument('--eval_idx', default=1000,
                    type=int, help='Validation index')
parser.add_argument('--out_dir', default='./output/',
                    type=str, help='Output directory')
args = parser.parse_args()

args.out_dir = '{}{}/'.format(args.out_dir,
                              datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
args.model_path = '{}best_model.pth'.format(args.out_dir)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with open('{}args.txt'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.mod == 'semisup':
    assert args.sup_num == 4000, "Remove assertion if you wish to have semi sup training with sup set != 4K"

if args.mod == 'sup':
    assert args.sup_num == 49000, "Remove assertion if you wish to have sup training with sup set != 49K"

writer = SummaryWriter(args.out_dir)


######################## Data initialization ###########################
'''
Input data is initialized here, along with the train (sup & unsup), valid and test dataloaders:
    - transform_train_sup contains the list of transformations (input params) to be applied to supervised and unsupervised samples.
    - transform_train_unsup contains the list of transformations (input params) to be applied to unsupervised samples (noise injection).
    - transform_test contains the list of transformations (tensor & norm) to be applied to valid and test samples.
'''

args.sup_aug += ["tensor", "normalize"]
args.unsup_aug += ["tensor", "normalize"]

transforms_aug = {"crop": transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                  "hflip": transforms.RandomHorizontalFlip(),
                  "cutout": transforms.RandomErasing(value='random'),
                  "randaug": RandAugment(2, 15),
                  "tensor": transforms.ToTensor(),
                  "normalize": transforms.Normalize((0.49138702, 0.48217663, 0.44645257), (
                      0.24706201, 0.24354138, 0.2616881))}

transform_train_sup = transforms.Compose(
    [transforms_aug[val] for val in args.sup_aug])
transform_train_unsup = transforms.Compose(
    [transforms_aug[val] for val in args.unsup_aug])
transform_test = transforms.Compose(
    [transforms_aug[val] for val in ["tensor", "normalize"]])

trainset_sup = CIFAR10Sup(root='./data', train=True, download=True, transform=[
                          transform_train_sup], sup_num=args.sup_num, random_seed=args.rand_seed)
trainset_unsup = CIFAR10Unsup(root='./data', train=True, download=True, transform=[
                              transform_train_sup, transform_train_unsup], sup_num=args.sup_num, random_seed=args.rand_seed)
validset = CIFAR10Val(root='./data', train=True, download=True, transform=[
                      transform_test], val_num=args.val_num, random_seed=args.rand_seed)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

trainloader_sup = torch.utils.data.DataLoader(
    trainset_sup, batch_size=args.bsz_sup, num_workers=2, drop_last=True)
trainloader_unsup = torch.utils.data.DataLoader(
    trainset_unsup, batch_size=args.bsz_unsup, num_workers=2, drop_last=True)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=4, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


######################### Visualize data ###############################
'''
Some input samples are visualized here:
    - saved in output directory
    - plotted on tensorboard
'''


def unnormalize(img):
    mean = torch.Tensor([0.49138702, 0.48217663, 0.44645257]).unsqueeze(-1)
    std = torch.Tensor([0.24706201, 0.24354138, 0.2616881]).unsqueeze(-1)
    img = (img.view(3, -1) * std + mean).view(img.shape)
    img = img.clamp(0, 1)
    return img


def save_grid(img):
    npimg = img.numpy()
    plt.imsave('{}in_data.jpg'.format(args.out_dir),
               np.transpose(npimg, (1, 2, 0)))


dataiter = iter(trainloader_sup)
images, labels = dataiter.next()
images_grid = torchvision.utils.make_grid(images)
images_grid = unnormalize(images_grid)
save_grid(images_grid)
writer.add_image('input_images', images_grid, 0)


############################# Model ####################################
'''
Classification model is initialized here, along with exponential
moving average (EMA) module:
    - model is pushed to gpu if its available.
'''

net = Wide_ResNet(28, 2, 0.3, 10)  # VanillaNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
ema = EMA(net, decay=0.9999)


############################## Utils ###################################
'''
Training utils are initialized here, including:
    - CrossEntropyLoss - supervised loss.
    - KLDivLoss - unsupervised consistency loss
    - SGD optimizer
    - CosineAnnealingLR scheduler
    - Evaluation function
'''

criterion_sup = torch.nn.CrossEntropyLoss()
criterion_unsup = torch.nn.KLDivLoss(reduction='none')
optimizer = torch.optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.max_iter)


def eval_model(model, valloader, write, writer_id):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on validation set: %.2f %%' %
          (100.0 * correct / total))
    write.add_scalar('validation/Accuracy', 100.0 * correct / total, writer_id)
    model.train()
    return correct


############################ Training ##################################
'''
Training loop containing:
    - data loading
    - optimizer initialization
    - fixed model parameters to generate unsup target logits
    - prediction sharpening of unsup target logits
    - confidence threshold of unsup logits
    - supervised cross entropy loss
    - unsupervised consistency loss
    - exponential moving average of model parameters
    - printing/plotting of the training stats
    - model evaluation every args.eval_idx iterations
'''

running_loss = [0.0, 0.0, 0.0]
best_val = 0

trainloader_sup_iter = iter(trainloader_sup)
if args.mod == 'semisup':
    trainloader_unsup_iter = iter(trainloader_unsup)

for train_idx in range(args.max_iter):
    # data loading
    img_sup, labels_sup = trainloader_sup_iter.next()
    img_sup, labels_sup = img_sup.to(device), labels_sup.to(device)

    if args.mod == 'semisup':
        img_unsup, img_unsup_aug = trainloader_unsup_iter.next()
        img_unsup, img_unsup_aug = img_unsup.to(
            device), img_unsup_aug.to(device)
        img_in = torch.cat([img_sup, img_unsup_aug])
    else:
        img_in = img_sup

    # optimizer initilization
    optimizer.zero_grad()

    if args.mod == 'semisup':
        # fixed parameters of the model to stop gradient back propagation
        with torch.no_grad():
            logits_unsup = net(img_unsup)
            # prediction sharpening
            logits_unsup = logits_unsup / args.softmax_temp
            # confidence threshold (mask)
            conf_mask = F.softmax(logits_unsup, dim=1).max(dim=1)[
                0] > args.conf_thresh

    img_out = net(img_in)
    # supervised loss
    logits_sup = img_out[:args.bsz_sup]
    loss_sup = criterion_sup(logits_sup, labels_sup)

    if args.mod == 'semisup':
        if conf_mask.sum() > 0:
            # Unsupervised consistency loss
            logits_unsup_aug = img_out[args.bsz_sup:]
            loss_unsup = criterion_unsup(F.log_softmax(
                logits_unsup_aug, dim=1), F.softmax(logits_unsup, dim=1))
            loss_unsup = loss_unsup[conf_mask]
            loss_unsup = loss_unsup.sum(dim=1).mean()
        else:
            loss_unsup = 0
        loss = loss_sup + (loss_unsup * args.unsup_loss_w)
    else:
        loss = loss_sup

    # train optimization
    loss.backward()
    optimizer.step()
    scheduler.step()

    # exponential moving average
    ema(net, train_idx // (args.bsz_sup+args.bsz_unsup))

    # print/plot stats
    running_loss[0] += loss.item()
    running_loss[1] += loss_sup.item()
    if args.mod == 'semisup':
        loss_unsup = loss_unsup.item() if type(
            loss_unsup) == torch.Tensor else loss_unsup
        running_loss[2] += loss_unsup

    writer.add_scalar(
        'learning_rate', optimizer.param_groups[0]['lr'], train_idx)
    if train_idx % args.vis_idx == args.vis_idx-1:
        writer.add_scalar('training/total_loss', loss.item(), train_idx)
        writer.add_scalar('training/sup_loss', loss_sup.item(), train_idx)
        if args.mod == 'semisup':
            writer.add_scalar('training/unsup_loss', loss_unsup, train_idx)
            print('[%d] loss: %.3f loss_sup: %.3f loss_unsup: %.3f' % (
                train_idx, running_loss[0] / 100, running_loss[1] / 100, running_loss[2] / 100))
        else:
            print('[%d] loss: %.3f loss_sup: %.3f' %
                  (train_idx, running_loss[0] / 100, running_loss[1] / 100))
        running_loss = [0.0, 0.0, 0.0]

    # eval model
    if train_idx % args.eval_idx == args.eval_idx-1:
        ema.assign(net)
        curr_val = eval_model(net, validloader, writer, train_idx)
        ema.resume(net)
        # save model
        if curr_val > best_val:
            torch.save(net.state_dict(), args.model_path)

    # impose infinite loop
    if train_idx % trainloader_sup_iter.__len__() == trainloader_sup_iter.__len__()-1:
        trainloader_sup_iter = iter(trainloader_sup)
        if args.mod == 'semisup':
            trainloader_unsup_iter = iter(trainloader_unsup)

print('Finished Training')


######################### Model loading ################################
'''
Model loading:
    - Not necessary but kept as it was in the starting code.
'''

net = Wide_ResNet(28, 2, 0.3, 10)
net.load_state_dict(torch.load(args.model_path))


############################# Testing ##################################
'''
Testing loop:
    - kept as it was in the starting code.
'''

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' %
      (100.0 * correct / total))
writer.add_scalar('testing/Accuracy', 100.0 * correct / total, 0)


############################ Class stats ###############################
'''
Class level results:
    - kept as it was in the starting code.
'''

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %.2f %%' %
          (classes[i], 100.0 * class_correct[i] / class_total[i]))
    writer.add_scalar(
        'testing/Accuracy/{}'.format(classes[i]), 100.0 * class_correct[i] / class_total[i], 0)
