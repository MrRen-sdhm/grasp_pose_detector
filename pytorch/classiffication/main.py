import os
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn,optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from tensorboardX import SummaryWriter
from utils import *
from IPython import embed
from models.model import *
from models.mobilenet import *
from models.mobilenet_v2 import *
from models.resnet import *
from models.densenet import *
from models.inception import *

import time


# set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

log_path = os.path.join(config.logs, config.model_name)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)


# evaluate func
def evaluate(val_loader, model, criterion, epoch, lr, resume):
    # define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    # progress bar
    val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=config.epochs,
        model_name=config.model_name, total=len(val_loader), resume=resume)
    # switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    # model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current = i
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            # compute output
            output = model(input)
            loss = criterion(output,target)

            # measure accuracy and record loss
            precision1, precision2 = accuracy(output, target, topk=(1,2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            val_progressor.current_lr = lr
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()

        val_progressor.done()
        logger.add_scalar('test_acc', top1.avg, epoch)
        logger.add_scalar('test_loss', losses.avg, epoch)

    return [losses.avg, top1.avg]


# more details to build main function
def main():
    fold = config.fold
    # mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + fold + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep + fold + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + fold + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep + fold + os.sep)

    # get model and optimizer
    if config.model_name is "lenet":
        model = get_lenet(config.img_channels)
    elif config.model_name is "mobilenet":
        model = get_mobilenet(config.img_channels)
    elif config.model_name is "mobilenet_v2":
        model = mobilenet_v2(config.img_channels)
    elif config.model_name is "resnet":
        model = resnet18()
    elif config.model_name is "densenet":
        model = densenet121()
    elif config.model_name is "inception_v3":
        model = inception_v3()


    # from thop import profile
    # input = torch.randn(1, 12, 60, 60)

    # flops, params = profile(model, inputs=(input,))
    # print("[DEBUG] Number of FLOPs: %.2fM" % (flops / 1e6))  # 12ch: 36.91M  3ch: 22.80M
    # print("[DEBUG] Number of parameter: %.2fM" % (params / 1e6))  # 12ch: 3.63M  3ch: 3.63M


    # import time
    # model.cuda()
    # input = Variable(input).cuda()
    # sum = 0
    # for i in range(10000):
    #     start = time.time()
    #     output = model(input)
    #     sum += time.time()-start
    # print("forward time:", (sum / 10000) * 1000)  # 12ch: 0.2694ms  3ch: 0.2631ms


    print(model, "\n")
    # model = torch.nn.DataParallel(model)

    exit()
    model.cuda()

    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr = config.lr, amsgrad=True, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    # some parameters for restart model
    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0
    resume = False

    # restart the training process
    if resume:
        resume_file = config.best_models + config.model_name + "/" + fold + "/model_best_89.32.pt"
        print("resume" + resume_file)
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # load dataset
    with h5py.File(config.train_data, 'r') as db:
        num_train = db.attrs['size']
        # num_train = 10
        print('train dataset size:', num_train)
    train_dataloader = DataLoader(H5Dataset(config.train_data, 0, num_train), batch_size=config.batch_size, shuffle=True)
    with h5py.File(config.val_data, 'r') as db:
        num_val = db.attrs['size']
        print('val   dataset size:', num_val)

    val_dataloader = DataLoader(H5Dataset(config.val_data, 0, num_val), batch_size=config.batch_size * 2, shuffle=True)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", verbose=1, patience=3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

    # define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf,0,0]
    model.train()

    # train
    start = timer()
    for epoch in range(start_epoch, config.epochs):
        scheduler.step(epoch)
        lr = get_learning_rate(optimizer)
        # print("learning rate:", lr)

        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=config.epochs,
            model_name=config.model_name, total=len(train_dataloader), resume=resume)
        # train
        # global iter
        for iter, (input, target) in enumerate(train_dataloader):
            # switch to continue train process
            train_progressor.current = iter
            model.train()
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            starttime = time.time()
            output = model(input)
            endtime = time.time()
            print("forward time:", (endtime - starttime)*1000)

            loss = criterion(output,target)

            precision1_train, precision2_train = accuracy(output, target, topk=(1,2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0], input.size(0))
            train_progressor.current_lr = lr
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_progressor()

            logger.add_scalar('train_loss', train_losses.avg, iter + epoch * len(train_dataloader))
            logger.add_scalar('train_acc', train_top1.avg, iter + epoch * len(train_dataloader))

        train_progressor.done()

        # evaluate every epoch
        valid_loss = evaluate(val_dataloader, model, criterion, epoch, lr, resume)
        is_best = valid_loss[1] > best_precision1
        best_precision1 = max(valid_loss[1], best_precision1)
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_precision1":best_precision1,
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "valid_loss":valid_loss,
        },is_best,fold)

if __name__ =="__main__":
    main()





















