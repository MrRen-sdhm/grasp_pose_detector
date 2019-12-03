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
from utils import *
from IPython import embed
from models.model import *
from models.mobilenet import *
from models.mobilenet_v2 import *
from models.resnet import *
from models.densenet import *
from models.inception import *


# set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# test model on public dataset and save the probability matrix
def test(test_loader, model, folds):
    correct = 0
    total = 0
    # confirm the model converted to cuda
    model.cuda()
    model.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        # change everything to cuda
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            # print(inputs, inputs.shape)
            outputs = model(inputs)

            # Mapping outputs to (0,1), The sum of each row is 1
            smax = nn.Softmax(1)
            smax_out = smax(outputs)

            # Return the element with the largest value in each row and it's index
            _, pred_labels = torch.max(smax_out, 1)
            # print("pred_label:", pred_labels, "pred:", pred)

            total += labels.size(0)
            correct += (pred_labels == labels.long()).sum().item()

    accuracy = 100.0 * float(correct) / float(total)
    print("correct:", correct)
    print("total:", total)
    print('Accuracy of the network on the test set: %.3f%%' % (
        accuracy))


# 4. more details to build main function    
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
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)   

    # 4.2 get model
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

    print(model, "\n")
    # model = torch.nn.DataParallel(model)
    model.cuda()

    # load dataset
    with h5py.File(config.test_data, 'r') as db:
        num_test = db.attrs['size']
        #  num_test = 20
        print('test dataset size:', num_test)
    
    test_modle = "./checkpoints/%s/%s/_checkpoint.pt" % (config.model_name, config.fold)
    test_modle = "./checkpoints/best_model/%s/%s/model_best_88.75.pt" % (config.model_name, config.fold)
    print("Test modle:", test_modle)
    best_model = torch.load(test_modle)
    model.load_state_dict(best_model["state_dict"])

    test_dataloader = DataLoader(H5Dataset(config.test_data, 0, num_test), batch_size=20, shuffle=False)
    test(test_dataloader, model, fold)

if __name__ =="__main__":
    main()





















