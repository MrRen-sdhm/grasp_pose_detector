#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from os import path
from model.pointnet import PointNetCls
sys.path.append(path.dirname(path.dirname(path.abspath("__file__"))))

grasp_points_num = 750

torch.cuda.manual_seed(1)
np.random.seed(int(time.time()))

model_path = '/home/sdhm/Projects/SSGPD/Classifier/assets/learned_models/state_dict_160.model'
print('load model: {}'.format(model_path))

model = PointNetCls(num_points=grasp_points_num, input_chann=3, k=2)
model.cuda()
model.load_state_dict(torch.load(model_path))

model.eval()
torch.set_grad_enabled(False)


def classify_pcs(local_pcs):
    """ Classify point clouds """
    local_pcs = torch.FloatTensor(local_pcs) # [C, N, 3] C: pointcloud count   N:point num in pointcloud
    inputs = local_pcs.permute(0, 2, 1) # [C, 3, N] C: pointcloud count   N:point num in pointcloud
    inputs = inputs.cuda()
    # concate pointclouds
    # local_pc_list = [local_pc, local_pc, local_pc]
    # inputs = torch.cat(local_pc_list, 0)
    print("local_pcs.shape:", local_pcs.shape)
    print("inputs.shape:", inputs.shape)
    output = model(inputs)
    output = output.softmax(1)
    pred = output.data.max(1, keepdim=True)[1]
    output = output.cpu()
    output1 = list(output.data.numpy()[:, 1])
    print("[output] ", output.data.numpy())
    print("[output1] ", output1)
    print("[pred] ", pred)
    return output1


def main():
    local_pcs = np.array([
        [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]],
        [[0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1]],
        [[0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7]],
        [[0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7]],
        [[0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2]]
        ])
    print(local_pcs)
    classify_pcs(local_pcs)


if __name__ == "__main__":
    main()

