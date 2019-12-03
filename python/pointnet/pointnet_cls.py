#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# import os  # don't delete
import torch
import torch.utils.data
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import sys
from os import path
from model.pointnet import PointNetCls
sys.path.append(path.dirname(path.dirname(path.abspath("__file__"))))

# torch.cuda.manual_seed(1)  # don't delete

grasp_points_num = 750
model = PointNetCls(num_points=grasp_points_num, input_chann=3, k=2)
model.cuda()

model.eval()
torch.set_grad_enabled(False)


def load_weight(weights_path):
    print('[Python] Load weight: {}'.format(weights_path))
    model.load_state_dict(torch.load(weights_path))


def classify_pcs(local_pcs):
    """ Classify point clouds FPS:680 """
    print("[Python] Classify point clouds")

    local_pcs = torch.FloatTensor(local_pcs)  # [C, N, 3] C: pointcloud count   N:point num in pointcloud
    inputs = local_pcs.permute(0, 2, 1)  # [C, 3, N] C: pointcloud count   N:point num in pointcloud
    inputs = inputs.cuda()
    # concate pointclouds
    # local_pc_list = [local_pc, local_pc, local_pc]
    # inputs = torch.cat(local_pc_list, 0)
    print("[Python] local_pcs.shape:", local_pcs.shape)
    print("[Python] inputs.shape:", inputs.shape)
    output = model(inputs)
    output = output.softmax(1)
    # pred = output.data.max(1, keepdim=True)[1]
    # print("[pred] ", pred)

    output = output.cpu()
    output1 = list(output.data.numpy()[:, 1])
    # print("[output] ", output.data.numpy())
    # print("[output1] ", output1)
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

    weights_path = '/home/sdhm/Projects/SSGPD/Classifier/assets/learned_models/state_dict_160.model'
    print('load model: {}'.format(weights_path))
    load_weight(weights_path)
    for i in range(10000):
        classify_pcs(local_pcs)


if __name__ == "__main__":
    main()

