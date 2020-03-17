#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model import DGCNN

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath("__file__"))))

# torch.cuda.manual_seed(1)  # don't delete

grasp_points_num = 750
model = DGCNN(k=20, emb_dims=1024, dropout=0.5, output_channels=2)
model.cuda()

model.eval()
torch.set_grad_enabled(False)


def load_weight(weights_path):
    print('[Python] Load weight: {}'.format(weights_path))
    model.load_state_dict(torch.load(weights_path))


def classify_pcs(local_pcs, output_cls=0):
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
    output1 = list(output.data.numpy()[:, output_cls])
    # print("[output] ", output.data.numpy())
    # print("[output1] ", output1)
    return output1


def main():
    # local_pcs = np.array([
    #     [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]],
    #     [[0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1], [0.2, 0.3, 0.1]],
    #     [[0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7], [0.9, 0.3, 0.7]],
    #     [[0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7], [0.5, 0.4, 0.7]],
    #     [[0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2], [0.8, 0.3, 0.2]]
    #     ])

    local_pcs = torch.ones(5, 1024, 3)
    print(local_pcs, local_pcs.shape)

    weights_path = '/home/sdhm/Projects/SSGPD/Classifier/dgcnn/checkpoints/exp/models/model.t7'
    print('load model: {}'.format(weights_path))
    load_weight(weights_path)

    start = time.time()
    for i in range(100):
        classify_pcs(local_pcs)

    print("FPS:", 100/(time.time()-start))


if __name__ == "__main__":
    main()