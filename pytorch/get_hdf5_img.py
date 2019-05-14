#!/usr/bin/env python3
# coding:utf8
# 提取hdf5文件中存储的图像以及对应的标签

import h5py
import sys
import numpy as np
import cv2

np.set_printoptions(threshold=np.inf)

file_path = sys.argv[1]
h5_file = h5py.File(file_path)
images = h5_file.get('images')
labels = h5_file.get('labels')

print('Have', len(images), 'images')

print(images,"\n",labels)

start_idx , end_idx = 0, 1
images_np = np.array(images[start_idx : end_idx])
images_num = images_np.shape[0]
channels_num = images_np.shape[3]
print(images_np)
print(images_np.shape)

# img = np.zeros((height, width), dtype=np.uint8)

# print(images_np[1])
for img_num in range(images_num):
    print('lable%d:' % (img_num), labels[img_num])
    for ch_num in range(channels_num):
        image_ch_num = images_np[img_num][:, :, ch_num] # num通道对应图像
        # print(image_ch_num) # 打印图片

        cv2.imshow('ch%d' % ch_num, image_ch_num) # 显示图片

    if cv2.waitKey(0) == ord('q'):
        break

