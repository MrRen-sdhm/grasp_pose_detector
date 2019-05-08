#!/usr/bin/env python3
# coding:utf8
# 功能：提取bigbird_pcds文件夹下各物体, 不同相机、不同角度的pcd, 复制并重命名

import numpy as np
import random
import shutil
import os

obj_name_list = ["red_bull", "advil_liqui_gels", "band_aid_clear_strips", "blue_clover_baby_toy",
    "bumblebee_albacore", "campbells_soup_at_hand_creamy_tomato", "colgate_cool_mint", "crayola_24_crayons",
    "crest_complete_minty_fresh", "dove_beauty_cream_bar", "expo_marker_red", "haagen_dazs_butter_pecan",
    "hunts_paste", "krylon_short_cuts", "v8_fusion_peach_mango", "zilla_night_black_heat"]

camera_num = [1, 2, 3] # 相机
angles = [] # 角度
angle_step = 18 # 角度步进
save_cnt = 1 # 保存的文件数
root_path_datasets_raw = './datasets_raw/'

# 想要提取的角度
for i in range(360//angle_step):
    angles.append(angle_step*i)

print(angles, "angle numbers:", len(angles))


for obj_name in obj_name_list:
    multi_view_clouds_path = root_path_datasets_raw + obj_name + '/multi_view_clouds/'
    save_path_root = './datasets/' + obj_name + '/'

    # 创建保存文件夹
    try:
        os.makedirs(save_path_root)
    except OSError as e:
        if e.errno != 17:
            print('Some issue while creating the directory', save_path_root)

    # 复制多视角点云
    for camera in camera_num:
        for angle in angles:
            aim_obj = obj_name + 'NP' + str(camera) + '_' + str(angle) + '.pcd'
            aim_obj_path = multi_view_clouds_path + aim_obj
            print('\n[aim_obj_path]', aim_obj_path)
            # 复制文件
            save_path = save_path_root + obj_name + '_' + str(save_cnt) + '.pcd'
            print('[save_path]', save_path)
            shutil.copy(aim_obj_path, save_path) # 复制
            save_cnt += 1 # 文件数记录

    # 复制完整点云
    groundtruth_cloud_path = root_path_datasets_raw + obj_name + '/meshes/' + 'poisson.pcd'
    print('\n[groundtruth_cloud_path]', groundtruth_cloud_path)
    groundtruth_save_path = save_path_root + obj_name + '_gt.pcd'
    print('[groundtruth_save_path]', groundtruth_save_path)
    shutil.copy(groundtruth_cloud_path, groundtruth_save_path) # 复制

    save_cnt = 1 # 仅记录每个物体保存的文件数
