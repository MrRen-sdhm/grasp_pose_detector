#!/usr/bin/env python3
# coding:utf8
# 功能： 处理原始数据集, 将meshes文件夹中的ply文件转换为pcd文件, 作为ground truth点云, 并创建多视角点云保存文件夹

import numpy as np
import random
import shutil
import os

obj_name_list = ["red_bull", "advil_liqui_gels", "band_aid_clear_strips", "blue_clover_baby_toy",
    "bumblebee_albacore", "campbells_soup_at_hand_creamy_tomato", "colgate_cool_mint", "crayola_24_crayons",
    "crest_complete_minty_fresh", "dove_beauty_cream_bar", "expo_marker_red", "haagen_dazs_butter_pecan",
    "hunts_paste", "krylon_short_cuts", "v8_fusion_peach_mango", "zilla_night_black_heat"]

for obj_name in obj_name_list:
    meshes_path = './datasets_raw/' + obj_name + '/meshes/'

    ply_name_poisson = meshes_path + 'poisson.ply'
    pcd_name_poisson = meshes_path + 'poisson.pcd'

    ply_name_tsdf = meshes_path + 'tsdf.ply'
    pcd_name_tsdf = meshes_path + 'tsdf.pcd'

    # 将meshes文件夹中的ply文件转换为pcd文件
    ply2pcd_poisson_cmd = 'pcl_ply2pcd ' + ply_name_poisson + ' ' + pcd_name_poisson
    ply2pcd_tsdf_cmd = 'pcl_ply2pcd ' + ply_name_tsdf + ' ' + pcd_name_tsdf
    print(ply2pcd_poisson_cmd)
    print(ply2pcd_tsdf_cmd)
    os.system(ply2pcd_poisson_cmd)
    os.system(ply2pcd_tsdf_cmd)

    # 创建多视角点云保存文件夹
    multi_view_clouds_path = './datasets_raw/' + obj_name + '/multi_view_clouds/'
    try:
        os.makedirs(multi_view_clouds_path)
    except OSError as e:
        if e.errno != 17:
            print('Some issue while creating the directory', multi_view_clouds_path)
