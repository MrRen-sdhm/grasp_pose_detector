## Grasp Pose Detector

- 测试数据标注

./label_grasps ../cfg/label_params.cfg ../tutorials/red_bull_1.pcd ../tutorials/red_bull_gt.pcd



## 数据集创建

1、获得ground truth：将下载的文件解压到bigbird_pcds/datasets_raw文件夹，在bigbird_pcds文件夹中运行python3 datasets_raw_proc.py， 将meshes文件夹中的ply文件转换为pcd文件，此pcd文件将作为ground truth点云，用于数据集自动生成。同时，创建multi_view_clouds文件夹，用于存储多视角点云（转换到物体坐标系下的点云）

2、创建多视角点云：在build文件夹运行 ./bigbird_process ../cfg/generate_data.cfg 1

3、提取多视角点云以及ground truth点云到datasets文件夹并重命名： python3 rename.py

4、创建数据集：在build文件夹运行 ./bigbird_process ../cfg/generate_data.cfg 0  生成objects.txt中包含物体的数据集