## Grasp Pose Detector

- 开发环境

  Ubuntu16.04

- 依赖

  ```
  Opencv-3.4
  PCL-1.8.1
  Eigen-3.2
  Libtorch-1.0.0
  ```

- 编译安装

  ```
  cd grasp_pose_detector
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_DATA_GENERATION=ON 
  make -j*
  sudo make install
  ```

- 测试数据标注

  ```bash
  ./label_grasps ../cfg/label_params.cfg ../tutorials/red_bull_1.pcd ../tutorials/red_bull_gt.pcd
  ```

- 训练网络

  ```bash
  python3 train_net3_new.py /home/sdhm/Projects/gpd2/models/new/15channels/train.h5 /home/sdhm/Projects/gpd2/models/new/15channels/4objects/test.h5 15
  ```

- 测试网络

  ```
  python test_net.py /home/sdhm/Projects/gpd2/models/new/15channels/4objects/test.h5 15
  ```

- 测试eigen抓取姿态生成

  ```bash
  ./detect_grasps ../cfg/eigen_params.cfg ../tutorials/krylon.pcd
  ./detect_grasps ../cfg/eigen_params.cfg /home/sdhm/图片/kinect2点云样本/0004_cloud.pcd
  ```

- 测试在目标区域中采样（yolo+lenet）

  ```bash
  ./detect_grasps_yolo ../cfg/yolo_params_lenet.cfg lenet
  ```

- 测试在目标区域中采样（yolo+pointnet）

  ```bash
  ./detect_grasps_yolo ../cfg/yolo_params_pointnet.cfg pointnet
  ```

- 使用pointnet分类(Libtorch)

  ```
  ./detect_grasps_pointnet ../cfg/pointnet_params.cfg /home/sdhm/图片/kinect2点云样本/0004_cloud.pcd
  ```

- 使用pointnet分类(Python)

  ```
  ./detect_grasps_pointnet ../cfg/pointnet_python_params.cfg /home/sdhm/图片/kinect2点云样本/0004_cloud.pcd
  ```

## 数据集创建

1、获得ground truth：将下载的文件解压到bigbird_pcds/datasets_raw文件夹，在bigbird_pcds文件夹中运行python3 datasets_raw_proc.py， 将meshes文件夹中的ply文件转换为pcd文件，此pcd文件将作为ground truth点云，用于数据集自动生成。同时，创建multi_view_clouds文件夹，用于存储多视角点云（转换到物体坐标系下的点云）

2、创建多视角点云：在build文件夹运行 ./gpd_bigbird_process ../cfg/generate_data.cfg 1

3、提取多视角点云以及ground truth点云到datasets文件夹并重命名： python3 rename.py

4、创建数据集：在build文件夹运行 ./gpd_bigbird_process ../cfg/generate_data.cfg 0  生成objects.txt中包含物体的数据集



## 重要参数

```c++
hand_serch.cpp:控制手绕坐标轴的转动范围
// possible angles used for hand orientations
const Eigen::VectorXd angles_space = Eigen::VectorXd::LinSpaced(
    params_.num_orientations_ + 1, -1.0 * M_PI / 6.0, M_PI / 6.0);
```

```c++
finger_hand.cpp:控制手靠近物体的步长，可理解为距离物体的最小距离
// Attempt to deepen hand (move as far onto the object as possible without
// collision).
const double DEEPEN_STEP_SIZE = 0.01;
```

## 重要函数

```
hand_set.cpp
evalHands函数中使用transformToHandFrame将点云转换到手爪坐标系下。
```



## Libtorch使用

- Libtorch与Python Classifier切换：

  删除build文件夹，重新执行cmake

  ```sh
  cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_LIBTORCH=ON -DUSE_PYTHON=OFF
  make -j8
  ```

- 测试Libtorch抓取姿态生成

  ```sh
  ./detect_grasps ../cfg/libtorch_params.cfg /home/sdhm/图片/kinect2点云样本/0004_cloud.pcd
  ```



## 项目中遇到的工程问题

- C++环境中调用神经网络模型

  1、使用Libtorch

  2、C++调用Python

  ​	将数组或图片转换为ndarray

  ​	首先读取模型，后续仅使用模型
  
  3、C++回调函数中调用Python代码会因为获取不到GIL锁而产生死锁，在使用Python函数前需要显式地获得GIL锁。