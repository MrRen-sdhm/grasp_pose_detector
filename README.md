## Grasp Pose Detector

- 测试数据标注

  ```bash
  ./label_grasps ../cfg/label_params.cfg ../tutorials/red_bull_1.pcd ../tutorials/red_bull_gt.pcd
  ```

- 训练网络

  ```bash
  python3 train_net3_new.py /home/sdhm/Projects/gpd2/models/new/15channels/train.h5 /home/sdhm/Projects/gpd2/models/new/15channels/4objects/test.h5 15
  ```

- 测试eigen抓取姿态生成

  ```bash
  ./detect_grasps ../cfg/eigen_params.cfg ../tutorials/krylon.pcd
  ./detect_grasps ../cfg/eigen_params.cfg /home/sdhm/图片/kinect2点云样本/0004_cloud.pcd
  ```

- 测试在目标区域中采样（yolo）

  ```bash
  ./detect_grasps_yolo ../cfg/yolo_params.cfg
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



## Libtorch使用

- eigen与libtorch切换：

  修改classifier.cpp

```C++
//#include <gpd/net/eigen_classifier.h>
#include <gpd/net/libtorch_classifier.h>

//  return std::make_shared<EigenClassifier>(model_file, weights_file, device,
//                                           batch_size);
  return std::make_shared<LibtorchClassifier>(model_file, weights_file, device,
                                           batch_size);
```

​	修改CmakeLists.txt

```cmake
# eigen
#  add_library(${PROJECT_NAME}_conv_layer src/${PROJECT_NAME}/net/conv_layer.cpp)
#  add_library(${PROJECT_NAME}_dense_layer src/${PROJECT_NAME}/net/dense_layer.cpp)
#  set(classifier_src src/${PROJECT_NAME}/net/classifier.cpp src/${PROJECT_NAME}/net/eigen_classifier.cpp)
#  set(classifier_dep ${PROJECT_NAME}_conv_layer ${PROJECT_NAME}_dense_layer ${OpenCV_LIBRARIES})
# libtorch
set(classifier_src src/${PROJECT_NAME}/net/classifier.cpp src/${PROJECT_NAME}/net/libtorch_classifier.cpp)
set(classifier_dep ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
```



- 测试Libtorch抓取姿态生成

```bash
./detect_grasps ../cfg/libtorch_params.cfg /home/sdhm/图片/kinect2点云样本/0004_cloud.pcd
```

