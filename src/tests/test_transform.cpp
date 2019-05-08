#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>	//	pcl::transformPointCloud 用到这个头文件
#include <pcl/visualization/pcl_visualizer.h>

// 帮助函数
void
showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
}

// 主函数
int
main (int argc, char** argv)
{

    // 如果没有输入预期的参数程序将显示帮助
    if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
        showHelp (argv[0]);
        return 0;
    }

    // 从主函数参数查找点云数据文件 (.PCD|.PLY)
    std::vector<int> filenames;
    bool file_is_pcd = false;

    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");

    if (filenames.size () != 1)  {
        filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

        if (filenames.size () != 1) {
            showHelp (argv[0]);
            return -1;
        } else {
            file_is_pcd = true;
        }
    }

    // 加载点云数据文件
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    if (file_is_pcd) {
        if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
            showHelp (argv[0]);
            return -1;
        }
    } else {
        if (pcl::io::loadPLYFile (argv[filenames[0]], *source_cloud) < 0)  {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
            showHelp (argv[0]);
            return -1;
        }
    }

    /* 提示: 变换矩阵工作原理 :
             |-------> 变换矩阵列
      | 1 0 0 x |  \
      | 0 1 0 y |   }-> 左边是一个3阶的单位阵(无旋转)
      | 0 0 1 z |  /
      | 0 0 0 1 |    -> 这一行用不到 (这一行保持 0,0,0,1)
    */
    Eigen::Matrix4f transform_obj_from_NP5 = Eigen::Matrix4f::Identity(); // NP5坐标系到物体坐标系

    // NP5_0_pose.h5:
    // -0.1023643789864018	-0.9921044907772489	-0.07245835558680594	0.021073571241832845
    // -0.99472137166599	0.10156726804323772	0.01461105106297219	0.03693756424385908
    // -0.007136292150693041	0.07357152602636938	-0.9972644102205249	0.9714648832970901
    // 0.0	0.0	0.0	1.0
    transform_obj_from_NP5 (0,0) = -0.1023643789864018;
    transform_obj_from_NP5 (0,1) = -0.9921044907772489;
    transform_obj_from_NP5 (0,2) = -0.07245835558680594;
    transform_obj_from_NP5 (0,3) = 0.021073571241832845;
    transform_obj_from_NP5 (1,0) = -0.99472137166599;
    transform_obj_from_NP5 (1,1) = 0.10156726804323772;
    transform_obj_from_NP5 (1,2) = 0.01461105106297219;
    transform_obj_from_NP5 (1,3) = 0.03693756424385908;
    transform_obj_from_NP5 (2,0) = -0.007136292150693041;
    transform_obj_from_NP5 (2,1) = 0.07357152602636938;
    transform_obj_from_NP5 (2,2) = -0.9972644102205249;
    transform_obj_from_NP5 (2,3) = 0.9714648832970901;
    //    	(行, 列)

    Eigen::Matrix4f transform_NP1_from_NP5 = Eigen::Matrix4f::Identity(); // NP5坐标系到NP1坐标系的转换矩阵
    // H_NP1_from_NP5
    // 0.9984373102761732	0.05475578856187051	-0.01116875406875924	-0.033355375145305816
    // 0.014214434914719156	-0.05555295127846576	0.9983545559791409	-0.823273391935995
    // 0.05404523372635102	-0.9969531951015655	-0.056244461845091485	0.7241792643040872
    // 0.0	0.0	0.0	1.0
    transform_NP1_from_NP5 (0,0) = 0.9984373102761732;
    transform_NP1_from_NP5 (0,1) = 0.05475578856187051;
    transform_NP1_from_NP5 (0,2) = -0.01116875406875924;
    transform_NP1_from_NP5 (0,3) = -0.033355375145305816;
    transform_NP1_from_NP5 (1,0) = 0.014214434914719156;
    transform_NP1_from_NP5 (1,1) = -0.05555295127846576;
    transform_NP1_from_NP5 (1,2) = 0.9983545559791409;
    transform_NP1_from_NP5 (1,3) = -0.823273391935995;
    transform_NP1_from_NP5 (2,0) = 0.05404523372635102;
    transform_NP1_from_NP5 (2,1) = -0.9969531951015655;
    transform_NP1_from_NP5 (2,2) = -0.056244461845091485;
    transform_NP1_from_NP5 (2,3) = 0.7241792643040872;

    Eigen::Matrix4f transform_NP1_to_NP5 = transform_NP1_from_NP5.inverse(); // 求逆矩阵得到 NP1坐标系到NP5坐标系的转换矩阵

    // 物体从相机NP1坐标系转换到相机NP5坐标系，再从相机NP5坐标系转换到物体坐标系（NP1 -> NP5 -> obj） 注意：点云的变换一般都是左乘变换矩阵
    //                                 NP5 -> obj       <--      NP1 -> NP5
    Eigen::Matrix4f transform_NP1_to_obj = transform_obj_from_NP5 * transform_NP1_to_NP5;


    // 打印变换矩阵
    printf ("transform_obj_from_NP5:\n");
    std::cout << transform_obj_from_NP5 << std::endl;
    printf ("transform_NP1_from_NP5:\n");
    std::cout << transform_NP1_from_NP5 << std::endl;
    printf ("transform_NP1_to_NP5:\n");
    std::cout << transform_NP1_to_NP5 << std::endl;
    printf ("transform_NP1_to_obj:\n");
    std::cout << transform_NP1_to_obj << std::endl;

    // 执行变换，并将结果保存在新创建的‎‎ transformed_cloud ‎‎中
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    // # 方法1 直接进行两次变换
//    // 相机NP1坐标系转换到相机NP5坐标系
//    pcl::transformPointCloud (*source_cloud, *transformed_cloud, transform_NP1_from_NP5.inverse());
//    // 相机NP5坐标系转换到物体坐标系
//    pcl::transformPointCloud (*transformed_cloud, *transformed_cloud, transform_obj_from_NP5);

    // # 方法2 求出两次变换矩阵后进行一次变换
    pcl::transformPointCloud (*source_cloud, *transformed_cloud, transform_NP1_to_obj);
//    pcl::io::savePCDFileASCII("transformed.pcd", *transformed_cloud);

    // 可视化
    // 可视化将原始点云显示为白色，变换后的点云为红色，还设置了坐标轴、背景颜色、点显示大小
    printf(  "\nPoint cloud colors :  white  = original point cloud\n"
             "                        red  = transformed point cloud\n");
    pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");

    // 为点云定义 R,G,B 颜色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 255, 255, 255);
    // 输出点云到查看器，使用颜色管理
    viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20); // 红
    viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

    viewer.addCoordinateSystem (1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // 设置背景为深灰
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    //viewer.setPosition(800, 400); // 设置窗口位置

    while (!viewer.wasStopped ()) { // 在按下 "q" 键之前一直会显示窗口
        viewer.spinOnce ();
    }

    return 0;
}
