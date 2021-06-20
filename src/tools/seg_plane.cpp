#include <string>
#include <gpd/grasp_detector_pointnet.h>
#include <gpd/util/plot.h>

/**
 * 法线估计(调整方向)+平面分割
 */


namespace gpd {
namespace apps {
namespace detect_grasps {

    int DoMain(int argc, char *argv[]) {
        std::string pcd_filename;
        bool show = false;
        if (argc > 2) {
            pcd_filename = argv[1];
            if(strcmp(argv[2], "true") == 0) show = true;
        } else {
            pcd_filename = "/home/sdhm/点云数据/200924_00_cloud.pcd";
        }

        // Set the camera position. Assumes a single camera view.
        std::vector<double> camera_position = {0, 0, 0};

        Eigen::Matrix3Xd view_points(3, 1);
        view_points << camera_position[0], camera_position[1], camera_position[2];

        // Load point cloud from file.
        util::Cloud cloud(pcd_filename, view_points);
        if (cloud.getCloudOriginal()->size() == 0) {
            std::cout << "Error: Input point cloud is empty or does not exist!\n";
            return (-1);
        }

        util::Plot plotter(0, 0);
        if(show)
            plotter.plotCloud(cloud.getCloudOriginal(), "origion");

        // Workspace filtering
        std::vector<double> workspace = {0.4, 2.0, -0.1, 0.1, -2.0, 2.0};
        cloud.filterWorkspace(workspace);

        if(show)
            plotter.plotCloud(cloud.getCloudProcessed(), "filterWorkspace");

        // Voxelization
        if (0) {
            cloud.voxelizeCloud(0.003);

        //    util::Plot plotter(0, 0);
        //    plotter.plotCloud(cloud.getCloudProcessed(), "voxelizeCloud");
        }

        // Subsample the samples above plane
        printf("Sampling above plane ...\n");
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);
        std::vector<int> indices(0);
        pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        seg.setInputCloud(cloud.getCloudProcessed());
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() > 0) {
            pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
            extract.setInputCloud(cloud.getCloudProcessed());
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(indices);  // 获得桌面以上点云索引
            extract.filter(*cloud_filtered); // 获得桌面以上点云
        }

        if(show)
            plotter.plotCloud(cloud_filtered, "above_plane");

        // 移除nan点
        std::vector<int> indices2;
        pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, indices2);

        pcl::PCDWriter writer;
        writer.writeASCII("/home/sdhm/cloud_above_plane.pcd", *cloud_filtered); // 保存桌面以上的点云(无法线)

        return 0;
    }

}  // namespace detect_grasps
}  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
    return gpd::apps::detect_grasps::DoMain(argc, argv);
}
