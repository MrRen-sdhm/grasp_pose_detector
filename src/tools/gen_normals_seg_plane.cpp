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

        // Calculate surface normals using integral images if possible.
        if (cloud.getCloudOriginal()->isOrganized() && cloud.getNormals().cols() == 0)
        {
            std::cout << "[INFO Organize] Input cloud is organized." << "\n";
            cloud.calculateNormals(0);
            if (0) {
                cloud.setNormals(cloud.getNormals() * (-1.0));
            }
        }
        if(show)
            plotter.plotNormals(cloud.getCloudProcessed(), cloud.getNormals());

        // Workspace filtering
        std::vector<double> workspace = {-0.30, 0.30, -0.25, 0.25, -1.0, 1.0};
        cloud.filterWorkspace(workspace);

        if(show)
            plotter.plotCloud(cloud.getCloudProcessed(), "filterWorkspace");

        /// 保存带法线点云
        Eigen::Matrix3Xd normals = cloud.getNormals();
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ = cloud.getCloudProcessed();
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normal (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        for (int i = 0; i < normals.cols(); i++) {
            pcl::PointXYZRGBNormal p;
            p.x = cloud_->points[i].x;
            p.y = cloud_->points[i].y;
            p.z = cloud_->points[i].z;
            p.r = cloud_->points[i].r;
            p.g = cloud_->points[i].g;
            p.b = cloud_->points[i].b;
            p.normal_x = normals(0, i);
            p.normal_y = normals(1, i);
            p.normal_z = normals(2, i);
            cloud_with_normal->points.push_back(p);
        }

        pcl::PCDWriter writer;
        cloud_with_normal->width = 1;
        cloud_with_normal->height = cloud_with_normal->points.size();
        writer.writeASCII("/home/sdhm/cloud_with_normals.pcd", *cloud_with_normal); // 保存计算过法线的点云

        // Voxelization
        if (0) {
            cloud.voxelizeCloud(0.003);

        //    util::Plot plotter(0, 0);
        //    plotter.plotCloud(cloud.getCloudProcessed(), "voxelizeCloud");
        }

        // Normals calculating
        if(cloud.getNormals().cols() == 0)
        {
            cloud.calculateNormals(8);
            if (0) {
                cloud.setNormals(cloud.getNormals() * (-1.0));
            }
        }

        // Subsample the samples above plane
        if (1) {
            cloud.sampleAbovePlane();
        }

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

        writer.writeASCII("/home/sdhm/cloud_above_plane.pcd", *cloud_filtered); // 保存桌面以上的点云(无法线)

        return 0;
    }

}  // namespace detect_grasps
}  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
    return gpd::apps::detect_grasps::DoMain(argc, argv);
}
