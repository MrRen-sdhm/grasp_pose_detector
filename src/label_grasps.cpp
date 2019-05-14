#include <string>
#include <stdio.h>

#include <gpd/grasp_detector.h>
#include <gpd/descriptor/image_15_channels_strategy.h>

namespace gpd {
    namespace apps {
        namespace detect_grasps {

            bool checkFileExists(const std::string &file_name) {
                std::ifstream file;
                file.open(file_name.c_str());
                if (!file) {
                    std::cout << "File " + file_name + " could not be found!\n";
                    return false;
                }
                file.close();
                return true;
            }

            void showImage(const cv::Mat &image) {
                int border = 5;
                int n = 3;
                int image_size = 60;
                int total_size = n * (image_size + border) + border;

                cv::Mat image_out(total_size, total_size, CV_8UC3, cv::Scalar(0.5));
                std::vector<cv::Mat> channels;
                cv::split(image, channels);

                for (int i = 0; i < n; i++) {
                    // OpenCV requires images to be in BGR or grayscale to be displayed.
                    cv::Mat normals_rgb, depth_rgb, shadow_rgb;
                    std::vector<cv::Mat> normals_channels(3);
                    for (int j = 0; j < normals_channels.size(); j++) {
                        normals_channels[j] = channels[i * 5 + j];
                    }
                    cv::merge(normals_channels, normals_rgb);
                    // OpenCV requires images to be in BGR or grayscale to be displayed.
                    cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
                    cvtColor(channels[i * 5 + 3], depth_rgb, cv::COLOR_GRAY2RGB);
                    cvtColor(channels[i * 5 + 4], shadow_rgb, cv::COLOR_GRAY2RGB);
                    normals_rgb.copyTo(image_out(cv::Rect(
                            border, border + i * (border + image_size), image_size, image_size)));
                    depth_rgb.copyTo(image_out(cv::Rect(2 * border + image_size,
                                                        border + i * (border + image_size),
                                                        image_size, image_size)));
                    shadow_rgb.copyTo(image_out(cv::Rect(3 * border + 2 * image_size,
                                                         border + i * (border + image_size),
                                                         image_size, image_size)));
                }

                cv::namedWindow("Grasp Image (15 channels)", cv::WINDOW_NORMAL);
                cv::imshow("Grasp Image (15 channels)", image_out);
                cv::waitKey(0);
                cv::destroyWindow("Grasp Image (15 channels)");
            }


            int DoMain(int argc, char *argv[]) {
                // Read arguments from command line.
                if (argc < 4) {
                    std::cout << "Error: Not enough input arguments!\n\n";
                    std::cout << "Usage: label_grasps CONFIG_FILE PCD_FILE MESH_FILE\n\n";
                    std::cout << "Find grasp poses for a point cloud, PCD_FILE (*.pcd), "
                                 "using parameters from CONFIG_FILE (*.cfg), and check them "
                                 "against a mesh, MESH_FILE (*.pcd).\n\n";
                    return (-1);
                }

                std::string config_filename = argv[1];
                std::string pcd_filename = argv[2];
                std::string mesh_filename = argv[3];
                if (!checkFileExists(config_filename)) {
                    printf("Error: CONFIG_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(pcd_filename)) {
                    printf("Error: PCD_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(mesh_filename)) {
                    printf("Error: MESH_FILE not found!\n");
                    return (-1);
                }

                // Read parameters from configuration file.
                const double VOXEL_SIZE = 0.003;
                util::ConfigFile config_file(config_filename);
                config_file.ExtractKeys();
                std::vector<double> workspace =
                        config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
                int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
                int num_samples = config_file.getValueOfKey<int>("num_samples", 30);
                bool sample_above_plane =
                        config_file.getValueOfKey<int>("sample_above_plane", 1);
                printf("num_threads: %d, num_samples: %d\n", num_threads, num_samples);

                // View point from which the camera sees the point cloud.
                Eigen::Matrix3Xd view_points(3, 1);
                view_points.setZero();

                // Load point cloud from file.
                util::Cloud cloud(pcd_filename, view_points);
                if (cloud.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Input point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Load point cloud from file.
                util::Cloud mesh(mesh_filename, view_points);
                if (mesh.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Mesh point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Prepare the point cloud.
                cloud.filterWorkspace(workspace);
                cloud.voxelizeCloud(VOXEL_SIZE);
                cloud.calculateNormals(num_threads);
                cloud.setNormals(cloud.getNormals() * (-1.0));  // NOTE: do not do this! 翻转单视角点云表面法线（坐标系在物体内部时使用）
                if (sample_above_plane) {
                    cloud.sampleAbovePlane();
                }
                cloud.subsample(num_samples);

                // Prepare the mesh.
                mesh.calculateNormals(num_threads);
                mesh.setNormals(mesh.getNormals() * (-1.0)); // NOTE: 翻转 ground truth 的表面法线

                // Detect grasp poses.
                std::vector<std::unique_ptr<candidate::Hand>> hands;
                std::vector<std::unique_ptr<cv::Mat>> images;
                GraspDetector detector(config_filename);
                detector.createGraspImages(cloud, hands, images);

                printf("hands: %zu\n", hands.size());
                printf("images: %zu\n", images.size());

                std::vector<int> labels = detector.evalGroundTruth(mesh, hands);
                printf("labels: %zu\n", labels.size());

                const candidate::HandSearch::Parameters &params =
                        detector.getHandSearchParameters();
                util::Plot plot(params.hand_axes_.size(), params.num_orientations_);

                // 显示单视角点云表面法线
                plot.plotNormals(cloud.getCloudOriginal(), cloud.getNormals());
                // 显示ground truth点云表面法线
                plot.plotNormals(mesh.getCloudOriginal(), mesh.getNormals());

                // 显示所有抓取姿态
//                plot.plotAntipodalHands(hands, cloud.getCloudProcessed(), "Labeled Hands",
//                                        params.hand_geometry_);

                // 单独显示各个抓取姿态
//                for(int i = 0; i < hands.size(); i++) {
//                    if (hands[i]->isFullAntipodal()) {
//                        plot.plotAntipodalHand(*hands[i], cloud.getCloudProcessed(), "Hand" + std::to_string(i),
//                                               params.hand_geometry_);
//                    }
//                }

                // 单独显示各个抓取姿态以及点云
                for(int i = 0; i < hands.size(); i++) {
//                    if (! hands[i]->isFullAntipodal()) {
                    if (1) {
                        std::cout << "sample: " << hands[i]->getSample().transpose() << std::endl;
//                        std::cout << "grasp orientation:\n" << hands[i]->getFrame() << std::endl;
//                        std::cout << "grasp position: " << hands[i]->getPosition().transpose() << std::endl << std::endl;
                        printf("lable%d: %d\n", i, labels[i]);
                        plot.plotValidHand(*hands[i], cloud.getCloudProcessed(),
                                           mesh.getCloudProcessed(), "Antipodal Hand" + std::to_string(i),
                                           params.hand_geometry_, true);
                        showImage(*images[i]); // 显示多通道图像
                    }
                }



                // 分开显示有效和无效抓取姿态
//                std::vector<std::unique_ptr<candidate::Hand>> valid_hands;
//                for (size_t i = 0; i < hands.size(); i++) {
//                //    printf("(%zu) label: %d\n", i, labels[i]);
//                    if (hands[i]->isFullAntipodal()) { // 有效抓取姿态
//                        std::string old_path = "/home/sdhm/图片/images/GraspImage" + std::to_string(i) + ".jpg";
//                        std::string new_path = "/home/sdhm/图片/images/1/GraspImage" + std::to_string(i) + ".jpg";
//                        printf("(%zu) %d ", i, labels[i]);
//                        printf("move from %s to %s\n", old_path.c_str(), new_path.c_str());
//
//                        rename(old_path.c_str(), new_path.c_str()); // 移动文件
//
//                        valid_hands.push_back(std::move(hands[i]));
//                    }
//                    else { // 无效抓取姿态
//                        std::string old_path = "/home/sdhm/图片/images/GraspImage" + std::to_string(i) + ".jpg";
//                        std::string new_path = "/home/sdhm/图片/images/0/GraspImage" + std::to_string(i) + ".jpg";
//                        printf("(%zu) %d ", i, labels[i]);
//                        printf("move from %s to %s\n", old_path.c_str(), new_path.c_str());
//
//                        rename(old_path.c_str(), new_path.c_str()); // 移动文件
//                    }
//                }
//                plot.plotValidHands(valid_hands, cloud.getCloudProcessed(),
//                                    mesh.getCloudProcessed(), "Antipodal Hands",
//                                    params.hand_geometry_);

                return 0;
            }

        }  // namespace detect_grasps
    }  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
    return gpd::apps::detect_grasps::DoMain(argc, argv);
}
