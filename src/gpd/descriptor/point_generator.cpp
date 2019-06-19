#include <gpd/descriptor/point_generator.h>
#include <opencv2/core/eigen.hpp>

namespace gpd {
    namespace descriptor {

        PointGenerator::PointGenerator(const candidate::HandGeometry &hand_geometry, int num_threads,
                                            int num_orientations,bool is_plotting, bool remove_plane)
                : hand_geometry_(hand_geometry),
                  num_threads_(num_threads),
                  num_orientations_(num_orientations),
                  remove_plane_(remove_plane) {
        }

        void PointGenerator::createPointGroups(
                const util::Cloud &cloud_cam,
                const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
                std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups_out,
                std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const{

            double t0 = omp_get_wtime();

            Eigen::Matrix3Xd points =
                    cloud_cam.getCloudProcessed()->getMatrixXfMap().cast<double>().block(
                            0, 0, 3, cloud_cam.getCloudProcessed()->points.size());
            util::PointList point_list(points, cloud_cam.getNormals(),
                                       cloud_cam.getCameraSource(),
                                       cloud_cam.getViewPoints());

            // Segment the support/table plane to speed up shadow computation.
            if (remove_plane_) {
                removePlane(cloud_cam, point_list);
            }

            // Prepare kd-tree for neighborhood searches in the point cloud.
            pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
            kdtree.setInputCloud(cloud_cam.getCloudProcessed());
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;

            // Set the radius for the neighborhood search to the largest image dimension.
            Eigen::Vector3d image_dims;
            image_dims << hand_geometry_.depth_, hand_geometry_.height_/2.0,
                    hand_geometry_.outer_diameter_;
            double radius = image_dims.maxCoeff();

            // 1. Find points within image dimensions.
            std::vector<util::PointList> nn_points_list;
            nn_points_list.resize(hand_set_list.size());

            double t_slice = omp_get_wtime();

#ifndef DEBUG
#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists) num_threads(num_threads_)
#endif
#endif
            for (int i = 0; i < hand_set_list.size(); i++) {
                pcl::PointXYZRGBA sample_pcl;
                sample_pcl.getVector3fMap() = hand_set_list[i]->getSample().cast<float>();

                if (kdtree.radiusSearch(sample_pcl, radius, nn_indices, nn_dists) > 0) {
                    nn_points_list[i] = point_list.slice(nn_indices);
                }
            }
            printf("neighborhoods search time: %3.4f\n", omp_get_wtime() - t_slice);

            createPointList(hand_set_list, nn_points_list, point_groups_out, hands_out);
            printf("Created %zu Point groups in %3.4fs\n", point_groups_out.size(),
                   omp_get_wtime() - t0);
        }

        void PointGenerator::createPointList(
                const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
                const std::vector<util::PointList> &nn_points_list,
                std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups_out,
                std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const {
            double t0_images = omp_get_wtime();

            int m = hand_set_list[0]->getHands().size();
            int n = hand_set_list.size() * m;
            std::vector<std::vector<std::unique_ptr<Eigen::Matrix3Xd>>> point_groups_list(n);

#ifndef DEBUG
#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
#endif
            for (int i = 0; i < hand_set_list.size(); i++) {
                point_groups_list[i] = createPointGroups(*hand_set_list[i], nn_points_list[i]);
            }

            for (int i = 0; i < hand_set_list.size(); i++) {
                for (int j = 0; j < hand_set_list[i]->getHands().size(); j++) {
                    if (hand_set_list[i]->getIsValid()(j)) {
                        point_groups_out.push_back(std::move(point_groups_list[i][j]));
                        hands_out.push_back(std::move(hand_set_list[i]->getHands()[j]));
                    }
                }
            }
        }

        std::vector<std::unique_ptr<Eigen::Matrix3Xd>> PointGenerator::createPointGroups(
                const candidate::HandSet &hand_set, const util::PointList &nn_points) const {

            const std::vector<std::unique_ptr<candidate::Hand>> &hands = hand_set.getHands();
            std::vector<std::unique_ptr<Eigen::Matrix3Xd>> point_groups_list(hands.size());

            printf("\n[INFO] hands size:%zu\n", hands.size());
            for (int i = 0; i < hands.size(); i++) {
                if (hand_set.getIsValid()(i)) {
                    point_groups_list[i] = std::make_unique<Eigen::Matrix3Xd>(Eigen::Matrix3Xd::Zero(3, 1000));
                    createPointGroup(nn_points, *hands[i], *point_groups_list[i]);
                    printf("[INFO] point_groups_list[%d]\n", i);
                }
            }
            return point_groups_list;
        }

        void PointGenerator::createPointGroup(const util::PointList &point_list,
                                         const candidate::Hand &hand, Eigen::Matrix3Xd &point_groups) const {
            // 1. Transform points in neighborhood into the unit image.
            Eigen::Matrix3Xd point_groups_eigen = transformToUnitImage(point_list, hand);
            cout << "point_groups_eigen cols: " << point_groups_eigen.cols() << endl;
            if(point_groups_eigen.cols() > 50) point_groups = point_groups_eigen;
        }

        Eigen::Matrix3Xd PointGenerator::transformToUnitImage(
                const util::PointList &point_list, const candidate::Hand &hand) const {
            // 1. Transform points and normals in neighborhood into the hand frame.
            const Eigen::Matrix3Xd rotation_p2h = hand.getFrame().transpose(); // 原始点云坐标系到手抓坐标系旋转矩阵
            const Eigen::Vector3d &position = hand.getPosition();

            // 将原始点云坐标系下的点转换到手抓坐标系下
            Eigen::Matrix3Xd points(rotation_p2h *(point_list.getPoints() - position.replicate(1, point_list.size())));

            // 2. Find points in hand closing area.
            Eigen::Matrix3Xd points_close_p2h = findPointsInHand(hand, points);

            return points_close_p2h;
        }

        Eigen::Matrix3Xd PointGenerator::findPointsInHand(
                const candidate::Hand &hand, const Eigen::Matrix3Xd &points) const {
            std::vector<int> indices;
            const double half_outer_diameter = hand_geometry_.outer_diameter_ / 2.0;

            const Eigen::Matrix3Xd rotation_h2p = hand.getFrame(); // 手抓坐标系到原始点云坐标系旋转矩阵
            const Eigen::Matrix3Xd rotation_p2h = hand.getFrame().transpose(); // 原始点云坐标系到手抓坐标系旋转矩阵
            const Eigen::Vector3d &position = hand.getPosition();

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_closing ( new pcl::PointCloud<pcl::PointXYZ> ); // 手抓闭合区域点云

            // 获取手抓闭合区域点云索引
            for (int i = 0; i < points.cols(); i++) {
                pcl::PointXYZ p;
                    // approach
                if ((points(0, i) > hand.getBottom()) &&
                    (points(0, i) < hand.getBottom() + hand_geometry_.depth_) &&
                    // binormal
                    (points(1, i) > hand.getCenter() - half_outer_diameter) &&
                    (points(1, i) < hand.getCenter() + half_outer_diameter) &&
                    // axis
                    (points(2, i) > -0.5 * hand_geometry_.height_) &&
                    (points(2, i) < 0.5 * hand_geometry_.height_)) {
                    indices.push_back(i);

                    p.x = points(0, i);
                    p.y = points(1, i);
                    p.z = points(2, i);
                    cloud_closing->points.push_back(p);

//                    cout << "[debug] points_out:" << " x:" << points(0, i) << " y:" <<
//                                            points(1, i) << " z:" <<  points(2, i) << endl;
                }
            }

            // 获取手抓闭合区域点云
            Eigen::Matrix3Xd points_close_p2h = util::EigenUtils::sliceMatrix(points, indices);
            if (points_close_p2h.cols() < 50) return points_close_p2h;
#if 1
            // 将手抓坐标系内的点转换回点云坐标系
            Eigen::Matrix3Xd points_h2p = rotation_h2p * points + position.replicate(1, points.size());
            pcl::PointCloud<pcl::PointXYZ>::Ptr points_h2p_cloud ( new pcl::PointCloud<pcl::PointXYZ> );

            for (int i = 0; i < indices.size(); i++) {
                pcl::PointXYZ p;
                p.x = points_h2p(0, indices[i]);
                p.y = points_h2p(1, indices[i]);
                p.z = points_h2p(2, indices[i]);
//                cout << "[debug] points_h2p:" << " x:" << points_h2p(0, i) << " y:" <<
//                                              points_h2p(1, i) << " z:" <<  points_h2p(2, i) << endl;
                points_h2p_cloud->points.push_back(p);
            }

            // 可视化显示
            pcl::visualization::PCLVisualizer viewer ("points in closing area");
            viewer.setSize(640, 480);
            viewer.setBackgroundColor(1.0, 1.0, 1.0);
            viewer.addCoordinateSystem(0.1, "Coordinate");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> points_color_handler(
                    cloud_closing, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> points_h2p_cloud_color_handler(
                    points_h2p_cloud, 0, 0, 255);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb (cloud_.getCloudOriginal());

            // 手抓坐标系下闭合区域点云
            viewer.addPointCloud<pcl::PointXYZ>(cloud_closing, points_color_handler, "points_p2h");
            // 原始点云坐标系下闭合区域点云
            viewer.addPointCloud<pcl::PointXYZ>(points_h2p_cloud, points_h2p_cloud_color_handler, "points_h2p");
            // 原始点云
            viewer.addPointCloud<pcl::PointXYZRGBA>(cloud_.getCloudOriginal(), rgb, "cloud");

            // 原始点云坐标系下手抓
            Eigen::Vector3d color_hand(0.0, 0.7, 0.0);
            plotHand3D(viewer, hand, hand_geometry_, 1, color_hand);

            // 世界坐标系下手抓
            Eigen::Matrix3d hand_world_frame = rotation_p2h * hand.getFrame();
            Eigen::Vector3d hand_world_position = rotation_p2h * (hand.getPosition() -
                                                            position.replicate(1, hand.getSample().size()));
            Eigen::Vector3d color_hand_world(0.0, 0.7, 0.0);
            plotHand3D(viewer, hand_world_position, hand_world_frame, hand_geometry_.outer_diameter_,
                    hand_geometry_.finger_width_, hand_geometry_.depth_, hand_geometry_.height_, 2, color_hand_world);

            cout << "[debug] hand_world_frame:\n" << hand_world_frame << endl;
            cout << "[debug] hand_world_position:\n" << hand_world_position << endl;

            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "points_p2h");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "points_h2p");

            while (!viewer.wasStopped ()) {
                viewer.spinOnce ();
            }
#endif
            return points_close_p2h;
        }

        void PointGenerator::plotHand3D(pcl::visualization::PCLVisualizer &viewer, const candidate::Hand &hand,
                              const candidate::HandGeometry &geometry, int idx,
                              const Eigen::Vector3d &rgb) const {
            plotHand3D(viewer, hand, geometry.outer_diameter_, geometry.finger_width_,
                       geometry.depth_, geometry.height_, idx, rgb);
        }

        void PointGenerator::plotHand3D(pcl::visualization::PCLVisualizer &viewer, const candidate::Hand &hand,
                              double outer_diameter, double finger_width,
                              double hand_depth, double hand_height, int idx,
                              const Eigen::Vector3d &rgb) const {
            const double hw = 0.5 * outer_diameter;
            const double base_depth = 0.02;
            const double approach_depth = 0.07;

            Eigen::Vector3d left_bottom =
                    hand.getPosition() - (hw - 0.5 * finger_width) * hand.getBinormal();
            Eigen::Vector3d right_bottom =
                    hand.getPosition() + (hw - 0.5 * finger_width) * hand.getBinormal();
            Eigen::VectorXd left_center =
                    left_bottom + 0.5 * hand_depth * hand.getApproach();
            Eigen::VectorXd right_center =
                    right_bottom + 0.5 * hand_depth * hand.getApproach();
            Eigen::Vector3d base_center = left_bottom +
                                          0.5 * (right_bottom - left_bottom) -
                                          0.01 * hand.getApproach();
            Eigen::Vector3d approach_center = base_center - 0.04 * hand.getApproach();

            cout << "[debug] hand.getFrame:\n" << hand.getFrame() << "\n[debug] hand.getPosition:\n" << hand.getPosition() << endl;

            const Eigen::Quaterniond quat(hand.getFrame());
            const std::string num = std::to_string(idx);

            plotCube(viewer, left_center, quat, hand_depth, finger_width, hand_height,
                     "left_finger_" + num, rgb);
            plotCube(viewer, right_center, quat, hand_depth, finger_width, hand_height,
                     "right_finger_" + num, rgb);
            plotCube(viewer, base_center, quat, base_depth, outer_diameter, hand_height,
                     "base_" + num, rgb);
//          plotCube(viewer, approach_center, quat, approach_depth, finger_width,
//                   0.5 * hand_height, "approach_" + num, rgb);
        }

        void PointGenerator::plotHand3D(pcl::visualization::PCLVisualizer &viewer, Eigen::Vector3d position,
                                        Eigen::Matrix3d frame, double outer_diameter, double finger_width,
                                        double hand_depth, double hand_height, int idx,
                                        const Eigen::Vector3d &rgb) const {
            const double hw = 0.5 * outer_diameter;
            const double base_depth = 0.02;
            const double approach_depth = 0.07;
            Eigen::Vector3d approach = frame.col(0);
            Eigen::Vector3d binormal = frame.col(1);

            Eigen::Vector3d left_bottom =
                    position - (hw - 0.5 * finger_width) * binormal;
            Eigen::Vector3d right_bottom =
                    position + (hw - 0.5 * finger_width) * binormal;
            Eigen::VectorXd left_center =
                    left_bottom + 0.5 * hand_depth * approach;
            Eigen::VectorXd right_center =
                    right_bottom + 0.5 * hand_depth * approach;
            Eigen::Vector3d base_center = left_bottom +
                                          0.5 * (right_bottom - left_bottom) -
                                          0.01 * approach;
            Eigen::Vector3d approach_center = base_center - 0.04 * approach;

            const Eigen::Quaterniond quat(frame);
            const std::string num = std::to_string(idx);

            plotCube(viewer, left_center, quat, hand_depth, finger_width, hand_height,
                     "left_finger_" + num, rgb);
            plotCube(viewer, right_center, quat, hand_depth, finger_width, hand_height,
                     "right_finger_" + num, rgb);
            plotCube(viewer, base_center, quat, base_depth, outer_diameter, hand_height,
                     "base_" + num, rgb);
//          plotCube(viewer, approach_center, quat, approach_depth, finger_width,
//                   0.5 * hand_height, "approach_" + num, rgb);
        }

        void PointGenerator::plotCube(pcl::visualization::PCLVisualizer &viewer, const Eigen::Vector3d &position,
                            const Eigen::Quaterniond &rotation, double width,
                            double height, double depth, const std::string &name,
                            const Eigen::Vector3d &rgb) const {
            viewer.addCube(position.cast<float>(), rotation.cast<float>(), width, height,
                            depth, name);
            viewer.setShapeRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, name);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                rgb(0), rgb(1), rgb(2), name);
            viewer.setShapeRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_OPACITY, 0.25, name);
        }

        void PointGenerator::removePlane(const util::Cloud &cloud_cam,
                                         util::PointList &point_list) const {
            pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            seg.setInputCloud(cloud_cam.getCloudProcessed());
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.size() > 0) {
                pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
                extract.setInputCloud(cloud_cam.getCloudProcessed());
                extract.setIndices(inliers);
                extract.setNegative(true);
                std::vector<int> indices;
                extract.filter(indices);
                if (indices.size() > 0) {
                    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
                    extract.filter(*cloud);
                    point_list = point_list.slice(indices);
                    printf("Removed plane from point cloud. %zu points remaining.\n",
                           cloud->size());
                } else {
                    printf("Plane fit failed. Using entire point cloud ...\n");
                }
            }
        }

    }  // namespace descriptor
}  // namespace gpd
