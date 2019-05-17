#include <gpd/util/cloud.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>

namespace gpd {
namespace util {

    Cloud::Cloud()
            : cloud_original_(new PointCloudRGB), cloud_processed_(new PointCloudRGB) {
        view_points_.resize(3, 1);
        view_points_ << 0.0, 0.0, 0.0;
        sample_indices_.resize(0);
        samples_.resize(3, 0);
        normals_.resize(3, 0);
    }

    Cloud::Cloud(const PointCloudRGB::Ptr &cloud,
                 const Eigen::MatrixXi &camera_source,
                 const Eigen::Matrix3Xd &view_points)
            : cloud_processed_(new PointCloudRGB),
              cloud_original_(new PointCloudRGB),
              camera_source_(camera_source),
              view_points_(view_points) {
        sample_indices_.resize(0);
        samples_.resize(3, 0);
        normals_.resize(3, 0);

        pcl::copyPointCloud(*cloud, *cloud_original_);
        *cloud_processed_ = *cloud_original_;
    }

    Cloud::Cloud(const PointCloudPointNormal::Ptr &cloud,
                 const Eigen::MatrixXi &camera_source,
                 const Eigen::Matrix3Xd &view_points)
            : cloud_processed_(new PointCloudRGB),
              cloud_original_(new PointCloudRGB),
              camera_source_(camera_source),
              view_points_(view_points) {
        sample_indices_.resize(0);
        samples_.resize(3, 0);
        normals_.resize(3, 0);

        pcl::copyPointCloud(*cloud, *cloud_original_);
        *cloud_processed_ = *cloud_original_;
    }

    Cloud::Cloud(const PointCloudPointNormal::Ptr &cloud, int size_left_cloud,
                 const Eigen::Matrix3Xd &view_points)
            : cloud_processed_(new PointCloudRGB),
              cloud_original_(new PointCloudRGB),
              view_points_(view_points) {
        sample_indices_.resize(0);
        samples_.resize(3, 0);

        pcl::copyPointCloud(*cloud, *cloud_original_);
        *cloud_processed_ = *cloud_original_;

        // set the camera source matrix: (i,j) = 1 if point j is seen by camera i
        if (size_left_cloud == 0)  // one camera
        {
            camera_source_ = Eigen::MatrixXi::Ones(1, cloud->size());
        } else  // two cameras
        {
            int size_right_cloud = cloud->size() - size_left_cloud;
            camera_source_ = Eigen::MatrixXi::Zero(2, cloud->size());
            camera_source_.block(0, 0, 1, size_left_cloud) =
                    Eigen::MatrixXi::Ones(1, size_left_cloud);
            camera_source_.block(1, size_left_cloud, 1, size_right_cloud) =
                    Eigen::MatrixXi::Ones(1, size_right_cloud);
        }

        normals_.resize(3, cloud->size());
        for (int i = 0; i < cloud->size(); i++) {
            normals_.col(i) << cloud->points[i].normal_x, cloud->points[i].normal_y,
                    cloud->points[i].normal_z;
        }
    }

    Cloud::Cloud(const PointCloudRGB::Ptr &cloud, int size_left_cloud,
                 const Eigen::Matrix3Xd &view_points)
            : cloud_processed_(cloud),
              cloud_original_(cloud),
              view_points_(view_points) {
        sample_indices_.resize(0);
        samples_.resize(3, 0);
        normals_.resize(3, 0);

        // set the camera source matrix: (i,j) = 1 if point j is seen by camera i
        if (size_left_cloud == 0)  // one camera
        {
            camera_source_ = Eigen::MatrixXi::Ones(1, cloud->size());
        } else  // two cameras
        {
            int size_right_cloud = cloud->size() - size_left_cloud;
            camera_source_ = Eigen::MatrixXi::Zero(2, cloud->size());
            camera_source_.block(0, 0, 1, size_left_cloud) =
                    Eigen::MatrixXi::Ones(1, size_left_cloud);
            camera_source_.block(1, size_left_cloud, 1, size_right_cloud) =
                    Eigen::MatrixXi::Ones(1, size_right_cloud);
        }
    }

    Cloud::Cloud(const std::string &filename, const Eigen::Matrix3Xd &view_points)
            : cloud_processed_(new PointCloudRGB),
              cloud_original_(new PointCloudRGB),
              view_points_(view_points) {
        sample_indices_.resize(0);
        samples_.resize(3, 0);
        normals_.resize(3, 0);
        cloud_processed_ = loadPointCloudFromFile(filename);
        cloud_original_ = cloud_processed_;
        camera_source_ = Eigen::MatrixXi::Ones(1, cloud_processed_->size());
        std::cout << "Loaded point cloud with " << camera_source_.cols()
                  << " points \n";
    }

    Cloud::Cloud(const std::string &filename_left,
                 const std::string &filename_right,
                 const Eigen::Matrix3Xd &view_points)
            : cloud_processed_(new PointCloudRGB),
              cloud_original_(new PointCloudRGB),
              view_points_(view_points) {
        sample_indices_.resize(0);
        samples_.resize(3, 0);
        normals_.resize(3, 0);

        // load and combine the two point clouds
        std::cout << "Loading point clouds ...\n";
        PointCloudRGB::Ptr cloud_left(new PointCloudRGB),
                cloud_right(new PointCloudRGB);
        cloud_left = loadPointCloudFromFile(filename_left);
        cloud_right = loadPointCloudFromFile(filename_right);

        std::cout << "Concatenating point clouds ...\n";
        *cloud_processed_ = *cloud_left + *cloud_right;
        cloud_original_ = cloud_processed_;

        std::cout << "Loaded left point cloud with " << cloud_left->size()
                  << " points \n";
        std::cout << "Loaded right point cloud with " << cloud_right->size()
                  << " points \n";

        // set the camera source matrix: (i,j) = 1 if point j is seen by camera i
        camera_source_ = Eigen::MatrixXi::Zero(2, cloud_processed_->size());
        camera_source_.block(0, 0, 1, cloud_left->size()) =
                Eigen::MatrixXi::Ones(1, cloud_left->size());
        camera_source_.block(1, cloud_left->size(), 1, cloud_right->size()) =
                Eigen::MatrixXi::Ones(1, cloud_right->size());
    }

    void Cloud::removeNans() {
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_processed_, *cloud_processed_, indices);
        printf("Cloud after removing NANs: %zu\n", cloud_processed_->size());
    }

    void Cloud::removeStatisticalOutliers() {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
        sor.setInputCloud(cloud_processed_);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*cloud_processed_);
        printf("Cloud after removing statistical outliers: %zu\n",
               cloud_processed_->size());
    }

    void Cloud::filterWorkspace(const std::vector<double> &workspace) {
        // Filter indices into the point cloud.
        if (sample_indices_.size() > 0) {
            std::vector<int> indices_to_keep;

            for (int i = 0; i < sample_indices_.size(); i++) {
                const pcl::PointXYZRGBA &p = cloud_processed_->points[sample_indices_[i]];
                if (p.x > workspace[0] && p.x < workspace[1] && p.y > workspace[2] &&
                    p.y < workspace[3] && p.z > workspace[4] && p.z < workspace[5]) {
                    indices_to_keep.push_back(i);
                }
            }

            sample_indices_ = indices_to_keep;
            std::cout << sample_indices_.size()
                      << " sample indices left after workspace filtering \n";
        }

        // Filter (x,y,z)-samples.
        if (samples_.cols() > 0) {
            std::vector<int> indices_to_keep;

            for (int i = 0; i < samples_.cols(); i++) {
                if (samples_(0, i) > workspace[0] && samples_(0, i) < workspace[1] &&
                    samples_(1, i) > workspace[2] && samples_(1, i) < workspace[3] &&
                    samples_(2, i) > workspace[4] && samples_(2, i) < workspace[5]) {
                    indices_to_keep.push_back(i);
                }
            }

            samples_ = EigenUtils::sliceMatrix(samples_, indices_to_keep);
            std::cout << samples_.cols()
                      << " samples left after workspace filtering \n";
        }

        // Filter the point cloud.
        std::vector<int> indices;
        for (int i = 0; i < cloud_processed_->size(); i++) {
            const pcl::PointXYZRGBA &p = cloud_processed_->points[i];
            if (p.x > workspace[0] && p.x < workspace[1] && p.y > workspace[2] &&
                p.y < workspace[3] && p.z > workspace[4] && p.z < workspace[5]) {
                indices.push_back(i);
            }
        }

        Eigen::MatrixXi camera_source(camera_source_.rows(), indices.size());
        PointCloudRGB::Ptr cloud(new PointCloudRGB);
        cloud->points.resize(indices.size());
        for (int i = 0; i < indices.size(); i++) {
            camera_source.col(i) = camera_source_.col(indices[i]);
            cloud->points[i] = cloud_processed_->points[indices[i]];
        }
        if (normals_.cols() > 0) {
            Eigen::Matrix3Xd normals(3, indices.size());
            for (int i = 0; i < indices.size(); i++) {
                normals.col(i) = normals_.col(indices[i]);
            }
            normals_ = normals;
        }
        cloud_processed_ = cloud;
        camera_source_ = camera_source;


        if(cloud_processed_->isOrganized()) std::cout << "[INFO Organize] Cloud is organized after filterWorkspace." << "\n";
        else std::cout << "[INFO Organize] Cloud is not organized after filterWorkspace." << "\n";
    }

    void Cloud::filterObjectRegion(cv::Rect rect) {
        // Filter the point cloud.

        cout << "size:" << cloud_processed_->size() << endl;
        cout << "height:" << cloud_processed_->height << endl;
        cout << "width:" << cloud_processed_->width << endl;

        int col_start = rect.x;
        int col_end = rect.x + rect.width;

        cout << "col_start:" << col_start << endl;
        cout << "col_end:" << col_end << endl;

        int row_start = rect.y;
        int row_end = rect.y + rect.height;

        cout << "row_start:" << row_start << endl;
        cout << "row_end:" << row_end << endl;

        int center_indice = (rect.y + rect.height/2) * cloud_processed_->width + (rect.x + rect.width/2);
        const pcl::PointXYZRGBA &center_p = cloud_processed_->points[center_indice];
        float depth = center_p.z;
        printf("depth:%f\n", depth);
        printf("rect_width:%d rect_height:%d width*height:%d\n", rect.width, rect.height, rect.width*rect.height);

        Eigen::Matrix3Xd samples(3, rect.width * rect.height);
        std::vector<int> indices;
        int sample_indices_num = 0;
        std::vector<int> sample_indices_to_keep;
        for (int row = rect.y; row < rect.y + rect.height; ++row) { // 540
            for (int col = rect.x; col < rect.x + rect.width; ++col) { // 960
                int indices_num = row * cloud_processed_->width + col;
                const pcl::PointXYZRGBA &p = cloud_processed_->points[indices_num];
                if (std::abs(p.z-depth) < 0.08 && -1 < p.z && p.z < 1) { // 限制深度

                    if (-1 < p.x && p.x < 1 && -1 < p.y && p.y < 1) {
                        printf("sample_indices_num:%d x:%f y:%f z:%f\n", sample_indices_num, p.x, p.y, p.z);
                        samples(0, sample_indices_num) = p.x;
                        samples(1, sample_indices_num) = p.y;
                        samples(2, sample_indices_num) = p.z;

                        sample_indices_num ++;
                        sample_indices_to_keep.push_back(sample_indices_num);
                    }

                    indices.push_back(indices_num);
                }
            }
        }

        printf("Save samples\n");
        samples_ = EigenUtils::sliceMatrix(samples, sample_indices_to_keep);

        PointCloudRGB::Ptr cloud(new PointCloudRGB);
        cloud->points.resize(indices.size());
        for (int i = 0; i < indices.size(); i++) {
            cloud->points[i] = cloud_processed_->points[indices[i]];
        }
        if (normals_.cols() > 0) {
            Eigen::Matrix3Xd normals(3, indices.size());
            for (int i = 0; i < indices.size(); i++) {
                normals.col(i) = normals_.col(indices[i]);
            }
            normals_ = normals;
        }
        cloud_processed_ = cloud;

        cout << "size_after:" << cloud_processed_->size() << endl;
        cout << "height_after:" << cloud_processed_->height << endl;
        cout << "width_after:" << cloud_processed_->width << endl;
    }

    void Cloud::getSamplesRegion(cv::Rect rect) {
        // Filter the point cloud.

        cout << "size:" << cloud_processed_->size() << endl;
        cout << "height:" << cloud_processed_->height << endl;
        cout << "width:" << cloud_processed_->width << endl;

        // TODO: 处理多个物体区域
        int col_start = rect.x;
        int col_end = rect.x + rect.width;
        int col_center = rect.x + rect.width/2;

        cout << "col_start:" << col_start << endl;
        cout << "col_end:" << col_end << endl;
        cout << "col_center:" << col_center << endl;

        int row_start = rect.y;
        int row_end = rect.y + rect.height;
        int row_center = rect.y + rect.height/2;

        cout << "row_start:" << row_start << endl;
        cout << "row_end:" << row_end << endl;
        cout << "row_center:" << row_center << endl;

        printf("rect_width:%d rect_height:%d width*height:%d\n", rect.width, rect.height, rect.width*rect.height);

        // 获取中心深度值（避免中心深度不存在-Nan）
        std::vector<float> depth;
        std::vector<int> cloud_rect_indices;
        for (int row = row_center - rect.height/6; row < row_center + rect.height/6; ++row) { // 540
            for (int col = col_center - rect.width/6; col < col_center + rect.width/6; ++col) { // 960
                int center_indices = row * cloud_original_->width + col; // 二维索引
                float depthValue = cloud_original_->points[center_indices].z;
                cloud_rect_indices.push_back(center_indices);
                printf("depth[%d %d] (indice: %d): %f\n", row, col, center_indices, depthValue);
                if (depthValue > 0.01 && depthValue < 2.0){ // 确保有深度信息
                    depth.push_back(depthValue);
                }
            }
        }
        auto min_depth = std::min_element(depth.begin(), depth.end());
        cout << "min_depth:" << *min_depth << endl;

        // 滤除偏离较大的点, 矩形区域内点在同一曲面上, 深度差距不会很大
        int validPointNum = 0; // 有效点数
        float centerAvgDepth = 0; // 中心平均深度值
        for (size_t i = 0; i < depth.size(); i++) {
            if (depth[i]-*min_depth < 0.05) { // 距离阈值, 超过阈值认为是无效点
                centerAvgDepth += depth[i];
                validPointNum++;
                printf("valid_depth(%zu): %f\n", i, depth[i]);
            }
        }
        centerAvgDepth /= validPointNum;
        printf("validPointNum:%d centerAvgDepth:%f\n", validPointNum, centerAvgDepth);
        if (validPointNum < 1) printf("\033[0;31m%s\033[0m\n", "[Error] Can't get obj region center depth.");

        Eigen::Matrix3Xd samples(3, rect.width * rect.height);
        int sample_indices_num = 0;
        std::vector<int> sample_indices, cloud_obj_indices;
        // get samples from objects' region
        for (int row = rect.y; row < rect.y + rect.height; ++row) { // 540
            for (int col = rect.x; col < rect.x + rect.width; ++col) { // 960
                int indices_num = row * cloud_original_->width + col; // 二维索引
                const pcl::PointXYZRGBA &p = cloud_original_->points[indices_num];
                if (0.01 < p.z && p.z < 2.0 && std::abs(p.z - centerAvgDepth) < 0.08) { // 限制深度
                    if (-1 < p.x && p.x < 1 && -1 < p.y && p.y < 1) {
                        samples(0, sample_indices_num) = p.x;
                        samples(1, sample_indices_num) = p.y;
                        samples(2, sample_indices_num) = p.z;

                        sample_indices_num ++;
                        sample_indices.push_back(sample_indices_num);
//                        printf("sample_indices_num:%d x:%f y:%f z:%f\n", sample_indices_num, p.x, p.y, p.z);
                    }
                    cloud_obj_indices.push_back(indices_num);
                }
            }
        }

        printf("Get %zu samples from object region.\n", sample_indices.size());
        samples_ = EigenUtils::sliceMatrix(samples, sample_indices);

        // 获取物体矩形框中的点云
        PointCloudRGB::Ptr cloud(new PointCloudRGB);
        cloud->points.resize(cloud_obj_indices.size());
        for (int i = 0; i < cloud_obj_indices.size(); i++) {
            cloud->points[i] = cloud_original_->points[cloud_obj_indices[i]];
        }

        // 获取中心矩形框中的点云
        PointCloudRGB::Ptr cloud_rect(new PointCloudRGB);
        cloud_rect->points.resize(cloud_rect_indices.size());
        for (int i = 0; i < cloud_rect_indices.size(); i++) {
            cloud_rect->points[i] = cloud_original_->points[cloud_rect_indices[i]];
        }

        cloud_obj_region_ = cloud;
        cloud_obj_center_ = cloud_rect;
    }

    void Cloud::filterSamples(const std::vector<double> &workspace) {
        std::vector<int> indices;
        for (int i = 0; i < samples_.size(); i++) {
            if (samples_(0, i) > workspace[0] && samples_(0, i) < workspace[1] &&
                samples_(1, i) > workspace[2] && samples_(1, i) < workspace[3] &&
                samples_(2, i) > workspace[4] && samples_(2, i) < workspace[5]) {
                indices.push_back(i);
            }
        }

        Eigen::Matrix3Xd filtered_samples(3, indices.size());
        for (int i = 0; i < indices.size(); i++) {
            filtered_samples.col(i) = samples_.col(i);
        }
        samples_ = filtered_samples;
    }

    void Cloud::voxelizeCloud(float cell_size) {
        const Eigen::MatrixXf pts = cloud_processed_->getMatrixXfMap();
        Eigen::Vector3f min_xyz;
        min_xyz << pts.row(0).minCoeff(), pts.row(1).minCoeff(),
                pts.row(2).minCoeff();

        // Find the cell that each point falls into.
        std::set<Eigen::Vector4i, Cloud::UniqueVectorFirstThreeElementsComparator>
                bins;
        std::vector<Eigen::Vector3d> avg_normals;
        avg_normals.resize(pts.cols());
        std::vector<int> counts;
        counts.resize(pts.cols());

        for (int i = 0; i < pts.cols(); i++) {
            Eigen::Vector4i v4;
            v4.head(3) =
                    EigenUtils::floorVector((pts.col(i).head(3) - min_xyz) / cell_size);
            v4(3) = i;
            std::pair<
                    std::set<Eigen::Vector4i,
                            Cloud::UniqueVectorFirstThreeElementsComparator>::iterator,
                    bool>
                    res = bins.insert(v4);

            if (res.second && normals_.cols() > 0) {
                avg_normals[i] = normals_.col(i);
                counts[i] = 1;
            } else if (normals_.cols() > 0) {
                const int &idx = (*res.first)(3);
                avg_normals[idx] += normals_.col(i);
                counts[idx]++;
            }
        }

        // Calculate the point value and the average surface normal for each cell, and
        // set the camera source for each point.
        Eigen::Matrix3Xf voxels(3, bins.size());
        Eigen::Matrix3Xd normals(3, bins.size());
        Eigen::MatrixXi camera_source(camera_source_.rows(), bins.size());
        int i = 0;
        std::set<Eigen::Vector4i,
                Cloud::UniqueVectorFirstThreeElementsComparator>::iterator it;

        for (it = bins.begin(); it != bins.end(); it++) {
            voxels.col(i) = (*it).block(0, 0, 3, 1).cast<float>();
            const int &idx = (*it)(3);

            for (int j = 0; j < camera_source_.rows(); j++) {
                camera_source(j, i) = (camera_source_(j, idx) == 1) ? 1 : 0;
            }
            if (normals_.cols() > 0) {
                normals.col(i) = avg_normals[idx] / (double)counts[idx];
            }
            i++;
        }
        voxels *= cell_size;
        voxels.colwise() += min_xyz;

        // Copy the voxels into the point cloud.
        cloud_processed_->points.resize(voxels.cols());
        for (int i = 0; i < voxels.cols(); i++) {
            cloud_processed_->points[i].getVector3fMap() = voxels.col(i);
        }

        camera_source_ = camera_source;

        if (normals_.cols() > 0) {
            normals_ = normals;
        }

        printf("Voxelized cloud: %zu\n", cloud_processed_->size());
        if(cloud_processed_->isOrganized()) std::cout << "[INFO Organize] Cloud is organized after voxelization." << "\n";
        else std::cout << "[INFO Organize] Cloud is not organized after voxelization." << "\n";
    }

    void Cloud::subsample(int num_samples) {
        if (num_samples == 0) {
            return;
        }

        if (samples_.cols() > 0) {
            subsampleSamples(num_samples);
        } else if (sample_indices_.size() > 0) {
            subsampleSampleIndices(num_samples);
        } else {
            subsampleUniformly(num_samples);
        }
    }

    void Cloud::subsampleUniformly(int num_samples) {
        sample_indices_.resize(num_samples);
        pcl::RandomSample<pcl::PointXYZRGBA> random_sample;
        random_sample.setInputCloud(cloud_processed_);
        random_sample.setSample(num_samples);
        random_sample.filter(sample_indices_);
    }

    void Cloud::subsampleSamples(int num_samples) {
        if (num_samples == 0 || num_samples >= samples_.cols()) {
            return;
        } else {
            std::cout << "Using " << num_samples << " out of " << samples_.cols()
                      << " available samples.\n";
            std::vector<int> seq(samples_.cols());
            for (int i = 0; i < seq.size(); i++) {
                seq[i] = i;
            }
            std::random_shuffle(seq.begin(), seq.end());

            Eigen::Matrix3Xd subsamples(3, num_samples);
            for (int i = 0; i < num_samples; i++) {
                subsamples.col(i) = samples_.col(seq[i]);
            }
            samples_ = subsamples;

            std::cout << "Subsampled " << samples_.cols()
                      << " samples at random uniformly.\n";
        }
    }

    void Cloud::subsampleSampleIndices(int num_samples) {
        if (sample_indices_.size() == 0 || num_samples >= sample_indices_.size()) {
            return;
        }

        std::vector<int> indices(num_samples);
        for (int i = 0; i < num_samples; i++) {
            indices[i] = sample_indices_[rand() % sample_indices_.size()];
        }
        sample_indices_ = indices;
    }

    void Cloud::sampleAbovePlane() {
        double t0 = omp_get_wtime();
        printf("Sampling above plane ...\n");
        std::vector<int> indices(0);
        pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        seg.setInputCloud(cloud_processed_);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() > 0) {
            pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
            extract.setInputCloud(cloud_processed_);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(indices);
        }
        if (indices.size() > 0) {
            sample_indices_ = indices;
            printf(" Plane fit succeeded. %zu samples above plane.\n",
                   sample_indices_.size());
        } else {
            printf(" Plane fit failed. Using entire point cloud ...\n");
        }
        std::cout << " runtime (plane fit): " << omp_get_wtime() - t0 << "\n";
    }

    void Cloud::writeNormalsToFile(const std::string &filename,
                                   const Eigen::Matrix3Xd &normals) {
        std::ofstream myfile;
        myfile.open(filename.c_str());

        for (int i = 0; i < normals.cols(); i++) {
            myfile << boost::lexical_cast<std::string>(normals(0, i)) << ","
                   << boost::lexical_cast<std::string>(normals(1, i)) << ","
                   << boost::lexical_cast<std::string>(normals(2, i)) << "\n";
        }

        myfile.close();
    }

    void Cloud::calculateNormals(int num_threads) {
        double t_gpu = omp_get_wtime();
        printf("Calculating surface normals ...\n");
        std::string mode;

#if defined(USE_PCL_GPU)
        calculateNormalsGPU();
  mode = "gpu";
#else
        if (cloud_processed_->isOrganized()) {
            calculateNormalsOrganized();
            mode = "integral images";
        } else {
            printf("num_threads: %d\n", num_threads);
            calculateNormalsOMP(num_threads);
            mode = "OpenMP";
        }
#endif

        t_gpu = omp_get_wtime() - t_gpu;
        printf("Calculated %zu surface normals in %3.4fs (mode: %s).\n",
               normals_.cols(), t_gpu, mode.c_str());
        printf(
                "Reversing direction of normals that do not point to at least one camera "
                "...\n");
        reverseNormals();
    }

    void Cloud::calculateNormalsOrganized() {
        if (!cloud_processed_->isOrganized()) {
            std::cout << "Error: point cloud is not organized!\n";
            return;
        }

        std::cout << "Using integral images for surface normals estimation ...\n";
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
                new pcl::PointCloud<pcl::Normal>);
        pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
        ne.setInputCloud(cloud_processed_);
        ne.setViewPoint(view_points_(0, 0), view_points_(1, 0), view_points_(2, 0));
        ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
        ne.setNormalSmoothingSize(20.0f);
        ne.compute(*cloud_normals);

        // // Visualize them.
        // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
        // viewer->addPointCloud<pcl::PointXYZRGBA>(cloud_processed_, "cloud");
        // // Display one normal out of 20, as a line of length 3cm.
        // viewer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal>(cloud_processed_, cloud_normals, 1, 0.03, "normals");
        // while (!viewer->wasStopped())
        // {
        // 	viewer->spinOnce(100);
        // 	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        // }

        // Please use release mode to build this C++ project, or you'll get an error:
        // /usr/include/eigen3/Eigen/src/Core/PlainObjectBase.h:258: void Eigen::PlainObjectBase<Derived>::resize(Eigen::Index, Eigen::Index)
        normals_ = cloud_normals->getMatrixXfMap().cast<double>();
    }

    void Cloud::calculateNormalsOMP(int num_threads) {
        std::vector<std::vector<int>> indices = convertCameraSourceMatrixToLists();

        // Calculate surface normals for each view point.
        std::vector<PointCloudNormal::Ptr> normals_list(view_points_.cols());
        pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> estimator(
                num_threads);
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_ptr(
                new pcl::search::KdTree<pcl::PointXYZRGBA>);
        estimator.setInputCloud(cloud_processed_);
        estimator.setSearchMethod(tree_ptr);
        estimator.setRadiusSearch(0.03);
        pcl::IndicesPtr indices_ptr(new std::vector<int>);

        for (int i = 0; i < view_points_.cols(); i++) {
            PointCloudNormal::Ptr normals_cloud(new PointCloudNormal);
            indices_ptr->assign(indices[i].begin(), indices[i].end());
            estimator.setIndices(indices_ptr);
            estimator.setViewPoint(view_points_(0, i), view_points_(1, i),
                                   view_points_(2, i));
            double t0 = omp_get_wtime();
            estimator.compute(*normals_cloud);
            printf(" runtime(computeNormals): %3.4f\n", omp_get_wtime() - t0);
            normals_list[i] = normals_cloud;
            printf("camera: %d, #indices: %d, #normals: %d \n", i,
                   (int)indices[i].size(), (int)normals_list[i]->size());
        }

        // Assign the surface normals to the points.
        normals_.resize(3, camera_source_.cols());

        for (int i = 0; i < normals_list.size(); i++) {
            for (int j = 0; j < normals_list[i]->size(); j++) {
                const pcl::Normal &normal = normals_list[i]->at(j);
                normals_.col(indices[i][j]) << normal.normal_x, normal.normal_y,
                        normal.normal_z;
            }
        }
    }

#if defined(USE_PCL_GPU)
        void Cloud::calculateNormalsGPU() {
  std::vector<std::vector<int>> indices = convertCameraSourceMatrixToLists();

  PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
  pcl::copyPointCloud(*cloud_processed_, *cloud_xyz);
  pcl::gpu::Feature::PointCloud cloud_device;
  cloud_device.upload(cloud_xyz->points);
  pcl::gpu::Feature::Normals normals_device;
  pcl::gpu::NormalEstimation ne;
  ne.setInputCloud(cloud_device);
  // ne.setRadiusSearch(0.03, 1000);
  ne.setRadiusSearch(0.03, 2000);
  // ne.setRadiusSearch(0.03, 4000);
  // ne.setRadiusSearch(0.03, 8000);
  pcl::gpu::Feature::Indices indices_device;
  std::vector<pcl::PointXYZ> downloaded;
  normals_.resize(3, camera_source_.cols());

  // Calculate surface normals for each view point.
  for (int i = 0; i < view_points_.cols(); i++) {
    const Eigen::Vector3d &view_point = view_points_.col(i);
    indices_device.upload(indices[i]);
    ne.setViewPoint(view_point(0), view_point(1), view_point(2));
    ne.setIndices(indices_device);
    ne.compute(normals_device);
    normals_device.download(downloaded);

    for (int j = 0; j < indices[i].size(); j++) {
      normals_.col(indices[i][j]) =
          downloaded[i].getVector3fMap().cast<double>();
    }
  }
}
#endif

        void Cloud::reverseNormals() {
            double t1 = omp_get_wtime();
            int c = 0;

            for (int i = 0; i < normals_.cols(); i++) {
                bool needs_reverse = true;

                for (int j = 0; j < view_points_.cols(); j++) {
                    if (camera_source_(j, i) == 1)  // point is seen by this camera
                    {
                        Eigen::Vector3d cam_to_point =
                                cloud_processed_->at(i).getVector3fMap().cast<double>() -
                                view_points_.col(j);

                        if (normals_.col(i).dot(cam_to_point) <
                            0)  // normal points toward camera
                        {
                            needs_reverse = false;
                            break;
                        }
                    }
                }

                if (needs_reverse) {
                    normals_.col(i) *= -1.0;
                    c++;
                }
            }

            std::cout << " reversed " << c << " normals\n";
            std::cout << " runtime (reverse normals): " << omp_get_wtime() - t1 << "\n";
        }

        std::vector<std::vector<int>> Cloud::convertCameraSourceMatrixToLists() {
            std::vector<std::vector<int>> indices(view_points_.cols());

            for (int i = 0; i < camera_source_.cols(); i++) {
                for (int j = 0; j < view_points_.cols(); j++) {
                    if (camera_source_(j, i) == 1)  // point is seen by this camera
                    {
                        indices[j].push_back(i);
                        break;  // TODO: multiple cameras
                    }
                }
            }

            return indices;
        }

        void Cloud::setNormalsFromFile(const std::string &filename) {
            std::ifstream in;
            in.open(filename.c_str());
            std::string line;
            normals_.resize(3, cloud_original_->size());
            int i = 0;

            while (std::getline(in, line)) {
                std::stringstream lineStream(line);
                std::string cell;
                int j = 0;

                while (std::getline(lineStream, cell, ',')) {
                    normals_(i, j) = boost::lexical_cast<double>(cell);
                    j++;
                }

                i++;
            }
        }

        PointCloudRGB::Ptr Cloud::loadPointCloudFromFile(
                const std::string &filename) const {
            PointCloudRGB::Ptr cloud(new PointCloudRGB);
            std::string extension = filename.substr(filename.size() - 3);
            printf("extension: %s\n", extension.c_str());

            if (extension == "pcd" &&
                pcl::io::loadPCDFile<pcl::PointXYZRGBA>(filename, *cloud) == -1) {
                printf("Couldn't read PCD file: %s\n", filename.c_str());
                cloud->points.resize(0);
            } else if (extension == "ply" &&
                       pcl::io::loadPLYFile<pcl::PointXYZRGBA>(filename, *cloud) == -1) {
                printf("Couldn't read PLY file: %s\n", filename.c_str());
                cloud->points.resize(0);
            }

            return cloud;
        }

        void Cloud::setSamples(const Eigen::Matrix3Xd &samples) { samples_ = samples; }

}  // namespace util
}  // namespace gpd
