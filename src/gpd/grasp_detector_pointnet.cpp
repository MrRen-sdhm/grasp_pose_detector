#include <gpd/grasp_detector_pointnet.h>

namespace gpd {

    GraspDetectorPointNet::GraspDetectorPointNet(const std::string &config_filename) {
        Eigen::initParallel();

        // Read parameters from configuration file.
        util::ConfigFile config_file(config_filename);
        config_file.ExtractKeys();

        // Read hand geometry parameters.
        std::string hand_geometry_filename =
                config_file.getValueOfKeyAsString("hand_geometry_filename", "");
        if (hand_geometry_filename == "0") {
            hand_geometry_filename = config_filename;
        }
        candidate::HandGeometry hand_geom(hand_geometry_filename);
        std::cout << hand_geom;

        // Read plotting parameters.
        plot_normals_ = config_file.getValueOfKey<bool>("plot_normals", false);
        plot_samples_ = config_file.getValueOfKey<bool>("plot_samples", true);
        plot_candidates_ = config_file.getValueOfKey<bool>("plot_candidates", false);
        plot_workspace_ = config_file.getValueOfKey<bool>("plot_workspace", false);
        plot_filtered_candidates_ =
                config_file.getValueOfKey<bool>("plot_filtered_candidates", false);
        plot_valid_grasps_ =
                config_file.getValueOfKey<bool>("plot_valid_grasps", false);
        plot_clustered_grasps_ =
                config_file.getValueOfKey<bool>("plot_clustered_grasps", false);
        plot_selected_grasps_ =
                config_file.getValueOfKey<bool>("plot_selected_grasps", false);
        printf("============ PLOTTING ========================\n");
        printf("plot_normals: %s\n", plot_normals_ ? "true" : "false");
        printf("plot_samples %s\n", plot_samples_ ? "true" : "false");
        printf("plot_candidates: %s\n", plot_candidates_ ? "true" : "false");
        printf("plot_workspace: %s\n", plot_workspace_ ? "true" : "false");
        printf("plot_filtered_candidates: %s\n", plot_filtered_candidates_ ? "true" : "false");
        printf("plot_valid_grasps: %s\n", plot_valid_grasps_ ? "true" : "false");
        printf("plot_clustered_grasps: %s\n", plot_clustered_grasps_ ? "true" : "false");
        printf("plot_selected_grasps: %s\n", plot_selected_grasps_ ? "true" : "false");
        printf("==============================================\n");

        // Create object to generate grasp candidates.
        candidate::CandidatesGenerator::Parameters generator_params;
        generator_params.num_samples_ =
                config_file.getValueOfKey<int>("num_samples", 1000);
        generator_params.num_threads_ =
                config_file.getValueOfKey<int>("num_threads", 1);
        generator_params.remove_statistical_outliers_ =
                config_file.getValueOfKey<bool>("remove_outliers", false);
        generator_params.sample_above_plane_ =
                config_file.getValueOfKey<bool>("sample_above_plane", false);
        generator_params.reverse_normals_ =
                config_file.getValueOfKey<bool>("reverse_normals", false);
        generator_params.voxelize_ =
                config_file.getValueOfKey<bool>("voxelize", true);
        generator_params.voxel_size_ =
                config_file.getValueOfKey<double>("voxel_size", 0.003);
        generator_params.workspace_ =
                config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");

        candidate::HandSearch::Parameters hand_search_params;
        hand_search_params.hand_geometry_ = hand_geom;
        hand_search_params.nn_radius_frames_ =
                config_file.getValueOfKey<double>("nn_radius", 0.01);
        hand_search_params.num_samples_ =
                config_file.getValueOfKey<int>("num_samples", 1000);
        hand_search_params.num_threads_ =
                config_file.getValueOfKey<int>("num_threads", 1);
        hand_search_params.num_orientations_ =
                config_file.getValueOfKey<int>("num_orientations", 8);
        hand_search_params.num_finger_placements_ =
                config_file.getValueOfKey<int>("num_finger_placements", 10);
        hand_search_params.deepen_hand_ =
                config_file.getValueOfKey<bool>("deepen_hand", true);
        hand_search_params.hand_axes_ =
                config_file.getValueOfKeyAsStdVectorInt("hand_axes", "2");
        hand_search_params.friction_coeff_ =
                config_file.getValueOfKey<double>("friction_coeff", 20.0);
        hand_search_params.min_viable_ =
                config_file.getValueOfKey<int>("min_viable", 6);
        candidates_generator_ = std::make_unique<candidate::CandidatesGenerator>(
                generator_params, hand_search_params);

        printf("============ CLOUD PREPROCESSING =============\n");
        printf("voxelize: %s\n", generator_params.voxelize_ ? "true" : "false");
        printf("voxl_size: %.3f\n", generator_params.voxel_size_);
        printf("remove_outliers: %s\n",
               generator_params.remove_statistical_outliers_ ? "true" : "false");
        printStdVector(generator_params.workspace_, "workspace");
        printf("sample_above_plane: %s\n",
               generator_params.sample_above_plane_ ? "true" : "false");
        printf("==============================================\n");

        printf("============ CANDIDATE GENERATION ============\n");
        printf("num_samples: %d\n", hand_search_params.num_samples_);
        printf("num_threads: %d\n", hand_search_params.num_threads_);
        printf("nn_radius: %3.2f\n", hand_search_params.nn_radius_frames_);
        printStdVector(hand_search_params.hand_axes_, "hand axes");
        printf("num_orientations: %d\n", hand_search_params.num_orientations_);
        printf("num_finger_placements: %d\n",
               hand_search_params.num_finger_placements_);
        printf("deepen_hand: %s\n",
               hand_search_params.deepen_hand_ ? "true" : "false");
        printf("friction_coeff: %3.2f\n", hand_search_params.friction_coeff_);
        printf("min_viable: %d\n", hand_search_params.min_viable_);
        printf("==============================================\n");

        // Read classification parameters and create classifier.
        std::string model_file = config_file.getValueOfKeyAsString("model_file", "");
        std::string weights_file = config_file.getValueOfKeyAsString("weights_file", "");
        int device = config_file.getValueOfKey<int>("device", 0);

        if (!model_file.empty() || !weights_file.empty()) {
            int batch_size = config_file.getValueOfKey<int>("batch_size", 256);
            classifier_ = net::Classifier::create(
                    model_file, weights_file, static_cast<net::Classifier::Device>(device), batch_size);
            min_score_ = config_file.getValueOfKey<double>("min_score", 0);
            printf("============ CLASSIFIER ======================\n");
            printf("model_file: %s\n", model_file.c_str());
            printf("weights_file: %s\n", weights_file.c_str());
            printf("batch_size: %d\n", batch_size);
            printf("min_score: %.2f\n", min_score_);
            printf("==============================================\n");
        }

        // Read additional grasp image creation parameters.
        bool remove_plane = config_file.getValueOfKey<bool>(
                "remove_plane_before_image_calculation", false);

        int grasp_points_num = config_file.getValueOfKey<int>("grasp_points_num", 750);
        int min_points_limit = config_file.getValueOfKey<int>("min_points_limit", 100);
        float min_points_depth = config_file.getValueOfKey<double>("min_points_depth", 0.01);
        printf("grasp_points_num: %d\n", grasp_points_num);
        printf("min_points_limit: %d\n", min_points_limit);
        printf("min_points_depth: %.3f\n", min_points_depth);

        // Create object to create grasp points from grasp candidates (used for
        // classification).
        point_generator_ = std::make_unique<descriptor::PointGenerator>(
                hand_geom, hand_search_params.num_threads_, hand_search_params.num_orientations_, grasp_points_num,
                min_points_limit, min_points_depth, remove_plane);

        // Read grasp filtering parameters based on robot workspace and gripper width.
        workspace_grasps_ = config_file.getValueOfKeyAsStdVectorDouble(
                "workspace_grasps", "-1 1 -1 1 -1 1");
        min_aperture_ = config_file.getValueOfKey<double>("min_aperture", 0.0);
        max_aperture_ = config_file.getValueOfKey<double>("max_aperture", 0.085);
        printf("============ CANDIDATE FILTERING =============\n");
        printStdVector(workspace_grasps_, "candidate_workspace");
        printf("min_aperture: %3.4f\n", min_aperture_);
        printf("max_aperture: %3.4f\n", max_aperture_);
        printf("==============================================\n");

        // Read grasp filtering parameters based on approach direction.
        filter_approach_direction_ =
                config_file.getValueOfKey<bool>("filter_approach_direction", false);
        std::vector<double> approach =
                config_file.getValueOfKeyAsStdVectorDouble("direction", "1 0 0");
        direction_ << approach[0], approach[1], approach[2];
        thresh_rad_ = config_file.getValueOfKey<double>("thresh_rad", 2.3);

        // Read clustering parameters.
        int min_inliers = config_file.getValueOfKey<int>("min_inliers", 1);
        clustering_ = std::make_unique<Clustering>(min_inliers);
        cluster_grasps_ = min_inliers > 0 ? true : false;
        printf("============ CLUSTERING ======================\n");
        printf("min_inliers: %d\n", min_inliers);
        printf("==============================================\n\n");

        // Read grasp selection parameters.
        num_selected_ = config_file.getValueOfKey<int>("num_selected", 100);

        // Create plotter.
        plotter_ = std::make_unique<util::Plot>(hand_search_params.hand_axes_.size(),
                                                hand_search_params.num_orientations_);
    }

    std::vector<std::unique_ptr<candidate::Hand>> GraspDetectorPointNet::detectGrasps(
            util::Cloud &cloud) {
        double t0_total = omp_get_wtime();
        std::vector<std::unique_ptr<candidate::Hand>> hands_out;

        const candidate::HandGeometry &hand_geom =
                candidates_generator_->getHandSearchParams().hand_geometry_;

        // Check if the point cloud is empty.
        if (cloud.getCloudOriginal()->size() == 0) {
            printf("[WARN] Point cloud is empty!");
            hands_out.resize(0);
            return hands_out;
        }

        // Plot samples/indices.
        if (plot_samples_) {
            if (cloud.getSamples().cols() > 0) {
                plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
            } else if (cloud.getSampleIndices().size() > 0) {
                plotter_->plotSamples(cloud.getSampleIndices(),
                                      cloud.getCloudProcessed());
            }
        }

        // Plot normals.
        if (plot_normals_) {
            std::cout << "Plotting normals for different camera sources\n";
            plotter_->plotNormals(cloud.getCloudProcessed(), cloud.getNormals());
        }

        // 1. Generate grasp candidates.
        double t0_candidates = omp_get_wtime();
        std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list = candidates_generator_->generateGraspCandidateSets(cloud);
        printf("[INFO] Generated %zu hand sets.\n", hand_set_list.size());
        if (hand_set_list.empty()) {
            return hands_out;
        }
        double t_candidates = omp_get_wtime() - t0_candidates;
        if (plot_candidates_) {
            plotter_->plotFingers3D(hand_set_list, cloud.getCloudProcessed(), "Grasp candidates", hand_geom);
        }

        // 2. Filter the candidates.
        double t0_filter = omp_get_wtime();
        std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered = filterGraspsWorkspace(hand_set_list, workspace_grasps_);
        if (hand_set_list_filtered.empty()) {
            return hands_out;
        }

        if (plot_workspace_) {
            plotter_->plotCloud(cloud.getCloudProcessed(), "Filtered Grasps Workspace");
        }
        if (plot_filtered_candidates_) {
            plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudProcessed(), "Filtered Grasps (Aperture, Workspace)", hand_geom);
        }
        if (filter_approach_direction_) {
            hand_set_list_filtered = filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
            if (plot_filtered_candidates_) {
                plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudProcessed(), "Filtered Grasps (Approach)", hand_geom);
            }
        }
        double t_filter = omp_get_wtime() - t0_filter;
        if (hand_set_list_filtered.empty()) {
            return hands_out;
        }

        // 3. Create points in the hand closing area.
        double t0_points = omp_get_wtime();
        std::vector<std::unique_ptr<candidate::Hand>> hands;
        std::vector<std::unique_ptr<Eigen::Matrix3Xd>> point_groups;
        point_generator_->getCloud(cloud);
        point_generator_->createPointGroups(cloud, hand_set_list_filtered, point_groups, hands);
        double t_points = omp_get_wtime() - t0_points;

        if (point_groups.empty()) { // created 0 point_groups
            printf("[WARN] Created 0 point groups!\n");
            return hands;
        }

        // 4. Classify the grasp candidates by pointnet.
        double t0_classify = omp_get_wtime();
        std::vector<double> scores = classifier_->classifyPointsBatch(point_groups);


        for (int i = 0; i < hands.size(); i++) {
            hands[i]->setScore(scores[i]);
        }
        double t_classify = omp_get_wtime() - t0_classify;

        // 5. Select the <num_selected> highest scoring grasps.
        hands = selectGrasps(hands);
        if (plot_valid_grasps_) {
            plotter_->plotFingers3D(hands, cloud.getCloudProcessed(), "Valid Grasps", hand_geom, false);
        }

        // 6. Cluster the grasps.
        double t0_cluster = omp_get_wtime();
        std::vector<std::unique_ptr<candidate::Hand>> clusters;
        if (cluster_grasps_) {
            clusters = clustering_->findClusters(hands);
            printf("Found %d clusters.\n", (int)clusters.size());
            if (clusters.size() <= 3) {
                printf("Not enough clusters found! Adding all grasps from previous step.");
                for (int i = 0; i < hands.size(); i++) {
                    clusters.push_back(std::move(hands[i]));
                }
            }
            if (plot_clustered_grasps_) {
                plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(), "Clustered Grasps", hand_geom);
            }
        } else {
            clusters = std::move(hands);
        }
        double t_cluster = omp_get_wtime() - t0_cluster;

        // 7. Sort grasps by their score.
        std::sort(clusters.begin(), clusters.end(), isScoreGreater);
        printf("\033[0;36m%s\033[0m\n", "======== SELECTED GRASPS ========");
        for (int i = 0; i < clusters.size(); i++) {
            std::cout << "Grasp " << i << ": " << clusters[i]->getScore() << "\n";
        }
        printf("Selected the %d best grasps.\n", (int)clusters.size());
        double t_total = omp_get_wtime() - t0_total;

        printf("\033[0;32m%s\033[0m\n", "========    RUNTIMES     ========");
        printf(" 1. Candidate generation: %3.4fs\n", t_candidates);
        printf(" 2. Descriptor extraction: %3.4fs\n", t_points);
        printf(" 3. Classification: %3.4fs\n", t_classify);
        // printf(" Filtering: %3.4fs\n", t_filter);
        // printf(" Clustering: %3.4fs\n", t_cluster);
        printf("==========\n");
        printf(" TOTAL: %3.4fs\n\n", t_total);

    //  const candidate::Hand &hand = *clusters[0];
    //  std::cout << "grasp orientation:\n" << hand.getFrame() << std::endl;
    //  std::cout << "grasp position: " << hand.getPosition().transpose()
    //            << std::endl;

        if (plot_selected_grasps_) {
//            plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
//                                    "Selected Grasps", hand_geom, false);
            plotter_->plotFingers3D(clusters, cloud.getCloudProcessed(), "Clustered Grasps", hand_geom, true); // 单色显示
            plotter_->plotFingers3DHighestScore3(clusters, cloud.getCloudProcessed(), "Selected Grasps", hand_geom); // RGB显示前三个
            plotter_->plotFingers3DHighestScore(clusters, cloud.getCloudProcessed(), "Selected Grasps", hand_geom, false); // 最高分显示为红色
            plotter_->plotFingers3DHighestScore(clusters, cloud.getCloudProcessed(), "Selected Grasps", hand_geom, true); // 仅显示最高分
        }

        return clusters;

    }

    void GraspDetectorPointNet::preprocessPointCloud(util::Cloud &cloud) {
        candidates_generator_->preprocessPointCloud(cloud);
    }

    void GraspDetectorPointNet::preprocessPointCloud(util::Cloud &cloud, cv::Rect rect) {
        candidates_generator_->preprocessPointCloud(cloud, rect);
    }

    std::vector<std::unique_ptr<candidate::HandSet>> GraspDetectorPointNet::filterGraspsWorkspace(
            std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list, const std::vector<double> &workspace) const {
        int remaining = 0;
        int hands_cnt = 0;
        int valid_hands_cnt = 0;
        std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
        printf("Filtering grasps outside of workspace ...\n");

        const candidate::HandGeometry &hand_geometry = candidates_generator_->getHandSearchParams().hand_geometry_;

        for (int i = 0; i < hand_set_list.size(); i++) {
            const std::vector<std::unique_ptr<candidate::Hand>> &hands = hand_set_list[i]->getHands();
            Eigen::Array<bool, 1, Eigen::Dynamic> is_valid = hand_set_list[i]->getIsValid();

            for (int j = 0; j < hands.size(); j++) {
                if (!is_valid(j)) {
                    continue;
                }

                double half_width = 0.5 * hand_geometry.outer_diameter_;
                Eigen::Vector3d left_bottom = hands[j]->getPosition() + half_width * hands[j]->getBinormal();
                Eigen::Vector3d right_bottom = hands[j]->getPosition() - half_width * hands[j]->getBinormal();
                Eigen::Vector3d left_top = left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
                Eigen::Vector3d right_top = left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
                Eigen::Vector3d approach = hands[j]->getPosition() - 0.05 * hands[j]->getApproach();
                Eigen::VectorXd x(5), y(5), z(5);
                x << left_bottom(0), right_bottom(0), left_top(0), right_top(0), approach(0);
                y << left_bottom(1), right_bottom(1), left_top(1), right_top(1), approach(1);
                z << left_bottom(2), right_bottom(2), left_top(2), right_top(2), approach(2);

                // Ensure the object fits into the hand and avoid grasps outside the
                // workspace.
                if (hands[j]->getGraspWidth() >= min_aperture_ &&
                    hands[j]->getGraspWidth() <= max_aperture_ &&
                    x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] &&
                    y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] &&
                    z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) {
                    is_valid(j) = true;
                    remaining++;
                } else {
                    is_valid(j) = false;
                }

                valid_hands_cnt += 1;
            }

            if (is_valid.any()) {
                hand_set_list_out.push_back(std::move(hand_set_list[i]));
                hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
            }

            hands_cnt += hands.size();
        }

        printf("[INFO] Number of grasp candidates generated: %d\n", hands_cnt);
        printf("[INFO] Number of grasp candidates valid: %d\n", valid_hands_cnt);
        printf("[INFO] Number of grasp candidates within workspace and gripper width: %d\n", remaining);

        return hand_set_list_out;
    }

    std::vector<std::unique_ptr<candidate::HandSet>>
    GraspDetectorPointNet::generateGraspCandidates(const util::Cloud &cloud) {
        return candidates_generator_->generateGraspCandidateSets(cloud);
    }

    std::vector<std::unique_ptr<candidate::Hand>> GraspDetectorPointNet::selectGrasps(
            std::vector<std::unique_ptr<candidate::Hand>> &hands) const {
//        printf("Selecting the %d highest scoring grasps ...\n", num_selected_);

        int middle = std::min((int)hands.size(), num_selected_);
        std::partial_sort(hands.begin(), hands.begin() + middle, hands.end(), isScoreGreater);
        std::vector<std::unique_ptr<candidate::Hand>> hands_out;

        for (int i = 0; i < middle; i++) {
            if (hands[i]->getScore() > min_score_)  hands_out.push_back(std::move(hands[i]));
//            printf(" grasp #%d, score: %3.4f\n", i, hands_out[i]->getScore());
        }

        return hands_out;
    }

    std::vector<std::unique_ptr<candidate::HandSet>>
    GraspDetectorPointNet::filterGraspsDirection(
            std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
            const Eigen::Vector3d &direction, const double thresh_rad) {
        std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
        int remaining = 0;

        for (int i = 0; i < hand_set_list.size(); i++) {
            const std::vector<std::unique_ptr<candidate::Hand>> &hands =
                    hand_set_list[i]->getHands();
            Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
                    hand_set_list[i]->getIsValid();

            for (int j = 0; j < hands.size(); j++) {
                if (is_valid(j)) {
                    double angle = acos(direction.transpose() * hands[j]->getApproach());
                    if (angle > thresh_rad) {
                        is_valid(j) = false;
                    } else {
                        remaining++;
                    }
                }
            }

            if (is_valid.any()) {
                hand_set_list_out.push_back(std::move(hand_set_list[i]));
                hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
            }
        }

        printf("Number of grasp candidates with correct approach direction: %d\n",
               remaining);

        return hand_set_list_out;
    }

    void GraspDetectorPointNet::printStdVector(const std::vector<int> &v,
                                               const std::string &name) const {
        printf("%s: ", name.c_str());
        for (int i = 0; i < v.size(); i++) {
            printf("%d ", v[i]);
        }
        printf("\n");
    }

    void GraspDetectorPointNet::printStdVector(const std::vector<double> &v,
                                               const std::string &name) const {
        printf("%s: ", name.c_str());
        for (int i = 0; i < v.size(); i++) {
            printf("%3.2f ", v[i]);
        }
        printf("\n");
    }

}  // namespace gpd
