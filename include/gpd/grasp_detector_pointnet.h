/*
 Created by Zhenyu Ren.
 */

#ifndef GRASP_DETECTOR_H_
#define GRASP_DETECTOR_H_

// System
#include <algorithm>
#include <memory>
#include <vector>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gpd/candidate/candidates_generator.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/clustering.h>
#include <gpd/descriptor/point_generator.h>
#include <gpd/net/classifier.h>
#include <gpd/util/config_file.h>
#include <gpd/util/plot.h>

namespace gpd {

    /**
     *
     * \brief Detect grasp poses in point clouds.
     *
     * This class detects grasp poses in a point clouds by first creating a large
     * set of grasp candidates, and then classifying each of them as a grasp or not.
     *
     */
    class GraspDetectorPointNet {
    public:
        /**
         * \brief Constructor.
         * \param node ROS node handle
         */
        GraspDetectorPointNet(const std::string &config_filename);

        /**
         * \brief Detect grasps in a point cloud.
         * \param cloud_cam the point cloud
         * \return list of grasps
         */
        std::vector<std::unique_ptr<candidate::Hand>> detectGrasps(
                util::Cloud &cloud);

        /**
         * \brief Preprocess the point cloud.
         * \param cloud_cam the point cloud
         */
        void preprocessPointCloud(util::Cloud &cloud);

        /**
        * \brief Preprocess the point cloud.
        * \param cloud_cam the point cloud
        * \param rect the region object exist
        */
        void preprocessPointCloud(util::Cloud &cloud, cv::Rect rect);

        /**
         * Filter grasps based on the robot's workspace.
         * \param hand_set_list list of grasp candidate sets
         * \param workspace the robot's workspace as a 3D cube, centered at the origin
         * \param thresh_rad the angle in radians above which grasps are filtered
         * \return list of grasps after filtering
         */
        std::vector<std::unique_ptr<candidate::HandSet>> filterGraspsWorkspace(
                std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
                const std::vector<double> &workspace) const;

        /**
         * Filter grasps based on their approach direction.
         * \param hand_set_list list of grasp candidate sets
         * \param direction the direction used for filtering
         * \param thresh_rad the angle in radians above which grasps are filtered
         * \return list of grasps after filtering
         */
        std::vector<std::unique_ptr<candidate::HandSet>> filterGraspsDirection(
                std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
                const Eigen::Vector3d &direction, const double thresh_rad);

        /**
         * \brief Generate grasp candidates.
         * \param cloud the point cloud
         * \return the list of grasp candidates
         */
        std::vector<std::unique_ptr<candidate::HandSet>> generateGraspCandidates(
                const util::Cloud &cloud);

        /**
         * \brief Select the k highest scoring grasps.
         * \param hands the grasps
         * \return the k highest scoring grasps
         */
        std::vector<std::unique_ptr<candidate::Hand>> selectGrasps(
                std::vector<std::unique_ptr<candidate::Hand>> &hands) const;

        /**
         * \brief Compare the scores of two given grasps.
         * \param hand1 the first grasp to be compared
         * \param hand1 the second grasp to be compared
         * \return `true` if \param hand1 has a larger score than \param hand2,
         * `false` otherwise
         */
        static bool isScoreGreater(const std::unique_ptr<candidate::Hand> &hand1,
                                   const std::unique_ptr<candidate::Hand> &hand2) {
            return hand1->getScore() > hand2->getScore();
        }

        /**
         * \brief Return the hand search parameters.
         * \return the hand search parameters
         */
        const candidate::HandSearch::Parameters &getHandSearchParameters() {
            return candidates_generator_->getHandSearchParams();
        }

    private:
        void printStdVector(const std::vector<int> &v, const std::string &name) const;

        void printStdVector(const std::vector<double> &v,
                            const std::string &name) const;

        std::unique_ptr<candidate::CandidatesGenerator> candidates_generator_;
        std::unique_ptr<descriptor::PointGenerator> point_generator_;
        std::unique_ptr<Clustering> clustering_;
        std::unique_ptr<util::Plot> plotter_;
        std::shared_ptr<net::Classifier> classifier_;

        // classification parameters
        double min_score_;           ///< minimum classifier confidence score
        /// memory usage)

        // plotting parameters
        bool plot_normals_;              ///< if normals are plotted
        bool plot_samples_;              ///< if samples/indices are plotted
        bool plot_candidates_;           ///< if grasp candidates are plotted
        bool plot_workspace_;            ///< if grasp workspace are plotted
        bool plot_filtered_candidates_;  ///< if filtered grasp candidates are plotted
        bool plot_valid_grasps_;         ///< if valid grasps are plotted
        bool plot_clustered_grasps_;     ///< if clustered grasps are plotted
        bool plot_selected_grasps_;      ///< if selected grasps are plotted

        // filtering parameters
        bool cluster_grasps_;  ///< if grasps are clustered
        double min_aperture_;  ///< the minimum opening width of the robot hand
        double max_aperture_;  ///< the maximum opening width of the robot hand
        std::vector<double> workspace_grasps_;  ///< the workspace of the robot with
        /// respect to hand poses
        bool filter_approach_direction_;
        Eigen::Vector3d direction_;
        double thresh_rad_;

        // selection parameters
        int num_selected_;  ///< the number of selected grasps
    };

}  // namespace gpd

#endif /* GRASP_DETECTOR_H_ */
