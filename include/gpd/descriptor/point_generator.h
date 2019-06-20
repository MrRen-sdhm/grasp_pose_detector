/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef IMAGE_GENERATOR_H_
#define IMAGE_GENERATOR_H_

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include <random>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>

#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

#include <gpd/candidate/hand_set.h>
#include <gpd/util/plot.h>
#include <gpd/util/eigen_utils.h>

typedef std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> Matrix3XdPair;
typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

#define DEBUG 0 // use for cloud visualize

namespace gpd {
    namespace descriptor {

        /**
         *
         * \brief Create grasp images for classification.
         *
         * Creates images for the input layer of a convolutional neural network. Each
         * image represents a grasp candidate. We call these "grasp images".
         *
         */
        class PointGenerator {
        public:
            /**
             * \brief Constructor.
             * \param params parameters for grasp images
             * \param num_threads number of CPU threads to be used
             * \param is_plotting if images are visualized
             * \param remove_plane if the support/table plane is removed before
             * calculating images
             */
            PointGenerator(const candidate::HandGeometry &hand_geometry, int num_threads, int num_orientations,
                    int grasp_points_num, int min_point_limit, bool is_plotting, bool remove_plane);

            /**
             * \brief Create a list of point groups for a given list of grasp candidates.
             * \param cloud_cam the point cloud
             * \param hand_set_list the list of grasp candidates
             * \return the list of grasp images
             */
            void createPointGroups(
                    const util::Cloud &cloud_cam,
                    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
                    std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups_out,
                    std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const;

            inline void getCloud(util::Cloud &cloud) {
                cloud_ = cloud;
//                plotter_ = std::make_unique<util::Plot>(0, 8);
//                plotter_->plotCloud(cloud.getCloudOriginal(), "cloud");
            }

        private:
            /**
             * \brief Remove the plane from the point cloud. Sets <point_list> to all
             * non-planar points if the plane is found, otherwise <point_list> has the
             * same points as <cloud>.
             * \param cloud the cloud
             * \param point_list the list of points corresponding to the cloud
             */
            void removePlane(const util::Cloud &cloud_cam,
                             util::PointList &point_list) const;

            void createPointList(
                    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
                    const std::vector<util::PointList> &nn_points_list,
                    std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups_out,
                    std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const;

            std::vector<std::unique_ptr<Eigen::Matrix3Xd>> createPointGroups(
                    const candidate::HandSet &hand_set, const util::PointList &nn_points) const;

            void createPointGroup(const util::PointList &point_list,
                                                  const candidate::Hand &hand, Eigen::Matrix3Xd &point_groups) const;

            Eigen::Matrix3Xd transformToHand(
                    const util::PointList &point_list, const candidate::Hand &hand) const;

            Eigen::Matrix3Xd findPointsInHand(
                    const candidate::Hand &hand, const Eigen::Matrix3Xd &points) const;

            void plotHand3D(pcl::visualization::PCLVisualizer &viewer, const candidate::Hand &hand,
                                            const candidate::HandGeometry &geometry, int idx,
                                            const Eigen::Vector3d &rgb) const;

            void plotHand3D(pcl::visualization::PCLVisualizer &viewer, const candidate::Hand &hand,
                                            double outer_diameter, double finger_width,
                                            double hand_depth, double hand_height, int idx,
                                            const Eigen::Vector3d &rgb) const;

            void plotHand3D(pcl::visualization::PCLVisualizer &viewer, Eigen::Vector3d position,
                                            Eigen::Matrix3d frame, double outer_diameter, double finger_width,
                                            double hand_depth, double hand_height, int idx,
                                            const Eigen::Vector3d &rgb) const;

            void plotCube(pcl::visualization::PCLVisualizer &viewer, const Eigen::Vector3d &position,
                                          const Eigen::Quaterniond &rotation, double width,
                                          double height, double depth, const std::string &name,
                                          const Eigen::Vector3d &rgb) const;

            int num_threads_;
            int num_orientations_;
            int grasp_points_num_;
            int min_point_limit_;
            candidate::HandGeometry hand_geometry_;
            bool is_plotting_;
            bool remove_plane_;
            util::Cloud cloud_;
            std::unique_ptr<util::Plot> plotter_;
        };


        /**
         * \brief generate non-repetitive random numbers.
         */
        template <typename IntType = int>
        class distinct_uniform_int_distribution {
        public:
            using result_type = IntType;

        private:
            using set_type    = std::unordered_set<result_type>;
            using distr_type  = std::uniform_int_distribution<result_type>;

        public:
            distinct_uniform_int_distribution(result_type inf, result_type sup) :
                    inf_(inf),
                    sup_(sup),
                    range_(sup_ - inf_ + 1),
                    distr_(inf_, sup_)
            {}
            void reset() {
                uset_.clear();
                distr_.reset();
            }

            template <typename Generator>
            result_type operator()(Generator& engine) {
                if (not(uset_.size() < range_)) { std::terminate(); }
                result_type res;
                do { res = distr_(engine); } while (uset_.count(res) > 0);
                uset_.insert(res);
                return res;
            }

            result_type min() const { return inf_; }
            result_type max() const { return sup_; }

        private:
            const result_type inf_;
            const result_type sup_;
            const size_t      range_;
            distr_type        distr_;
            set_type          uset_;
        };

    }  // namespace descriptor
}  // namespace gpd

#endif /* IMAGE_GENERATOR_H_ */
