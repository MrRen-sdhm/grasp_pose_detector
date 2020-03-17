#ifndef LIBTORCH_CLASSIFIER_H_
#define LIBTORCH_CLASSIFIER_H_

// System
#include <string>
#include <vector>
#include <exception>

// Libtorch
#include <torch/script.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// OpenCV
#include <opencv2/core/core.hpp>

#include <gpd/net/classifier.h>

#define DEGUBLIBTORCH 0 // use for debug

namespace gpd {
    namespace net {

        /**
         *
         * \brief Classify grasp candidates as viable grasps or not with Caffe
         *
         * Classifies grasps as viable or not using a convolutional neural network (CNN)
         *  with the Caffe framework.
         *
         */
        class LibtorchClassifier : public Classifier {
        public:
            /**
             * \brief Constructor.
             * \param model_file the location of the file that describes the network model
             * \param weights_file the location of the file that contains the network
             * weights
             */
            LibtorchClassifier(const std::string &model_file,
                            const std::string &weights_file, Classifier::Device device, int batch_size);

            /**
             * \brief Classify grasp candidates as viable grasps or not.
             * \param image_list the list of grasp images
             * \return the classified grasp candidates
             */
            std::vector<double> classifyImages(
                    const std::vector<std::unique_ptr<cv::Mat>> &image_list);

            /**
             * \brief Classify grasp candidates as viable grasps or not.
             * \param point_list the points in the hand closed area.
             * \return the classified grasp candidates
             */
            std::vector<double> classifyPoints(
                    const std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups);

            /**
             * \brief Classify grasp candidates as viable grasps or not.
             * \param point_list the points in the hand closed area.
             * \return the classified grasp candidates
             */
            std::vector<double> classifyPointsBatch(
                    const std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups);

            int getBatchSize() const { return batch_size_; }

        private:
            std::shared_ptr<torch::jit::script::Module> module_;
            bool use_cuda_;
            int batch_size_;

        };

    }  // namespace net
}  // namespace gpd

#endif /* LIBTORCH_CLASSIFIER_H_ */
