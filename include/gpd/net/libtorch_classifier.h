#ifndef LIBTORCH_CLASSIFIER_H_
#define LIBTORCH_CLASSIFIER_H_

// System
#include <string>
#include <vector>
#include <exception>

// Libtorch
#include <torch/script.h>

// OpenCV
#include <opencv2/core/core.hpp>

#include <gpd/net/classifier.h>

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
            std::vector<float> classifyImages(
                    const std::vector<std::unique_ptr<cv::Mat>> &image_list);

            int getBatchSize() const { return batch_size; }

        private:
            std::shared_ptr<torch::jit::script::Module> module_;
            bool use_cuda_;
            int batch_size;

        };

    }  // namespace net
}  // namespace gpd

#endif /* LIBTORCH_CLASSIFIER_H_ */
