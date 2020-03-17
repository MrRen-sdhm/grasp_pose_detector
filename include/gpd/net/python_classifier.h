#ifndef PYTHON_CLASSIFIER_H_
#define PYTHON_CLASSIFIER_H_

// System
#include <string>
#include <string.h>
#include <vector>
#include <exception>

// Python
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <Python.h>

#include <fstream>
#include <sys/time.h>
#include <iostream>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// OpenCV
#include <opencv2/core/core.hpp>

#include <gpd/net/classifier.h>

#define DEGUBPYTHON 0 // use for debug

using namespace std;
using namespace boost::python;

#if (PY_VERSION_HEX >= 0x03000000)
static void *init_ar() {
#else
static void init_ar(){
#endif
    Py_Initialize();

    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

namespace gpd {
namespace net {

    /**
     *
     * \brief Classify grasp candidates as viable grasps or not with Caffe
     *
     * Classifies grasps as viable or not using a convolutional neural network (CNN).
     *
     */
    class PythonClassifier : public Classifier {
    public:
        /**
         * \brief Constructor.
         * \param model_file the location of the file that describes the network model
         * \param weights_file the location of the file that contains the network
         * weights
         */
        PythonClassifier(const std::string &model_file,
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
        PyObject *pModule;
        PyObject *pDict;
        string module_name_; // python script name
        string func_name_; // function name
        int batch_size_;
        bool gil_init = false;

    };
}  // namespace net
}  // namespace gpd

#endif /* LIBTORCH_CLASSIFIER_H_ */
