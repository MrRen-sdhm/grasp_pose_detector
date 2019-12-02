#include <gpd/net/python_classifier.h>

namespace gpd {
namespace net {

PythonClassifier::PythonClassifier(const std::string &model_file,
                                 const std::string &weights_file,
                                 Classifier::Device device, int batch_size) : batch_size_(batch_size) {

    // Load pretrained network.
    init_ar();
    char str[] = "Python";
    Py_SetProgramName(str);
    if(!Py_IsInitialized())
        cout << "init faild/n" << endl;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../python')");

    /* import */
    pModule = PyImport_ImportModule("ndarray_module");
    if (pModule == NULL) {
        cout<<"ERROR importing module"<<endl;
    }

    pDict = PyModule_GetDict(pModule);
}

std::vector<float> PythonClassifier::classifyImages(
        const std::vector<std::unique_ptr<cv::Mat>> &image_list) {
}

std::vector<float> PythonClassifier::classifyPoints(
        const std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups) {
    std::vector<float> predictions;
    const int points_num = point_groups[0]->cols(); // 点数
    printf("[DEBUG] points num:%d", points_num);

//    cout << point_groups[0] << endl;

    npy_intp Dims[3] = {3, points_num, 3}; //给定维度信息

    PyObject *PyArray  = PyArray_SimpleNewFromData(3, Dims, NPY_DOUBLE, point_groups[0]->data()); //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    PyTuple_SetItem(ArgArray, 0, PyArray);

    PyObject *pFunc = PyDict_GetItemString(pDict, "PassArrayFromCToPython");
    PyObject_CallObject(pFunc, ArgArray);//调用函数，传入Numpy Array 对象

//    for(int i = 0; i < pred_2.size(0); i++) {
//        predictions.push_back(pred_2[i][0].item<float>());
//        if (DEGUBLIBTORCH) printf("%.2f\n", pred_2[i][0].item<float>());
//    }
//    printf("[Libtorch] Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);
    return predictions;
}

std::vector<float> PythonClassifier::classifyPointsBatch(
        const std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups) {
    std::vector<float> predictions;
    const int groups_num = point_groups.size(); // 点云数
    const int points_num = point_groups[0]->cols(); // 各点云点数
    printf("[DEBUG] points num:%d\n", points_num);
    printf("[DEBUG] groups num:%d\n", groups_num);

    double omp_timer_start = omp_get_wtime();
    std::vector<double> point_list;
    for (size_t i = 0; i < point_groups.size(); i++) {
        for (size_t j = 0; j < point_groups[i]->size(); j++) {
            point_list.push_back(*(point_groups[i]->data() + j));
//            cout << "data: " << *(point_groups[i]->data() + j) << endl;
        }
    }

    npy_intp Dims[3] = {groups_num, points_num, 3}; //给定维度信息

    PyObject *PyArray  = PyArray_SimpleNewFromData(3, Dims, NPY_DOUBLE, point_list.data()); //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    PyTuple_SetItem(ArgArray, 0, PyArray);

    printf("[Python] Inputs generate runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_start);

    double omp_timer_predict = omp_get_wtime();
    PyObject *pFunc = PyDict_GetItemString(pDict, "PassArrayFromCToPython");
    PyObject_CallObject(pFunc, ArgArray);//调用函数，传入Numpy Array 对象

    printf("[Python] Predict runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_predict);

//    for(int i = 0; i < pred_2.size(0); i++) {
//        predictions.push_back(pred_2[i][0].item<float>());
//        if (DEGUBLIBTORCH) printf("%.2f\n", pred_2[i][0].item<float>());
//    }
    printf("[Python] Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_start);
    return predictions;
}

}  // namespace net
}  // namespace gpd
