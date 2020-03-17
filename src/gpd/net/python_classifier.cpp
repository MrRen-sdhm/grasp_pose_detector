#include <gpd/net/python_classifier.h>

string get_dirname(const string& path) {
    char *str = (char*)path.c_str();
    char *ret = strrchr(str, '/');

    int length = strlen(str) - strlen(ret);
    string dirname = path.substr(0, length);
    return dirname;
}

string get_basename(const string& path) {
    char *str = (char*)path.c_str();
    char *ret = strrchr(str, '/');

    char *basename_c = ret+1;

    string basename(basename_c);
    return basename;
}

string get_prefix(const string& name) {
    char *str = (char*)name.c_str();
    char *ret = strrchr(str, '.');

    int length = strlen(str) - strlen(ret);
    string prefix = name.substr(0, length);
    return prefix;
}

namespace gpd {
namespace net {

PythonClassifier::PythonClassifier(const std::string &model_file, const std::string &weights_file,
                                        Classifier::Device device, int batch_size) : batch_size_(batch_size) {
    // Init python
    init_ar();
    char str[] = "Python";
    Py_SetProgramName(str);
    if(!Py_IsInitialized())
        cout << "Python init faild/n" << endl;

    // get model dirname
    string dirname = get_dirname(model_file);
    PyRun_SimpleString("import sys");
    string path_append_cmd = "sys.path.append('" + dirname + "')";
    PyRun_SimpleString(path_append_cmd.c_str()); //example: sys.path.append('../python/pointnet')

    // get module name
    string basename = get_basename(model_file);
    module_name_ = get_prefix(basename); //example: pointnet_cls
    func_name_ = "classify_pcs";

    printf("[DEBUG] dirname: %s base_name: %s module_name: %s\n", dirname.c_str(), basename.c_str(), module_name_.c_str());

    /* import */
    pModule = PyImport_ImportModule(module_name_.c_str());
    if (pModule == NULL) {
        cout << "[Python] ERROR importing module" << endl;
        PyErr_Print();
    }

    pDict = PyModule_GetDict(pModule);

    // load weight
    PyObject *ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    PyTuple_SetItem(ArgArray, 0, Py_BuildValue("s", weights_file.c_str()));

    PyObject *pFunc = PyDict_GetItemString(pDict, "load_weight");
    PyObject_CallObject(pFunc, ArgArray); //调用函数

    Py_DECREF(pFunc);
    Py_DECREF(ArgArray);
}

std::vector<double> PythonClassifier::classifyImages(
    const std::vector<std::unique_ptr<cv::Mat>> &image_list) {
}

std::vector<double> PythonClassifier::classifyPoints(
        const std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups) {
    std::vector<double> predictions;

    const int groups_num = point_groups.size(); //点云数
    const int points_num = point_groups[0]->cols(); //各点云点数
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

    if(!gil_init) { // 确保GIL锁已被创建
        PyEval_InitThreads();
        PyEval_SaveThread();
        gil_init = true;
    }

    // 获得GIL锁
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    npy_intp Dims[3] = {groups_num, points_num, 3}; //给定维度信息

    PyObject *PyArray  = PyArray_SimpleNewFromData(3, Dims, NPY_DOUBLE, point_list.data()); //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    PyTuple_SetItem(ArgArray, 0, PyArray);

    printf("[Python] Inputs generate runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_start);

    double omp_timer_predict = omp_get_wtime();
    PyObject *pFunc = PyDict_GetItemString(pDict, func_name_.c_str());
    PyObject *FuncBack = PyObject_CallObject(pFunc, ArgArray); //调用函数，传入Numpy Array 对象


    printf("[Python] Predict runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_predict);

    // 获取返回值
    if (PyList_Check(FuncBack)) { //检查是否为List对象
        int SizeOfList = PyList_Size(FuncBack); //List对象的大小，这里SizeOfList = 3
        cout << "Size of return List:" << SizeOfList <<endl;
        for (int i = 0; i < SizeOfList; i++) {
            PyObject *ListItem = PyList_GetItem(FuncBack, i); //获取List对象中的每一个元素
            const double value = PyFloat_AsDouble(ListItem);
            predictions.push_back(value);
            cout << value << " "; //输出元素
            Py_DECREF(ListItem); //释放空间
        }
        cout << endl;

    } else {
        cout << "[Python] Function return is not a List" << endl;
    }

//    Py_DECREF(PyArray);
//    Py_DECREF(ArgArray);
//    Py_DECREF(pFunc);
//    Py_DECREF(FuncBack);

    // 释放GIL锁
    PyGILState_Release(gstate);

    printf("[Python] Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_start);
    return predictions;
}

std::vector<double> PythonClassifier::classifyPointsBatch(
        const std::vector<std::unique_ptr<Eigen::Matrix3Xd>> &point_groups) {
    std::vector<double> predictions;

    const int groups_num = point_groups.size(); //点云数
    const int points_num = point_groups[0]->cols(); //各点云点数
//    printf("[DEBUG] points num:%d\n", points_num);
//    printf("[DEBUG] groups num:%d\n", groups_num);

    double omp_timer_start = omp_get_wtime();
    std::vector<double> point_list;

    size_t batch_num =  point_groups.size()/batch_size_ + 1;
//    printf("[DEBUG] batch_size:%d batch_num: %zu\n", batch_size_, batch_num);

    for (size_t batch = 0; batch < batch_num; batch++) { // 最后一次循环为不完整的batch
        point_list.clear(); // Clear the vector of input points.

        int groups_num_curr_batch = 0;
        for (size_t i = batch * batch_size_; i < (batch + 1) * batch_size_; i++) {
            if (i > point_groups.size() - 1) break; // 已处理完所有数据, 退出
//            printf("[Python] Processing group num: %zu\n", i);

            /// 将所有点的x/y/z值依次写入point_list [x1,y1,z1,x2,y2,z2...xn,yn,zn]
            for (size_t j = 0; j < point_groups[i]->size(); j++) {
                point_list.push_back(*(point_groups[i]->data() + j));
//                cout << "data: " << *(point_groups[i]->data() + j) << endl;
            }
            groups_num_curr_batch ++; // 已写入点云计数
        }

//        printf("[Python] Groups num curr batch:%d\n", groups_num_curr_batch);
        if (groups_num_curr_batch == 0) continue;


        if(!gil_init) { // 确保GIL锁已被创建
            PyEval_InitThreads();
            PyEval_SaveThread();
            gil_init = true;
        }

        // 获得GIL锁
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        /// 调用Python脚本处理point_list
        npy_intp Dims[3] = {groups_num_curr_batch, points_num, 3}; //给定维度信息
        PyObject *PyArray  = PyArray_SimpleNewFromData(3, Dims, NPY_DOUBLE, point_list.data()); //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
        PyObject *ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
        PyTuple_SetItem(ArgArray, 0, PyArray);

        PyObject *pFunc = PyDict_GetItemString(pDict, func_name_.c_str());
        PyObject *FuncBack = PyObject_CallObject(pFunc, ArgArray); //调用函数，传入Numpy Array 对象

        // 获取返回值
        if (PyList_Check(FuncBack)) { //检查是否为List对象
            int SizeOfList = PyList_Size(FuncBack); //List对象的大小，这里SizeOfList = 3
//            cout << "Size of return List:" << SizeOfList << endl;
            for (int i = 0; i < SizeOfList; i++) {
                PyObject *ListItem = PyList_GetItem(FuncBack, i); //获取List对象中的每一个元素
                const double value = PyFloat_AsDouble(ListItem);
                predictions.push_back(value); // 存储各预测值
                Py_DECREF(ListItem); //释放空间
//                cout << value << " "; //输出元素
            }
//            cout << endl;
        } else {
            cout << "[ERROR] Function return is not a List" << endl;
        }

//        WARN: don't do that!
//        Py_DECREF(PyArray);
//        Py_DECREF(ArgArray);

        // 释放GIL锁
        PyGILState_Release(gstate);
    }

    printf("[Python] Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_start);
    return predictions;
}

}  // namespace net
}  // namespace gpd
