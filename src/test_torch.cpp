#include <string>
#include <stdio.h>
#include <time.h>

#include <gpd/grasp_detector.h>
#include <gpd/descriptor/image_15_channels_strategy.h>
#include <gpd/net/libtorch_classifier.h>
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>
#include <opencv2/opencv.hpp>

#include <boost/timer.hpp>


using namespace cv;

namespace gpd {
    namespace apps {
        namespace detect_grasps {

            struct Instance {
                std::unique_ptr<cv::Mat> image_;
                bool label_;

                Instance(std::unique_ptr<cv::Mat> image, bool label)
                        : image_(std::move(image)), label_(label) {}
            };

            bool checkFileExists(const std::string &file_name) {
                std::ifstream file;
                file.open(file_name.c_str());
                if (!file) {
                    std::cout << "File " + file_name + " could not be found!\n";
                    return false;
                }
                file.close();
                return true;
            }

            void showImage(const cv::Mat &image) {
                int border = 5;
                int n = 3;
                int image_size = 60;
                int total_size = n * (image_size + border) + border;

                cv::Mat image_out(total_size, total_size, CV_8UC3, cv::Scalar(0.5));
                std::vector<cv::Mat> channels;
                cv::split(image, channels);

                for (int i = 0; i < n; i++) {
                    // OpenCV requires images to be in BGR or grayscale to be displayed.
                    cv::Mat normals_rgb, depth_rgb, shadow_rgb;
                    std::vector<cv::Mat> normals_channels(3);
                    for (int j = 0; j < normals_channels.size(); j++) {
                        normals_channels[j] = channels[i * 5 + j];
                    }
                    cv::merge(normals_channels, normals_rgb);
                    // OpenCV requires images to be in BGR or grayscale to be displayed.
                    cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
                    cvtColor(channels[i * 5 + 3], depth_rgb, cv::COLOR_GRAY2RGB);
                    cvtColor(channels[i * 5 + 4], shadow_rgb, cv::COLOR_GRAY2RGB);
                    normals_rgb.copyTo(image_out(cv::Rect(
                            border, border + i * (border + image_size), image_size, image_size)));
                    depth_rgb.copyTo(image_out(cv::Rect(2 * border + image_size,
                                                        border + i * (border + image_size),
                                                        image_size, image_size)));
                    shadow_rgb.copyTo(image_out(cv::Rect(3 * border + 2 * image_size,
                                                         border + i * (border + image_size),
                                                         image_size, image_size)));
                }

                cv::namedWindow("Grasp Image (15 channels)", cv::WINDOW_NORMAL);
                cv::imshow("Grasp Image (15 channels)", image_out);
                cv::waitKey(0);
                cv::destroyWindow("Grasp Image (15 channels)");
            }


            int DoMain(int argc, char *argv[]) {
                // Read arguments from command line.
                if (argc < 4) {
                    std::cout << "Error: Not enough input arguments!\n\n";
                    std::cout << "Usage: label_grasps CONFIG_FILE PCD_FILE MESH_FILE\n\n";
                    std::cout << "Find grasp poses for a point cloud, PCD_FILE (*.pcd), "
                                 "using parameters from CONFIG_FILE (*.cfg), and check them "
                                 "against a mesh, MESH_FILE (*.pcd).\n\n";
                    return (-1);
                }

                std::string config_filename = argv[1];
                std::string pcd_filename = argv[2];
                std::string mesh_filename = argv[3];
                if (!checkFileExists(config_filename)) {
                    printf("Error: CONFIG_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(pcd_filename)) {
                    printf("Error: PCD_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(mesh_filename)) {
                    printf("Error: MESH_FILE not found!\n");
                    return (-1);
                }

                // Read parameters from configuration file.
                const double VOXEL_SIZE = 0.003;
                util::ConfigFile config_file(config_filename);
                config_file.ExtractKeys();
                std::vector<double> workspace =
                        config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
                int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
                int num_samples = config_file.getValueOfKey<int>("num_samples", 30);
                bool sample_above_plane =
                        config_file.getValueOfKey<int>("sample_above_plane", 1);
                printf("num_threads: %d, num_samples: %d\n", num_threads, num_samples);

                // View point from which the camera sees the point cloud.
                Eigen::Matrix3Xd view_points(3, 1);
                view_points.setZero();

                // Load point cloud from file.
                util::Cloud cloud(pcd_filename, view_points);
                if (cloud.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Input point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Load point cloud from file.
                util::Cloud mesh(mesh_filename, view_points);
                if (mesh.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Mesh point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Prepare the point cloud.
                cloud.filterWorkspace(workspace);
                cloud.voxelizeCloud(VOXEL_SIZE);
                cloud.calculateNormals(num_threads);
                cloud.setNormals(cloud.getNormals() * (-1.0));  // NOTE: do not do this! 翻转单视角点云表面法线（坐标系在物体内部时使用）
                if (sample_above_plane) {
                    cloud.sampleAbovePlane();
                }
                cloud.subsample(num_samples);

                // Prepare the mesh.
                mesh.calculateNormals(num_threads);
                mesh.setNormals(mesh.getNormals() * (-1.0)); // NOTE: 翻转 ground truth 的表面法线

                // Detect grasp poses.
                std::vector<std::unique_ptr<candidate::Hand>> hands;
                std::vector<std::unique_ptr<cv::Mat>> images;
                GraspDetector detector(config_filename);
                detector.createGraspImages(cloud, hands, images);

//                cv::Mat image(*images[0]);
//                cout << image.size() << endl;

                // TODO: 处理images并作为网络输入，images中为60*60*15的mat
//                cv::Mat image_bgr, image;
//                image_bgr = cv::imread("/home/sdhm/图片/GraspImage_krylon.jpg");
//                cvtColor(image_bgr, image, cv::COLOR_BGR2RGB);
//                resize(image, image, cv::Size(1920, 1080));

//                for (int j=0;j<10;j++)
//                {
//                    cout<<image.at<cv::Vec3b>(0,j)<<endl;
//                }

                // The channel dimension is the last dimension in OpenCV
                at::Tensor tensor_image = torch::from_blob(images[0]->data, {1, images[0]->rows, images[0]->cols, 15}, at::kByte);
                tensor_image = tensor_image.to(at::kFloat);
                cout << tensor_image.size(1) << " " << tensor_image.size(2) << " " << tensor_image.size(3) << endl; //15,60,60

                // Transpose the image for [channels, rows, columns] format of pytorch tensor
                tensor_image = at::transpose(tensor_image, 1, 3);
                tensor_image = at::transpose(tensor_image, 2, 3);
                cout << tensor_image.size(1) << " " << tensor_image.size(2) << " " << tensor_image.size(3) << endl; //15,60,60


//                tensor_image = tensor_image.permute({0,3,1,2});
//                tensor_image = tensor_image.toType(torch::kFloat);
                tensor_image = tensor_image.div(256);
//                tensor_image = tensor_image.to(torch::kCUDA);

                // 打印输出
//                cout << "tensor_image" << tensor_image << endl;

                // Create a vector of torch inputs
                std::vector<torch::jit::IValue> input;
                input.emplace_back(tensor_image);

                // Deserialize the ScriptModule from a file using torch::jit::load().
                std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("/home/sdhm/Projects/pytorch_cpp/test.pt");
                // Execute the model and turn its output into a tensor.
//                auto output = module->forward(input).toTensor().clone().squeeze(0);

                auto output = module->forward(input).toTensor();
                cout << "[output]\n" << output << endl;

//                cout << output[0][1] << endl;
                // 分离各输出
                auto output_1 = output.slice(1, 0, 1); // 输出1
                auto output_2 = output.slice(1, 1, 2); // 输出2
                cout << "[output_1]\n" << output_1 << endl;
                cout << "[output_2]\n" << output_2 << endl;
                // 转换为浮点型
                auto output_1_f = output_1.data<float>();
                printf("output_1_f: %f\n", *output_1_f);

                cout << output_1_f << endl;

                auto max_result = output.max(1, true);
                auto max_index_1 = std::get<1>(max_result).item<float>();
                auto max_index_0 = std::get<0>(max_result).item<float>();

//                cout << max_index_1 << max_index_0 << endl;


                printf("hands: %zu\n", hands.size());
                printf("images: %zu\n", images.size());

                // 单独显示各个抓取姿态以及点云
//                for(int i = 0; i < hands.size(); i++) {
////                    if (! hands[i]->isFullAntipodal()) {
//                    if (1) {
//                        showImage(*images[i]); // 显示多通道图像
//                    }
//                }

                return 0;
            }

            void addInstances(
                    const std::vector<std::unique_ptr<candidate::Hand>> &grasps,
                    std::vector<std::unique_ptr<cv::Mat>> &images,
                    const std::vector<int> &positives, const std::vector<int> &negatives,
                    std::vector<Instance> &dataset) {
                for (int k = 0; k < positives.size(); k++) {
                    int idx = positives[k];
                    if (!images[idx]) {
                        printf(" => idx: %d is nullptr!\n", idx);
                        char c;
                        std::cin >> c;
                    }
                    dataset.push_back(
                            Instance(std::move(images[idx]), grasps[idx]->isFullAntipodal()));
                }

                for (int k = 0; k < negatives.size(); k++) {
                    int idx = negatives[k];
                    if (!images[idx]) {
                        printf(" => idx: %d is nullptr!\n", idx);
                        char c;
                        std::cin >> c;
                    }
                    dataset.push_back(
                            Instance(std::move(images[idx]), grasps[idx]->isFullAntipodal()));
                }
            }

            void copyMatrix(const cv::Mat &src, cv::Mat &dst, int idx_in,
                                           int *dims_img) {
                const int rows = dims_img[0];
                const int cols = dims_img[1];
                const int channels = dims_img[2];
                for (int j = 0; j < rows; j++) {
                    for (int k = 0; k < cols; k++) {
                        for (int l = 0; l < channels; l++) {
                            int idx_dst[4] = {idx_in, j, k, l};
                            dst.at<uchar>(idx_dst) = src.ptr<uchar>(j)[k * channels + l];
                        }
                    }
                }
            }

            void createDatasetsHDF5(const std::string &filepath,
                                                   int num_data) {
                printf("Opening HDF5 file at: %s\n", filepath.c_str());
                remove(filepath.c_str()); // 删除已存在文件
                cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(filepath);

                const int n_dims_labels = 2;
                int dsdims_labels[n_dims_labels] = {num_data, 1};
                int chunks_labels[n_dims_labels] = {1, dsdims_labels[1]};
                printf("Creating dataset <labels>: %d x %d\n", dsdims_labels[0],
                       dsdims_labels[1]);
                h5io->dscreate(n_dims_labels, dsdims_labels, CV_8UC1, "labels", 4,
                               chunks_labels);

                const int n_dims_images = 4;
                int dsdims_images[n_dims_images] = {
                        num_data, 60, 60, 15}; // NOTE 修改
                int chunks_images[n_dims_images] = {1, dsdims_images[1],
                                                    dsdims_images[2], dsdims_images[3]};
                h5io->dscreate(n_dims_images, dsdims_images, CV_8UC1, "images",
                               n_dims_images, chunks_images);

                h5io->close();
            }

            int insertIntoHDF5(const std::string &file_path,
                                              const std::vector<Instance> &dataset,
                                              int offset) {
                if (dataset.empty()) {
                    printf("Error: Dataset is empty!\n");
                    return offset;
                }

                printf("Storing %d items in HDF5: %s ... \n", (int)dataset.size(),
                       file_path.c_str());

                cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(file_path);
                printf("  Opened HDF5 file\n");

                const int num = static_cast<int>(dataset.size());
                const int rows = dataset[0].image_->rows;
                const int cols = dataset[0].image_->cols;
                const int channels = dataset[0].image_->channels();

                cv::Mat labels(dataset.size(), 1, CV_8UC1, cv::Scalar(0.0));

                for (int i = 0; i < dataset.size(); i++) {
                    labels.at<uchar>(i) = (uchar)dataset[i].label_;
                }

                const int dims_images = 4;
                int dsdims_images[dims_images] = {num, rows, cols, channels};
                cv::Mat images(dims_images, dsdims_images, CV_8UC1, cv::Scalar(0.0));
                const int dims_image = 3;
                int dsdims_image[dims_image] = {rows, cols, channels};

                for (int i = 0; i < dataset.size(); i++) {
                    if (!dataset[i].image_) {
                        printf("FATAL ERROR! %d is nullptr\n", i);
                        char c;
                        std::cin >> c;
                    }
                    copyMatrix(*dataset[i].image_, images, i, dsdims_image);
                }

                printf("  Inserting into images dataset ...\n");
                const int dims_offset_images = 4;
                int offsets_images[dims_offset_images] = {offset, 0, 0, 0};
                h5io->dsinsert(images, "images", offsets_images);

                printf("  Inserting into labels dataset ...\n");
                const int dims_offset_labels = 2;
                int offsets_labels[dims_offset_labels] = {offset, 0};
                h5io->dsinsert(labels, "labels", offsets_labels);

                h5io->close();

                return offset + static_cast<int>(images.size[0]);
            }

            // NOTE 将一批图像写入hdf5
            int DoTest(int argc, char *argv[]) {
                // Read arguments from command line.
                if (argc < 4) {
                    std::cout << "Error: Not enough input arguments!\n\n";
                    std::cout << "Usage: label_grasps CONFIG_FILE PCD_FILE MESH_FILE\n\n";
                    std::cout << "Find grasp poses for a point cloud, PCD_FILE (*.pcd), "
                                 "using parameters from CONFIG_FILE (*.cfg), and check them "
                                 "against a mesh, MESH_FILE (*.pcd).\n\n";
                    return (-1);
                }

                std::string config_filename = argv[1];
                std::string pcd_filename = argv[2];
                std::string mesh_filename = argv[3];
                if (!checkFileExists(config_filename)) {
                    printf("Error: CONFIG_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(pcd_filename)) {
                    printf("Error: PCD_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(mesh_filename)) {
                    printf("Error: MESH_FILE not found!\n");
                    return (-1);
                }

                // Read parameters from configuration file.
                const double VOXEL_SIZE = 0.003;
                util::ConfigFile config_file(config_filename);
                config_file.ExtractKeys();
                std::vector<double> workspace =
                        config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
                int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
                int num_samples = config_file.getValueOfKey<int>("num_samples", 30);
                bool sample_above_plane =
                        config_file.getValueOfKey<int>("sample_above_plane", 1);
                printf("num_threads: %d, num_samples: %d\n", num_threads, num_samples);

                // View point from which the camera sees the point cloud.
                Eigen::Matrix3Xd view_points(3, 1);
                view_points.setZero();

                // Load point cloud from file.
                util::Cloud cloud(pcd_filename, view_points);
                if (cloud.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Input point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Load point cloud from file.
                util::Cloud mesh(mesh_filename, view_points);
                if (mesh.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Mesh point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Prepare the point cloud.
                cloud.filterWorkspace(workspace);
                cloud.voxelizeCloud(VOXEL_SIZE);
                cloud.calculateNormals(num_threads);
                cloud.setNormals(cloud.getNormals() * (-1.0));  // NOTE: do not do this! 翻转单视角点云表面法线（坐标系在物体内部时使用）
                if (sample_above_plane) {
                    cloud.sampleAbovePlane();
                }
                cloud.subsample(num_samples);

                // Prepare the mesh.
                mesh.calculateNormals(num_threads);
                mesh.setNormals(mesh.getNormals() * (-1.0)); // NOTE: 翻转 ground truth 的表面法线

                // Detect grasp poses.
                std::vector<std::unique_ptr<candidate::Hand>> hands;
                std::vector<std::unique_ptr<cv::Mat>> images;
                GraspDetector detector(config_filename);
                detector.createGraspImages(cloud, hands, images);

                std::vector<int> labels = detector.evalGroundTruth(mesh, hands);
//                for(size_t i = 0; i < labels.size(); i++) {
//                    printf("%d ", labels[i]);
//                }
//                printf("\n");

                /// *************处理images并作为网络输入，images中为60*60*15的mat ************
                const int channels = 15;
                const bool use_cuda = true;
                std:: string module_path = "/home/sdhm/Projects/pytorch_cpp/test.pt";
                if (use_cuda) module_path = "/home/sdhm/Projects/pytorch_cpp/cuda.pt";

                // Deserialize the ScriptModule from a file using torch::jit::load().
                boost::timer timer_load;
                double omp_timer_load = omp_get_wtime();
                std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(module_path);
//                cout << "\nLoad module runtime(boost):" << timer_load.elapsed() << "s" << endl;
                printf("Load module runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_load);

                // Move the module to cuda.
                boost::timer timer;
                clock_t start = clock();
                double omp_timer = omp_get_wtime();
                if (use_cuda) module->to(at::kCUDA); // TODO: CUDA
//                cout << "Module to cuda runtime(boost):" << timer.elapsed() << "s" << endl;
//                cout << "Module to cuda runtime(clock):" << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
                printf("Module to cuda runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer);

                // Create a vector of torch inputs.
                std::vector<at::Tensor> inputs_tuple;

                clock_t start_loop = clock();
                boost::timer timer_loop_start;
                double omp_timer_loop = omp_get_wtime();
                printf("---------------channles:%d\n", images[0]->channels());
                for(size_t i = 0; i < images.size(); i++) {
                    // The channel dimension is the last dimension in OpenCV.
                    at::Tensor tensor_image = torch::from_blob(images[i]->data,
                            {1, images[i]->rows, images[i]->cols, channels}, at::kByte); // shape: {1,60,60,15}

                    tensor_image = tensor_image.to(at::kFloat);
                    tensor_image = tensor_image.div(256);
//                    cout << "tensor_image" << tensor_image << endl;

                    // Reshape the image for [channels, rows, columns] format of pytorch tensor
                    tensor_image = at::reshape(tensor_image,
                            {tensor_image.size(3), tensor_image.size(1), tensor_image.size(2)}); // shape: {15,60,60}

                    // Network's input shape is {1,15,60,60}, so add a dim.
                    tensor_image = tensor_image.unsqueeze(0); // shape: {1,15,60,60}
//                    cout << "tensor_image" << tensor_image << endl;

                    // Move the tensor to cuda.
                    if (use_cuda) tensor_image = tensor_image.to(torch::kCUDA); // TODO: CUDA

                    inputs_tuple.emplace_back(tensor_image);
                }
//                cout << "For loop runtime(boost):" << timer_loop_start.elapsed() << "s" << endl;
//                cout << "For loop runtime(clock):" << (double)(clock() - start_loop) / CLOCKS_PER_SEC << "s" << endl;
                printf("For loop runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);

                // Concatenate a batch of tensors.
                at::Tensor inputs = torch::cat(inputs_tuple, 0);

//                if (use_cuda)  inputs = inputs.cuda();

//                cout << "Inputs generate runtime(boost):" << timer_loop_start.elapsed() << "s" << endl;
//                cout << "Inputs generate runtime(clock):" << (double)(clock() - start_loop) / CLOCKS_PER_SEC << "s" << endl;
                printf("Inputs generate runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);

                // Execute the model and turn its output into a tensor.
                boost::timer timer_forward;
                clock_t start_forward = clock();
                double omp_timer_forward = omp_get_wtime();
                auto output = module->forward({inputs}).toTensor();
//                cout << "[output]\n" << output << endl;
//                cout << "Forward runtime(boost):" << timer_forward.elapsed() << "s" << endl;
//                cout << "Forward runtime(clock):" << (double)(clock() - start_forward) / CLOCKS_PER_SEC << "s" << endl;
                printf("Forward runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_forward);

//                cout << output[0][1] << endl;
                // 分离各输出
//                auto output_1 = output.slice(1, 0, 1); // 输出1
//                auto output_2 = output.slice(1, 1, 2); // 输出2
//                cout << "[output_1]\n" << output_1 << endl;
//                cout << "[output_2]\n" << output_2 << endl;

//                cout << "Total runtime(boost):" << timer.elapsed() << "s" << endl;
//                cout << "Total runtime(clock):" << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
                printf("Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer);

                /// **************************** 写入hdf5 ******************************
                printf("\nWrite data to hdf5...\n");
                std::string hdf5_name = "../test.h5";
                createDatasetsHDF5(hdf5_name, images.size());
                std::vector<Instance> test_data;
                for(size_t i = 0; i < images.size(); i++) {
                    test_data.push_back(
                            Instance(std::move(images[i]), hands[i]->isFullAntipodal()));
                }
                int offset = 0;
                offset = insertIntoHDF5(hdf5_name, test_data, offset);
                printf("offset:%d\n", offset);


                return 0;
            }

            int LibtorchTest(int argc, char *argv[]) {
                // Read arguments from command line.
                if (argc < 4) {
                    std::cout << "Error: Not enough input arguments!\n\n";
                    std::cout << "Usage: label_grasps CONFIG_FILE PCD_FILE MESH_FILE\n\n";
                    std::cout << "Find grasp poses for a point cloud, PCD_FILE (*.pcd), "
                                 "using parameters from CONFIG_FILE (*.cfg), and check them "
                                 "against a mesh, MESH_FILE (*.pcd).\n\n";
                    return (-1);
                }

                std::string config_filename = argv[1];
                std::string pcd_filename = argv[2];
                std::string mesh_filename = argv[3];
                if (!checkFileExists(config_filename)) {
                    printf("Error: CONFIG_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(pcd_filename)) {
                    printf("Error: PCD_FILE not found!\n");
                    return (-1);
                }
                if (!checkFileExists(mesh_filename)) {
                    printf("Error: MESH_FILE not found!\n");
                    return (-1);
                }

                // Read parameters from configuration file.
                const double VOXEL_SIZE = 0.003;
                util::ConfigFile config_file(config_filename);
                config_file.ExtractKeys();
                std::vector<double> workspace =
                        config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
                int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
                int num_samples = config_file.getValueOfKey<int>("num_samples", 30);
                bool sample_above_plane =
                        config_file.getValueOfKey<int>("sample_above_plane", 1);
                printf("num_threads: %d, num_samples: %d\n", num_threads, num_samples);

                std::string weights_file =
                        config_file.getValueOfKeyAsString("weights_file", "");
                int device = config_file.getValueOfKey<int>("device", 1);

                // View point from which the camera sees the point cloud.
                Eigen::Matrix3Xd view_points(3, 1);
                view_points.setZero();

                // Load point cloud from file.
                util::Cloud cloud(pcd_filename, view_points);
                if (cloud.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Input point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Load point cloud from file.
                util::Cloud mesh(mesh_filename, view_points);
                if (mesh.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Mesh point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // Prepare the point cloud.
                cloud.filterWorkspace(workspace);
                cloud.voxelizeCloud(VOXEL_SIZE);
                cloud.calculateNormals(num_threads);
                cloud.setNormals(cloud.getNormals() * (-1.0));  // NOTE: do not do this! 翻转单视角点云表面法线（坐标系在物体内部时使用）
                if (sample_above_plane) {
                    cloud.sampleAbovePlane();
                }
                cloud.subsample(num_samples);

                // Prepare the mesh.
                mesh.calculateNormals(num_threads);
                mesh.setNormals(mesh.getNormals() * (-1.0)); // NOTE: 翻转 ground truth 的表面法线

                // Detect grasp poses.
                std::vector<std::unique_ptr<candidate::Hand>> hands;
                std::vector<std::unique_ptr<cv::Mat>> images;
                GraspDetector detector(config_filename);
                detector.createGraspImages(cloud, hands, images);

                std::vector<int> labels = detector.evalGroundTruth(mesh, hands);
//                for(size_t i = 0; i < labels.size(); i++) {
//                    printf("%d ", labels[i]);
//                }
//                printf("\n");

                /// ************* 利用LibtorchClassifier进行打分 ************
                detector.detectGrasps(cloud);

                return 0;
            }

        }  // namespace detect_grasps
    }  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
//    return gpd::apps::detect_grasps::DoMain(argc, argv); // 测试单张图片
//    return gpd::apps::detect_grasps::DoTest(argc, argv); // 测试batch图片
    return gpd::apps::detect_grasps::LibtorchTest(argc, argv); // 测试LibtorchClassifier
}

/// 运行速度对比
/// num_samples = 200  num_orientations = 8  hand_axes = 1 2  Created 3200 images
/*
eigen_cpu:
Total runtime: 9.8501s

cpu
Created 3200 images in 2.1616s
Load module runtime(omp): 0.048851s
Module to cuda runtime(omp): 0.000000s
For loop runtime(omp): 0.588202s
Inputs generate runtime(omp): 0.878818s
Forward runtime(omp): 16.362420s
Total runtime(omp): 17.241315s

gpu
Created 3200 images in 1.5468s
Load module runtime(omp): 1.151793s
Module to cuda runtime(omp): 0.000017s
For loop runtime(omp): 0.376890s
Inputs generate runtime(omp): 0.380632s
Forward runtime(omp): 0.877071s
Total runtime(omp): 1.257801s

*/

/// num_samples = 100  num_orientations = 4  hand_axes = 1 2  Created 800 images
/*
eigen_cpu:
Total runtime: 2.4604s

cpu
Created 800 images in 0.4733s
Load module runtime(omp): 0.041543s
Module to cuda runtime(omp): 0.000000s
For loop runtime(omp): 0.083738s
Inputs generate runtime(omp): 0.137862s
Forward runtime(omp): 3.774969s
Total runtime(omp): 3.912866s

gpu
Created 800 images in 0.4678s
Load module runtime(omp): 0.957725s
Module to cuda runtime(omp): 0.000016s
For loop runtime(omp): 0.108117s
Inputs generate runtime(omp): 0.109781s
Forward runtime(omp): 0.721321s
Total runtime(omp): 0.831143s

*/