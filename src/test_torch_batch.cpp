#include <string>
#include <stdio.h>

#include <gpd/grasp_detector.h>
#include <gpd/descriptor/image_15_channels_strategy.h>
#include <torch/script.h>

namespace gpd {
    namespace apps {
        namespace detect_grasps {

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


                // TODO: 处理images并作为网络输入，images中为60*60*15的mat
                // Deserialize the ScriptModule from a file using torch::jit::load().
                std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(
                        "/home/sdhm/Projects/pytorch_cpp/test.pt");
                // Create a vector of torch inputs
                std::vector<at::Tensor> inputs_tuple;
                std::vector<torch::jit::IValue> input; // NOTE: 待删除

                for(size_t i = 0; i < images.size(); i++) {
//                for(size_t i = 0; i < 1; i++) {
                    // The channel dimension is the last dimension in OpenCV
                    at::Tensor tensor_image = torch::from_blob(images[i]->data,
                                                               {1, images[i]->rows, images[i]->cols, 15}, at::kByte);
                    tensor_image = tensor_image.to(at::kFloat);
                    cout << tensor_image.size(1) << " " << tensor_image.size(2) << " " << tensor_image.size(3) << endl; //60,60,15

                    // Transpose the image for [channels, rows, columns] format of pytorch tensor
//                    tensor_image = at::transpose(tensor_image, 1, 3);
//                    tensor_image = at::transpose(tensor_image, 2, 3);

//                    tensor_image = at::transpose(tensor_image, 1, 2);
//                    tensor_image = at::transpose(tensor_image, 1, 3);
                    cout << tensor_image.size(1) << " " << tensor_image.size(2) << " " << tensor_image.size(3) << endl; //15,60,60

//                tensor_image = tensor_image.permute({0,3,1,2});
//                tensor_image = tensor_image.toType(torch::kFloat);
                    tensor_image = tensor_image.div(255);
//                tensor_image = tensor_image.to(torch::kCUDA);

                    // 打印输出
//                    cout << "tensor_image" << tensor_image << endl;


                    inputs_tuple.emplace_back(tensor_image);
                    if (i == 0) input.emplace_back(tensor_image); // NOTE: 待删除
                }

                auto output_one = module->forward(input).toTensor(); // NOTE: 待删除
                cout << "[output_one]\n" << output_one << endl; // NOTE: 待删除

                // Execute the model and turn its output into a tensor.
                at::Tensor inputs = torch::cat(inputs_tuple, 0);
                auto output = module->forward({inputs}).toTensor();
                cout << "[output]\n" << output << endl;

//                cout << output[0][1] << endl;
                // 分离各输出
                auto output_1 = output.slice(1, 0, 1); // 输出1
                auto output_2 = output.slice(1, 1, 2); // 输出2
//                cout << "[output_1]\n" << output_1 << endl;
//                cout << "[output_2]\n" << output_2 << endl;
                // 转换为浮点型
                auto output_2_f = output_2.data<float>();
                printf("output_1_f: %f\n", output_2_f[1]);

//                auto max_result = output.max(1, true);
//                auto max_index_1 = std::get<1>(max_result).item<float>();
//                auto max_index_0 = std::get<0>(max_result).item<float>();
//                cout << max_index_1 << max_index_0 << endl;


                printf("hands: %zu\n", hands.size());
                printf("images: %zu\n", images.size());

                // 单独显示各个抓取姿态以及点云
                std::vector<int> labels = detector.evalGroundTruth(mesh, hands);
                for(size_t i = 0; i < labels.size(); i++) {
                    printf("%d ", labels[i]);
                }
                printf("\n");
                for(int i = 0; i < hands.size(); i++) {
                    if (! hands[i]->isFullAntipodal()) {
//                    if (1) {
                        printf("%d ", i);
                    }
                }

                return 0;
            }

        }  // namespace detect_grasps
    }  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
    return gpd::apps::detect_grasps::DoMain(argc, argv);
}
