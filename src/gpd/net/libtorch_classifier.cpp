#include <gpd/net/libtorch_classifier.h>

namespace gpd {
    namespace net {

        LibtorchClassifier::LibtorchClassifier(const std::string &model_file,
                                         const std::string &weights_file,
                                         Classifier::Device device, int batch_size) {

            // Load pretrained network.
            // Deserialize the ScriptModule from a file using torch::jit::load().
            double omp_timer_load = omp_get_wtime();
            try {
                module_ = torch::jit::load(weights_file);
            }
            catch (std::exception &e) {
                std::cerr << "Couldn't load weights file, please check the weights_file path!" << std::endl;
                exit(1);
            }

            printf("[Libtorch] Load module runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_load);

            // Initialize Libtorch.
            switch (device) {
                case Classifier::Device::eCPU:
                    use_cuda_ = false;
                    break;
                default:
                    use_cuda_ = true;
                    module_->to(at::kCUDA); // Move the module to cuda.
            }
        }

        std::vector<float> LibtorchClassifier::classifyImages(
                const std::vector<std::unique_ptr<cv::Mat>> &image_list) {
            const int channels = image_list[0]->channels();
            std::vector<float> predictions;

            // Create a vector of torch inputs.
            std::vector<at::Tensor> inputs_tuple;

            double omp_timer_loop = omp_get_wtime();
            for(size_t i = 0; i < image_list.size(); i++) {
                // The channel dimension is the last dimension in OpenCV.
                at::Tensor tensor_image = torch::from_blob(image_list[i]->data,
                        {1, image_list[i]->rows,image_list[i]->cols, channels}, at::kByte); // shape: {1,60,60,15}

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
                if (use_cuda_) tensor_image = tensor_image.to(torch::kCUDA);

                inputs_tuple.emplace_back(tensor_image);
            }
//            printf("[Libtorch] For loop runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);

            // Concatenate a batch of tensors.
            at::Tensor inputs = torch::cat(inputs_tuple, 0);
            printf("[Libtorch] Inputs generate runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);

            // Execute the model and turn its output into a tensor.
            double omp_timer_forward = omp_get_wtime();
            auto output = module_->forward({inputs}).toTensor();

//            std::cout << output << std::endl; // 输出
            printf("[Libtorch] Forward runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_forward);

            at::Tensor pred = at::softmax(output, 1); // softmax
//            std::cout << "[prediction]\n" << pred << std::endl;

            // 分离各输出
//            auto output_1 = output.slice(1, 0, 1); // 输出1
            auto output_2 = output.slice(1, 1, 2); // 输出2
//            std::cout << "[output_1]\n" << output_1 << std::endl;
//            std::cout << "[output_2]\n" << output_2 << std::endl;

            // 分离各预测
//            auto pred_1 = pred.slice(1, 0, 1); // 预测1
            auto pred_2 = pred.slice(1, 1, 2); // 预测2
//            std::cout << "[pred_1]\n" << pred_1 << std::endl;
//            std::cout << "[pred_2]\n" << pred_2 << std::endl;

//            printf("otput2_size:%d\n", (int)output_2.size(0));
//            for(int i = 0; i < output_2.size(0); i++) {
//                predictions.push_back(output_2[i][0].item<float>());
////                std::cout << output_2[i][0].item<float>() << std::endl;
//            }

//            printf("pred_2_size:%d\n", (int)pred_2.size(0));
            for(int i = 0; i < pred_2.size(0); i++) {
                predictions.push_back(pred_2[i][0].item<float>());
//                std::cout << pred_2[i][0].item<float>() << std::endl;
            }

            printf("[Libtorch] Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);
            return predictions;
        }

        std::vector<float> LibtorchClassifier::classifyPoints(
                const std::vector<std::unique_ptr<cv::Mat>> &point_list) {
            const int channels = point_list[0]->channels();
            std::vector<float> predictions;

            // Create a vector of torch inputs.
            std::vector<at::Tensor> inputs_tuple;

            double omp_timer_loop = omp_get_wtime();
            for(size_t i = 0; i < point_list.size(); i++) {
                // The channel dimension is the last dimension in OpenCV.
                at::Tensor tensor_image = torch::from_blob(point_list[i]->data,
                                                           {1, point_list[i]->rows,point_list[i]->cols, channels}, at::kByte); // shape: {1,60,60,15}

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
                if (use_cuda_) tensor_image = tensor_image.to(torch::kCUDA);

                inputs_tuple.emplace_back(tensor_image);
            }
//            printf("[Libtorch] For loop runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);

            // Concatenate a batch of tensors.
            at::Tensor inputs = torch::cat(inputs_tuple, 0);
            printf("[Libtorch] Inputs generate runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);

            // Execute the model and turn its output into a tensor.
            double omp_timer_forward = omp_get_wtime();
            auto output = module_->forward({inputs}).toTensor();

//            std::cout << output << std::endl; // 输出
            printf("[Libtorch] Forward runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_forward);

            at::Tensor pred = at::softmax(output, 1); // softmax
//            std::cout << "[prediction]\n" << pred << std::endl;

            // 分离各输出
//            auto output_1 = output.slice(1, 0, 1); // 输出1
            auto output_2 = output.slice(1, 1, 2); // 输出2
//            std::cout << "[output_1]\n" << output_1 << std::endl;
//            std::cout << "[output_2]\n" << output_2 << std::endl;

            // 分离各预测
//            auto pred_1 = pred.slice(1, 0, 1); // 预测1
            auto pred_2 = pred.slice(1, 1, 2); // 预测2
//            std::cout << "[pred_1]\n" << pred_1 << std::endl;
//            std::cout << "[pred_2]\n" << pred_2 << std::endl;

//            printf("otput2_size:%d\n", (int)output_2.size(0));
//            for(int i = 0; i < output_2.size(0); i++) {
//                predictions.push_back(output_2[i][0].item<float>());
////                std::cout << output_2[i][0].item<float>() << std::endl;
//            }

//            printf("pred_2_size:%d\n", (int)pred_2.size(0));
            for(int i = 0; i < pred_2.size(0); i++) {
                predictions.push_back(pred_2[i][0].item<float>());
//                std::cout << pred_2[i][0].item<float>() << std::endl;
            }

            printf("[Libtorch] Total runtime(omp): %3.6fs\n", omp_get_wtime() - omp_timer_loop);
            return predictions;
        }

    }  // namespace net
}  // namespace gpd
