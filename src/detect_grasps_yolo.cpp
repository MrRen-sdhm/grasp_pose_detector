#include <string>

#include <gpd/grasp_detector_pointnet.h>
#include <gpd/detector/yolo_detector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

            // Get the names of the output layers
            std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
            {
                static std::vector<cv::String> names;
                if (names.empty())
                {
                    //Get the indices of the output layers, i.e. the layers with unconnected outputs
                    std::vector<int> outLayers = net.getUnconnectedOutLayers();

                    //get the names of all the layers in the network
                    std::vector<cv::String> layersNames = net.getLayerNames();

                    // Get the names of the output layers in names
                    names.resize(outLayers.size());
                    for (size_t i = 0; i < outLayers.size(); ++i)
                        names[i] = layersNames[outLayers[i] - 1];
                }
                return names;
            }

            // Draw the predicted bounding box
            void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string> classes)
            {
                //Draw a rectangle displaying the bounding box
                rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 2);

                //Get the label for the class name and its confidence
                std::string label = cv::format("%.2f", conf);
                if (!classes.empty())
                {
                    CV_Assert(classId < (int)(classes.size()));
                    label = classes[classId] + ":" + label;
                }

                //Display the label at the top of the bounding box
                int baseLine;
                cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                top = std::max(top, labelSize.height);
                rectangle(frame, cv::Point(left-1, top - round(1.2*labelSize.height)),
                        cv::Point(left + round(1.2*labelSize.width), top + baseLine), cv::Scalar(255, 178, 50), cv::FILLED);
                putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0),1,CV_AA);

                rectangle(frame, cv::Point((left+right)/2-2, (top+bottom)/2-2),
                          cv::Point((left+right)/2+2, (top+bottom)/2+2), cv::Scalar(0, 0, 255), cv::FILLED);
                rectangle(frame, cv::Point((left+right)/2 - (right-left)/6, (top+bottom)/2 - (bottom-top)/6),
                          cv::Point((left+right)/2 + (right-left)/6, (top+bottom)/2 + (bottom-top)/6), cv::Scalar(0, 255, 0));
            }

            // Remove the bounding boxes with low confidence using non-maxima suppression
            std::vector<cv::Rect> postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, float confThreshold,
                             float nmsThreshold, std::vector<std::string> classes)
            {
                std::vector<int> classIds;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                for (size_t i = 0; i < outs.size(); ++i)
                {
                    // Scan through all the bounding boxes output from the network and keep only the
                    // ones with high confidence scores. Assign the box's class label as the class
                    // with the highest score for the box.
                    float* data = (float*)outs[i].data;
                    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
                    {
                        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                        cv::Point classIdPoint;
                        double confidence;
                        // Get the value and location of the maximum score
                        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                        if (confidence > confThreshold)
                        {
                            int centerX = (int)(data[0] * frame.cols);
                            int centerY = (int)(data[1] * frame.rows);
                            int width = (int)(data[2] * frame.cols);
                            int height = (int)(data[3] * frame.rows);
                            int left = centerX - width / 2;
                            int top = centerY - height / 2;

                            classIds.push_back(classIdPoint.x);
                            confidences.push_back((float)confidence);
                            boxes.push_back(cv::Rect(left, top, width, height));
                        }
                    }
                }

                // Perform non maximum suppression to eliminate redundant overlapping boxes with
                // lower confidences
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

                int bottleNum = 0; // 瓶子个数
                std::vector<cv::Rect> obj_boxes;
                for (size_t i = 0; i < indices.size(); ++i)
                {
                    int idx = indices[i];
                    cv::Rect box = boxes[idx];
                    drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);

                    // 获取各瓶子位置
                    if (classes[classIds[idx]] == "bottle" && confidences[idx] > 0.4) {
                        std::cout << box << std::endl;
                        obj_boxes.push_back(box);
                        bottleNum ++;
                    }
                }

                return obj_boxes;
            }

            int TestYoloDetectorPointnet(int argc, char *argv[]) {
                printf("[INFO] detect grasps use pointnet...\n");

                // Read arguments from command line.
                if (argc < 2) {
                    std::cout << "Error: Not enough input arguments!\n\n";
                    std::cout << "Usage: detect_grasps CONFIG_FILE\n\n";
                    std::cout << "Detect grasp poses for a point cloud, PCD_FILE (*.pcd), "
                                 "using parameters from CONFIG_FILE (*.cfg).\n\n";
                    return (-1);
                }

                std::string config_filename = argv[1];
                if (!checkFileExists(config_filename)) {
                    printf("Error: config file not found!\n");
                    return (-1);
                }

                // Read parameters from configuration file.
                util::ConfigFile config_file(config_filename);
                config_file.ExtractKeys();

                // Set the camera position. Assumes a single camera view.
                std::vector<double> camera_position =
                        config_file.getValueOfKeyAsStdVectorDouble("camera_position",
                                                                   "0.0 0.0 0.0");
                Eigen::Matrix3Xd view_points(3, 1);
                view_points << camera_position[0], camera_position[1], camera_position[2];

                // Read yolo config and weights file path.
                std::string yolo_config_filename =
                        config_file.getValueOfKeyAsString("yolo_config_filename", "");
                std::string yolo_weights_filename =
                        config_file.getValueOfKeyAsString("yolo_weights_filename", "");

                // Read the rgb image file path.
                std::string rgb_image_filename =
                        config_file.getValueOfKeyAsString("rgb_image_filename", "");

                // Read the pcd file path.
                std::string pcd_filename =
                        config_file.getValueOfKeyAsString("pcd_filename", "");

                if (!checkFileExists(rgb_image_filename)) {
                    return (-1);
                }

                if (!checkFileExists(pcd_filename)) {
                    return (-1);
                }

                // Load point cloud from file.
                util::Cloud cloud(pcd_filename, view_points);
                if (cloud.getCloudOriginal()->size() == 0) {
                    std::cout << "Error: Input point cloud is empty or does not exist!\n";
                    return (-1);
                }

                // 读取图像
                cv::Mat image;
                image = cv::imread(rgb_image_filename);

                std::vector<cv::Rect> obj_boxes;
                detector::YoloDetector yoloDetector(yolo_config_filename, yolo_weights_filename);
                obj_boxes = yoloDetector.getObjRect(image);

                cv::imshow("image", image);
                cv::waitKey(0);

                GraspDetectorPointNet detector(config_filename);
                if (obj_boxes.size() > 0) {
                    for (size_t i = 0; i < obj_boxes.size(); i++) {

                        // Prepare the point cloud.
                        detector.preprocessPointCloud(cloud, obj_boxes[i]);

                        // Detect grasp poses.
                        detector.detectGrasps(cloud);
                    }
                } else {
                    printf("Don't have aim objects in the input file.\n");
                }

                printf("[INFO] detect grasps use lenet done.\n");

                return 0;
            }

        }  // namespace detect_grasps
    }  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
    return gpd::apps::detect_grasps::TestYoloDetectorPointnet(argc, argv);
}
