//
// Created by sdhm on 5/17/19.
//

#include "gpd/detector/yolo_detector.h"

namespace gpd {
    namespace detector {
        YoloDetector::YoloDetector(std::string config_file, std::string weights_file) {
            net_ = cv::dnn::readNetFromDarknet(config_file, weights_file);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        std::vector<cv::Rect> YoloDetector::getObjRect(cv::Mat image) {
            cv::Mat blob;
            std::vector<cv::Rect> obj_boxes;
            cv::dnn::blobFromImage(image, blob, 1/255.0, cvSize(inpWidth_, inpHeight_), cv::Scalar(0,0,0), true, false); // Create a 4D blob from a frame.
            net_.setInput(blob); // Sets the input to the network
            std::vector<cv::Mat> outs;
            net_.forward(outs, getOutputsNames(net_)); // Runs the forward pass to get output of the output layers
            obj_boxes = postprocess(image, outs, confThreshold_, nmsThreshold_, get_classes_vec()); // Remove the bounding boxes with low confidence
            return obj_boxes;
        }

        // Remove the bounding boxes with low confidence using non-maxima suppression
        std::vector<cv::Rect> YoloDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, float confThreshold,
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

        // Draw the predicted bounding box
        void YoloDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string> classes)
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

        // Get the names of the output layers
        std::vector<cv::String> YoloDetector::getOutputsNames(const cv::dnn::Net& net)
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

    }
}