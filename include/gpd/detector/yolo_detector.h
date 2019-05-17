//
// Created by sdhm on 5/17/19.
//
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifndef GPD_YOLO_DETECTOR_H
#define GPD_YOLO_DETECTOR_H

namespace gpd {
    namespace detector {
        class YoloDetector {
        public:
            YoloDetector(std::string config_file, std::string weights_file);

            // Get the bounding boxes of objects
            std::vector<cv::Rect> getObjRect(cv::Mat image);

        private:
            // Remove the bounding boxes with low confidence using non-maxima suppression
            std::vector<cv::Rect> postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, float confThreshold,
                                              float nmsThreshold, std::vector<std::string> classes);

            // Draw the predicted bounding box
            void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame,
                    std::vector<std::string> classes);

            // Get the names of the output layers
            std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);


            // yolo相关参数
            cv::dnn::Net net_;
            float confThreshold_ = 0.5; // Confidence threshold
            float nmsThreshold_ = 0.4;  // Non-maximum suppression threshold
            int inpWidth_ = 416;  // Width of network's input image 416
            int inpHeight_ = 416; // Height of network's input image 416
            std::string classes_[80]={"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
                                     "parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
                                     "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
                                     "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
                                     "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
                                     "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
                                     "scissors","teddy bear","hair drier","toothbrush"};

            std::vector<std::string> get_classes_vec() {
                std::vector<std::string> classes_vec(classes_, classes_+80);
                return classes_vec;
            }
        };
    }
}

#endif //GPD_YOLO_DETECTOR_H
