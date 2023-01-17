//
// Created by haoyuefan on 2021/9/22.
//

#include <memory>
#include <chrono>
#include <thread>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

cv::Mat image00 = cv::imread("/workspace/SuperPoint-SuperGlue-TensorRT/image/image0.png", cv::IMREAD_GRAYSCALE);
cv::Mat image11 = cv::imread("/workspace/SuperPoint-SuperGlue-TensorRT/image/image1.png", cv::IMREAD_GRAYSCALE);

void infer_image00() {
    Configs configs("../config/config.yaml", "../weights/");
    cv::resize(image00, image00, cv::Size(320, 240));
    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;
    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;
    double image00_tcount = 0;
    for (int i = 0; i <= 10000; ++i) {
        std::cout << "---------------------------image00------------------------------" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image00, feature_points0)) {
            std::cerr << "Failed when extracting features from first image." << std::endl;
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 0) {
            std::cout << "First image feature points number: " << feature_points0.cols() << std::endl;
            image00_tcount += duration.count();
            std::cout << "First image infer cost " << image00_tcount / i << " MS" << std::endl;
        }
    }
}

void infer_image11(){
    Configs configs("../config/config.yaml", "../weights/");
    cv::resize(image11, image11, cv::Size(320, 240));
    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()){
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()){
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;
    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points1;
    double image11_tcount = 0;
    for (int i = 0; i <= 10000; ++i) {
        std::cout << "---------------------------image11------------------------------" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image11, feature_points1)) {
            std::cerr << "Failed when extracting features from second image." << std::endl;
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 0) {
            std::cout << "Second image feature points number: " << feature_points1.cols() << std::endl;
            image11_tcount += duration.count();
            std::cout << "Second image infer cost " << image11_tcount / i << " MS" << std::endl;
        }
    }
}

int main(int argc, char** argv){
    std::thread t1(infer_image00);
    std::thread t2(infer_image11);
    t1.join();
    t2.join();

    return 0;
}
