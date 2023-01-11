#include <memory>
#include <chrono>
#include "super_glue.h"
#include "super_point.h"

int main(int argc, char** argv){
  if(argc != 5){
    std::cerr << "./superpointglueacceleration config_path model_dir image0_absolutely_path image1_absolutely_path" << std::endl;
    return 0;
  }
  std::string config_path = argv[1];
  std::string model_dir = argv[2];
  std::string image0_path = argv[3];
  std::string image1_path = argv[4];
  cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
  if(image0.empty() || image1.empty()){
    std::cerr << "Image Path Error." << std::endl;
    return 0;
  }
  Configs configs(config_path, model_dir);
  std::cout << "Building Engine......" << std::endl;
  auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
  if (!superpoint->build()){
    std::cerr << "Error in SuperPoint building" << std::endl;
    return 0;
  }
  auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
  if (!superglue->build()){
    std::cerr << "Error in SuperGlue building" << std::endl;
    return 0;
  }

  std::cout << "SuperPoint and SuperGlue Build Success." << std::endl;
  Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
  auto start = std::chrono::high_resolution_clock::now();
  if(!superpoint->infer(image0, feature_points0)){
    std::cerr << "Failed when extracting features of image0 !" << std::endl;
    return 0;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Infer First Image Cost " << duration.count() << " MS" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  if(!superpoint->infer(image1, feature_points1)){
    std::cerr << "Failed when extracting features of image1 !" << std::endl;
    return 0;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Infer Second Image Cost " << duration.count() << " MS" << std::endl;
  std::vector<cv::DMatch> superglue_matches;
  start = std::chrono::high_resolution_clock::now();
  superglue->matching_points(feature_points0, feature_points1, superglue_matches);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Match Image Cost " << duration.count() << " MS" << std::endl;
  cv::Mat match_image;
  std::vector<cv::KeyPoint> keypoints0, keypoints1;
  for(size_t i = 0; i < feature_points0.cols(); ++i){
    double score = feature_points0(0, i);
    double x = feature_points0(1, i);
    double y = feature_points0(2, i);
    keypoints0.emplace_back(x, y, 8, -1, score);
  }
  for(size_t i = 0; i < feature_points1.cols(); ++i){
    double score = feature_points1(0, i);
    double x = feature_points1(1, i);
    double y = feature_points1(2, i);
    keypoints1.emplace_back(x, y, 8, -1, score);
  }
  cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
  cv::imwrite("match_image.jpg", match_image);
  //  visualize
  //  cv::imshow("match_image", match_image);
  //  cv::waitKey(-1);
  return 0;
}