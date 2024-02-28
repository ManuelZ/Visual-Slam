#ifndef LIBFEATURES_H_
#define LIBFEATURES_H_
#include "opencv2/opencv.hpp"

std::vector<cv::DMatch> feature_matching(
  const cv::Ptr<cv::FeatureDetector>& detector,
  const cv::Ptr<cv::DescriptorMatcher>& matcher,
  const cv::Mat &image_1,
  const cv::Mat &image_2,
  std::vector<cv::KeyPoint> &keypoints_1,
  std::vector<cv::KeyPoint> &keypoints_2,
  cv::Mat &descriptors_1,
  cv::Mat &descriptors_2
);
#endif