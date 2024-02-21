#ifndef LIBPOSEESTIMATION_H_
#define LIBPOSEESTIMATION_H_
#include "opencv2/opencv.hpp"

void pose_estimation_2d2d(std::vector<cv::KeyPoint>& keypoints_1,
                          std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches,
                          cv::Mat& R,
                          cv::Mat& t);
#endif