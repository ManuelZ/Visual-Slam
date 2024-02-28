#pragma once
#include <opencv2/opencv.hpp>

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);

void pose_estimation_2d2d(const std::vector<cv::KeyPoint>& keypoints_1,
                          const std::vector<cv::KeyPoint>& keypoints_2,
                          const std::vector<cv::DMatch>& matches,
                           cv::Mat& R,
                           cv::Mat& t);

void print_matrices_and_constraints(
    const cv::Mat& R, const cv::Mat& t, const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints_1,
    const std::vector<cv::KeyPoint>& keypoints_2); 