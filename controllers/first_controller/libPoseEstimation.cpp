
#include <iostream>
#include <opencv2/opencv.hpp>

// Transform from pixel coordinates to the image plane
// Eq. 4.5 of the Slambook.
// The pixel coordinate system has the origin in the upper left corner of the
// image There is a translation of the origin and a zoom
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
  double u = p.x;
  double v = p.y;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  return cv::Point2d((u - cx) / fx, (v - cy) / fy);
}

void pose_estimation_2d2d(const std::vector<cv::KeyPoint>& keypoints_1,
                          const std::vector<cv::KeyPoint>& keypoints_2,
                          const std::vector<cv::DMatch>& matches,
                          cv::Mat& R, cv::Mat& t) {
  // Camera Intrinsics, TUM Freiburg2
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  // −− Convert the matching point to the form of vector<Point2f>
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;

  for (int i = 0; i < (int)matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  // −− Calculate fundamental matrix
  cv::Mat fundamental_matrix;
  fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
  std::cout << "fundamental_matrix is " << std::endl
            << fundamental_matrix << std::endl;

  // −− Calculate essential matrix
  cv::Point2d principal_point(
      325.1, 249.7);  // camera principal point, calibrated in TUM dataset
  double focal_length = 521;

  // camera focal length, calibrated in TUM dataset
  cv::Mat essential_matrix;
  essential_matrix =
      findEssentialMat(points1, points2, focal_length, principal_point);
  std::cout << "essential_matrix is " << std::endl
            << essential_matrix << std::endl;

  // −− Calculate homography matrix
  // −− But the scene is not planar, and calculating the homography matrix here
  // is of little significance
  cv::Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, cv::RANSAC, 3);
  std::cout << "homography_matrix is " << std::endl
            << homography_matrix << std::endl;

  // −− Recover rotation and translation from the essential matrix.
  // t is the direction of the translation vector and has unit length.
  cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length,
                  principal_point);

}

void print_matrices_and_constraints(
    const cv::Mat& R, const cv::Mat& t, const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints_1,
    const std::vector<cv::KeyPoint>& keypoints_2) {

  std::cout << "R is " << std::endl << R << std::endl;
  std::cout << "t is " << std::endl << t << std::endl;

  // −− Check E=t^R∗scale
  cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0),
                 t.at<double>(1, 0), t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                 -t.at<double>(1, 0), t.at<double>(0, 0), 0);

  std::cout << "t^R=" << std::endl << t_x * R << std::endl;

  // −− Check epipolar constraints
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  for (cv::DMatch m : matches) {
    cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);

    cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

    cv::Mat d = y2.t() * t_x * R * y1;
    // std::cout << "epipolar constraint = " << d << std::endl;
  }
}
