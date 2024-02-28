// Standard Library imports
#include <ctime>
#include <iostream>
#include <limits>

// External imports
#include <Eigen/Core>
#include <Eigen/Dense>
#include <webots/Camera.hpp>
#include <webots/Motor.hpp>
#include <webots/Robot.hpp>

// Local imports
#include "libFeatures.h"
#include "libHelloSLAM.h"
#include "libPoseEstimation.h"

int init_wait_steps = 20;

cv::Mat process_image(const unsigned char *image, int image_width,
                      int image_height) {
  cv::Mat image_mat = cv::Mat(cv::Size(image_width, image_height), CV_8UC4);
  image_mat.data = const_cast<uchar *>(image);

  return image_mat;
}

void triangulation(const std::vector<cv::KeyPoint> &keypoint_1,
                   const std::vector<cv::KeyPoint> &keypoint_2,
                   const std::vector<cv::DMatch> &matches, const cv::Mat &R,
                   const cv::Mat &t, std::vector<cv::Point3d> &points) {
  // Projection matrix at t=0
  cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

  // Projection matrix after motion
  cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
                R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0),
                R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                t.at<double>(2, 0));

  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  std::vector<cv::Point2f> pts_1, pts_2;
  for (cv::DMatch m : matches) {
    // Convert pixel coordinates to camera coordinates
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  cv::Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // Convert to non−homogeneous coordinates by dividing by the last coordinate
  for (int i = 0; i < pts_4d.cols; i++) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0);
    cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}

// Note that only one instance of Robot should be created in a controller
// program. The arguments of the main function can be specified by the
// "controllerArgs" field of the Robot node
int main(int argc, char **argv) {
  printHello();

  print_version();

  webots::Robot *robot = new webots::Robot();

  // get the time step of the current world.
  int timeStep = (int)robot->getBasicTimeStep();

  webots::Camera *camera_left = robot->getCamera("MultiSense S21 left camera");
  webots::Camera *camera_right =
      robot->getCamera("MultiSense S21 right camera");
  webots::Camera *camera_center =
      robot->getCamera("MultiSense S21 meta camera");

  webots::Motor *motor_left = robot->getMotor("middle_left_wheel_joint");
  webots::Motor *motor_right = robot->getMotor("middle_right_wheel_joint");

  motor_left->setPosition(std::numeric_limits<double>::infinity());
  motor_right->setPosition(std::numeric_limits<double>::infinity());

  motor_left->setVelocity(2);   // rad/s
  motor_right->setVelocity(4);  // rad/s

  camera_left->enable(timeStep);
  camera_right->enable(timeStep);
  camera_center->enable(timeStep);

  int image_width = camera_left->getWidth();
  int image_height = camera_left->getHeight();

  // −− initialization
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  cv::Mat prev_image_left;

  while (robot->step(timeStep) != -1) {
    const unsigned char *im_left = camera_left->getImage();
    const unsigned char *im_right = camera_right->getImage();
    const unsigned char *im_center = camera_center->getImage();
    cv::Mat image_left = process_image(im_left, image_width, image_height);
    cv::Mat image_right = process_image(im_right, image_width, image_height);
    cv::Mat image_center = process_image(im_center, image_width, image_height);

    // 2D monocular slam initialization requires some translation movement
    // Store the first image and skip some simulation steps
    if (init_wait_steps > 0) {
      std::cout << "Skipping first iterations..." << std::endl;
      init_wait_steps -= 1;

      if (prev_image_left.empty()) {
        std::cout << "Replacing prev image" << std::endl;
        prev_image_left = image_left.clone();
      }
      continue;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // FEATURE MATCHING
    ////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "Previous image not empty, hence, starting feature detection."
              << std::endl;

    std::vector<cv::DMatch> matches = feature_matching(
        detector, matcher, prev_image_left, image_left, keypoints_1,
        keypoints_2, descriptors_1, descriptors_2);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Solving Camera Motion with Epipolar Constraints
    //
    // Estimate the camera’s motion based on the feature matching
    // Monocular camera: pose estimation with 2D-2D epipolar geometry
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Create the Essential matrix and extract R and t from it
    // t is the direction of the translation vector and has unit length.
    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    print_matrices_and_constraints(R, t, matches, keypoints_1, keypoints_2);

    // Obtain the depth of the points by triangulation
    std::vector<cv::Point3d> points_3d;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points_3d);
    for (int i = 0; i < (int)points_3d.size(); i++) {
      // std::cout << "3D point = " << points_3d[i] << std::endl;
    }

    prev_image_left = image_left;
  };

  delete robot;
  return 0;
}

// std::cout << cv::typeToString(inputMat.type()) << std::endl;
