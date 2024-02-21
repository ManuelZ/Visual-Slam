// You may need to add webots include files such as
// <webots/DistanceSensor.hpp>, <webots/Motor.hpp>, etc.
// and/or to add some other includes
#include <webots/Robot.hpp>
#include <webots/Camera.hpp>
#include <iostream>
#include <ctime>
#include <Eigen/Core>

// Algebraic operations of dense matrices (inverse, eigenvalues, etc.)
#include <Eigen/Dense>
#include "libHelloSLAM.h"
#include "libPoseEstimation.h"

#define MATRIX_SIZE 50

// All the webots classes are defined in the "webots" namespace
using namespace webots;

cv::Mat process_image(const unsigned char *image, int image_width, int image_height)
{

  cv::Mat image_mat = cv::Mat(cv::Size(image_width, image_height), CV_8UC4);
  image_mat.data = const_cast<uchar *>(image);

  return image_mat;
}

// Transform from pixel coordinates to the image plane
// Eq. 4.5 of the Slambook.
// The pixel coordinate system has the origin in the upper left corner of the image
// There is a translation of the origin and a zoom 
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{

  double u = p.x;
  double v = p.y;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  
  return cv::Point2d(
      (u - cx) / fx,
      (v - cy) / fy);
}

void triangulation(
    const std::vector<cv::KeyPoint> &keypoint_1,
    const std::vector<cv::KeyPoint> &keypoint_2,
    const std::vector<cv::DMatch> &matches,
    const cv::Mat &R, const cv::Mat &t,
    std::vector<cv::Point3d> &points)
{
  // Projection matrix at t=0
  cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);

  // Projection matrix after motion
  cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  std::vector<cv::Point2f> pts_1, pts_2;
  for (cv::DMatch m : matches)
  {
    // Convert pixel coordinates to camera coordinates
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  cv::Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // Convert to non−homogeneous coordinates by dividing by the last coordinate
  for (int i = 0; i < pts_4d.cols; i++)
  {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0);
    cv::Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0));
    points.push_back(p);
  }
}

// This is the main program of your controller.
// It creates an instance of your Robot instance, launches its
// function(s) and destroys it at the end of the execution.
// Note that only one instance of Robot should be created in a controller program.
// The arguments of the main function can be specified by the "controllerArgs" field of the Robot node
int main(int argc, char **argv)
{

  printHello();

  Robot *robot = new Robot();

  // get the time step of the current world.
  int timeStep = (int)robot->getBasicTimeStep();

  Camera *camera_left = robot->getCamera("MultiSense S21 left camera");
  Camera *camera_right = robot->getCamera("MultiSense S21 right camera");
  Camera *camera_center = robot->getCamera("MultiSense S21 meta camera");

  camera_left->enable(timeStep);
  camera_right->enable(timeStep);
  camera_center->enable(timeStep);

  int image_width = camera_left->getWidth();
  int image_height = camera_left->getHeight();

  // −− initialization
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  if (__cplusplus == 202101L)
    std::cout << "C++23";
  else if (__cplusplus == 202002L)
    std::cout << "C++20";
  else if (__cplusplus == 201703L)
    std::cout << "C++17";
  else if (__cplusplus == 201402L)
    std::cout << "C++14";
  else if (__cplusplus == 201103L)
    std::cout << "C++11";
  else if (__cplusplus == 199711L)
    std::cout << "C++98";
  else
    std::cout << "pre-standard C++." << __cplusplus;
  std::cout << "\n";

  while (robot->step(timeStep) != -1)
  {

    const unsigned char *im_left = camera_left->getImage();
    const unsigned char *im_right = camera_right->getImage();
    const unsigned char *im_center = camera_center->getImage();
    cv::Mat image_left = process_image(im_left, image_width, image_height);
    cv::Mat image_right = process_image(im_right, image_width, image_height);
    cv::Mat image_center = process_image(im_center, image_width, image_height);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // START: FEATURE MATCHING
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // −− detect Oriented FAST
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(image_left, keypoints_1);
    detector->detect(image_right, keypoints_2);

    // −− compute BRIEF descriptor
    descriptor->compute(image_left, keypoints_1, descriptors_1);
    descriptor->compute(image_right, keypoints_2, descriptors_2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds. " << std::endl;

    cv::Mat outimg1;
    cv::drawKeypoints(image_left, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outimg1);

    // −− use Hamming distance to match the features
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds. " << std::endl;

    // −− sort and remove the outliers
    //  min and max distance
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                       [](const cv::DMatch &m1, const cv::DMatch &m2)
                                       { return m1.distance < m2.distance; });

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // remove the bad matching
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
      if (matches[i].distance <= std::max(2 * min_dist, 30.0))
      {
        good_matches.push_back(matches[i]);
      }
    }

    std::cout << "Good matches: " << good_matches.size() << std::endl;

    // Ch. 6.3.2 Essential Matrix estimation requires at minimum 8 matches.
    if (std::size(good_matches) < 8)
    {
      throw std::runtime_error("Only " + std::to_string(std::size(good_matches)) + " matches found, but at least 8 were required.");
    }

    // −− draw the results
    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(image_left, keypoints_1, image_right, keypoints_2, matches, img_match);
    cv::drawMatches(image_left, keypoints_1, image_right, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("all matches", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(0);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // START: Solving Camera Motion with Epipolar Constraints
    // Estimate the camera’s motion based on the feature matching
    // Monocular camera: pose estimation with 2D-2D epipolar geometry
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Create the Essential matrix and extract R and t from it
    // t is the direction of the translation vector and has unit length. 
    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // −− Check E=t^R∗scale
    cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                   t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                   -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    std::cout << "t^R=" << std::endl
              << t_x * R << std::endl;

    // −− Check epipolar constraints
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (cv::DMatch m : matches)
    {
      cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
      cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);

      cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
      cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

      cv::Mat d = y2.t() * t_x * R * y1;
      // std::cout << "epipolar constraint = " << d << std::endl;
    }

    std::vector<cv::Point3d> points_3d;
    triangulation(keypoints_1, keypoints_2, good_matches, R, t, points_3d);
    for (int i = 0; i < (int)points_3d.size(); i++)
    {
      std::cout << "3D point = " << points_3d[i] << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Process sensor data here.
    // Enter here functions to send actuator commands, like:
    // motor->setPosition(10.0);
  };

  delete robot;
  return 0;
}

// std::cout << cv::typeToString(inputMat.type()) << std::endl;
