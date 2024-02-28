#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

std::vector<cv::DMatch> feature_matching(
    const cv::Ptr<cv::FeatureDetector> &detector,
    const cv::Ptr<cv::DescriptorMatcher> &matcher,
    const cv::Mat &image_1,
    const cv::Mat &image_2,
    std::vector<cv::KeyPoint> &keypoints_1,
    std::vector<cv::KeyPoint> &keypoints_2,
    cv::Mat &descriptors_1,
    cv::Mat &descriptors_2) {
  // Detect keypoints and features
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  detector->detectAndCompute(image_1, cv::Mat(), keypoints_1, descriptors_1);
  detector->detectAndCompute(image_2, cv::Mat(), keypoints_2, descriptors_2);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "Extract ORB cost = " << time_used.count() << " seconds. "
            << std::endl;

  // cv::Mat outimg1;
  // cv::drawKeypoints(image_1, keypoints_1, outimg1, cv::Scalar::all(-1),
  // cv::DrawMatchesFlags::DEFAULT); cv::imshow("ORB features image 1",
  // outimg1);

  // cv::Mat outimg2;
  // cv::drawKeypoints(image_2, keypoints_2, outimg2, cv::Scalar::all(-1),
  // cv::DrawMatchesFlags::DEFAULT); cv::imshow("ORB features image 2",
  // outimg2);

  // −− use Hamming distance to match the features
  std::vector<cv::DMatch> matches;
  t1 = std::chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = std::chrono::steady_clock::now();
  time_used =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "Match ORB cost = " << time_used.count() << " seconds. "
            << std::endl;

  // −− sort and remove the outliers
  //  min and max distance
  auto min_max =
      std::minmax_element(matches.begin(), matches.end(),
                          [](const cv::DMatch &m1, const cv::DMatch &m2) {
                            return m1.distance < m2.distance;
                          });

  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // remove the bad matching
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  std::cout << "Good matches: " << good_matches.size() << std::endl;

  // Ch. 6.3.2 Essential Matrix estimation requires at minimum 8 matches.
  if (std::size(good_matches) < 8) {
    throw std::runtime_error("Only " + std::to_string(std::size(good_matches)) +
                             " matches found, but at least 8 were required.");
  }

  // −− draw the results
  cv::Mat img_match;
  cv::Mat img_goodmatch;
  cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches,
                  img_match);
  cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2, good_matches,
                  img_goodmatch);
  // cv::imshow("all matches", img_match);
  cv::imshow("good matches", img_goodmatch);
  cv::waitKey(0);

  return good_matches;
}
