#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <point_cloud.h>

TEST_CASE("Point cloud tests") {
  pcs::PointCloud point_cloud;
  CHECK(point_cloud.empty());
  CHECK_FALSE(point_cloud.has_points());
  CHECK_FALSE(point_cloud.has_colors());
  CHECK(point_cloud.estimate_mean_distance() == 0);

  point_cloud.add_point_and_color(Eigen::Vector3d(0, 0, 0),
                                  Eigen::Vector3d(0, 0, 0));
  CHECK_FALSE(point_cloud.empty());
  CHECK(point_cloud.has_points());
  CHECK(point_cloud.has_colors());
  CHECK(point_cloud.get_point(0) == Eigen::Vector3d(0, 0, 0));
  CHECK(point_cloud.get_color(0) == Eigen::Vector3d(0, 0, 0));
  CHECK(point_cloud.get_point_and_color(0).first == Eigen::Vector3d(0, 0, 0));
  CHECK(point_cloud.get_point_and_color(0).second == Eigen::Vector3d(0, 0, 0));
  point_cloud.clear();

  auto min_bound = point_cloud.get_min_bound();
  CHECK(min_bound.x() == Approx(0.0));
  CHECK(min_bound.y() == Approx(0.0));
  CHECK(min_bound.z() == Approx(0.0));

  auto max_bound = point_cloud.get_max_bound();
  CHECK(max_bound.x() == Approx(0.0));
  CHECK(max_bound.y() == Approx(0.0));
  CHECK(max_bound.z() == Approx(0.0));

  point_cloud.clear();
  CHECK(point_cloud.empty());
  CHECK_FALSE(point_cloud.has_points());
  CHECK_FALSE(point_cloud.has_colors());

  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(5, 3);
  Eigen::MatrixXd colors = Eigen::MatrixXd::Zero(5, 3);
  point_cloud.add_points_and_colors(points, colors);
  CHECK(point_cloud.num_points() == 5);
  CHECK(point_cloud.get_points().rows() == 5);
  CHECK(point_cloud.get_colors().rows() == 5);

  point_cloud = pcs::PointCloud(points, colors);
  CHECK(point_cloud.num_points() == 5);
  CHECK(point_cloud.get_points().rows() == 5);
  CHECK(point_cloud.get_colors().rows() == 5);

  point_cloud = pcs::PointCloud(points);
  CHECK(point_cloud.has_points());
  CHECK_FALSE(point_cloud.has_colors());
  CHECK(point_cloud.num_points() == 5);
  CHECK(point_cloud.get_point(0) == Eigen::Vector3d(0, 0, 0));
  CHECK(point_cloud.get_points().rows() == 5);
  CHECK(point_cloud.get_colors().rows() == 0);
  point_cloud.add_point(Eigen::Vector3d(0, 0, 0));
  CHECK(point_cloud.num_points() == 6);
  point_cloud.add_points(points);
  CHECK(point_cloud.num_points() == 11);

  CHECK(point_cloud.estimate_mean_distance() == 0);
}
