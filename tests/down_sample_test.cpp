#include "catch.hpp"

#include <point_cloud.h>

TEST_CASE("Downsample tests") {
  pcs::PointCloud point_cloud;
  auto down_sampled = pcs::voxel_down_sample(point_cloud, 0.5);
  CHECK(down_sampled->empty());

  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(8, 3);
  points.row(0) = Eigen::Vector3d(1.0, 0.0, 0.0);
  points.row(1) = Eigen::Vector3d(0.0, 1.0, 0.0);
  points.row(2) = Eigen::Vector3d(0.0, 0.0, 1.0);
  points.row(3) = Eigen::Vector3d(1.0, 1.0, 0.0);
  points.row(4) = Eigen::Vector3d(0.0, 1.0, 1.0);
  points.row(5) = Eigen::Vector3d(1.0, 0.0, 1.0);
  points.row(6) = Eigen::Vector3d(0.0, 0.0, 0.0);
  points.row(7) = Eigen::Vector3d(1.0, 1.0, 1.0);
  point_cloud.add_points(points);

  down_sampled = pcs::voxel_down_sample(point_cloud, 3);
  CHECK(down_sampled->num_points() == 1);
  down_sampled = pcs::voxel_down_sample(point_cloud, -1.0);
  CHECK(down_sampled->empty());
  down_sampled = pcs::voxel_down_sample(point_cloud, 1.0E-15);
  CHECK(down_sampled->empty());

  Eigen::MatrixXd colors = Eigen::MatrixXd::Ones(8, 3);
  point_cloud.clear();
  point_cloud.add_points_and_colors(points, colors);
  down_sampled = pcs::voxel_down_sample(point_cloud, 10);
  REQUIRE(down_sampled->num_points() == 1);
  REQUIRE(down_sampled->has_colors());
  CHECK(down_sampled->get_color(0) == Eigen::Vector3d(1.0, 1.0, 1.0));
}