#include "point_cloud.h"
#include "kd_tree.hpp"
#include "rgb_to_hsv.hpp"

#include <Eigen/Dense>

namespace pcs {

PointCloud::PointCloud(const Eigen::MatrixXd &points,
                       const Eigen::MatrixXd &colors) {
  points_.resize(points.rows());
  colors_.resize(colors.rows());
  for (auto i = 0u; i < points.rows(); ++i) {
    points_[i] = points.row(i);
  }
  for (auto i = 0u; i < colors.rows(); ++i) {
    colors_[i] = rgb_to_hsv(colors.row(i));
  }
}

PointCloud::PointCloud(const Eigen::MatrixXd &points) {
  points_.resize(points.rows());
  for (auto i = 0u; i < points.rows(); ++i) {
    points_[i] = points.row(i);
  }
}

void PointCloud::clear() {
  points_.clear();
  colors_.clear();
}

bool PointCloud::empty() const { return !has_points(); }

Eigen::Vector3d PointCloud::get_min_bound() const {
  if (!has_points()) {
    return Eigen::Vector3d(0.0, 0.0, 0.0);
  }
  auto itr_x =
      std::min_element(points_.begin(), points_.end(),
                       [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                         return a(0) < b(0);
                       });
  auto itr_y =
      std::min_element(points_.begin(), points_.end(),
                       [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                         return a(1) < b(1);
                       });
  auto itr_z =
      std::min_element(points_.begin(), points_.end(),
                       [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                         return a(2) < b(2);
                       });
  return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d PointCloud::get_max_bound() const {
  if (!has_points()) {
    return Eigen::Vector3d(0.0, 0.0, 0.0);
  }
  auto itr_x =
      std::max_element(points_.begin(), points_.end(),
                       [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                         return a(0) < b(0);
                       });
  auto itr_y =
      std::max_element(points_.begin(), points_.end(),
                       [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                         return a(1) < b(1);
                       });
  auto itr_z =
      std::max_element(points_.begin(), points_.end(),
                       [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                         return a(2) < b(2);
                       });
  return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

bool PointCloud::has_points() const { return points_.size() > 0; }

bool PointCloud::has_colors() const {
  return points_.size() > 0 && colors_.size() == points_.size();
}

double PointCloud::estimate_mean_distance() const {
  double mean_distance = 0;
  if (points_.size() < 2) {
    return mean_distance;
  }

  KDTree<3> tree(points_);
  std::vector<double> distances;
  distances.reserve(points_.size());
  for (const auto &point : points_) {
    auto neighbors = tree.find_nns(point, 2);
    distances.push_back(neighbors[1].second);
  }

  for (auto d : distances) {
    mean_distance += d;
  }
  mean_distance /= distances.size();

  return mean_distance;
}

void PointCloud::add_point(const Eigen::Vector3d &point) {
  points_.push_back(point);
}

void PointCloud::add_point_and_color(const Eigen::Vector3d &point,
                                     const Eigen::Vector3d &color) {
  points_.push_back(point);
  colors_.push_back(rgb_to_hsv(color));
}

void PointCloud::add_points(const Eigen::MatrixXd &points) {
  for (auto i = 0u; i < points.rows(); ++i) {
    points_.push_back(points.row(i));
  }
}

void PointCloud::add_points_and_colors(const Eigen::MatrixXd &points,
                                       const Eigen::MatrixXd &colors) {
  for (auto i = 0u; i < points.rows(); ++i) {
    points_.push_back(points.row(i));
    colors_.push_back(rgb_to_hsv(colors.row(i)));
  }
}

Eigen::Vector3d PointCloud::get_point(std::size_t idx) const {
  return points_[idx];
}

Eigen::Vector3d PointCloud::get_color(std::size_t idx) const {
  return hsv_to_rgb(colors_[idx]);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> PointCloud::get_point_and_color(
    std::size_t idx) const {
  return std::make_pair(points_[idx], hsv_to_rgb(colors_[idx]));
}
std::size_t PointCloud::num_points() const { return points_.size(); }

Eigen::MatrixXd PointCloud::get_points() const {
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(points_.size(), 3);
  for (auto i = 0u; i < points_.size(); ++i) {
    points.row(i) = points_[i];
  }
  return points;
}

Eigen::MatrixXd PointCloud::get_colors() const {
  Eigen::MatrixXd colors = Eigen::MatrixXd::Zero(colors_.size(), 3);
  for (auto i = 0u; i < colors_.size(); ++i) {
    colors.row(i) = hsv_to_rgb(colors_[i]);
  }
  return colors;
}

}  // namespace pcs
