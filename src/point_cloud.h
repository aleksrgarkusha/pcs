#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Core>

namespace pcs {

class FeatureEstimator;

class PointCloud {
  friend class FeatureEstimator;
  friend std::shared_ptr<PointCloud> voxel_down_sample(const PointCloud& input,
                                                       double voxel_size);

 public:
  PointCloud() = default;
  PointCloud(const Eigen::MatrixXd& points);
  PointCloud(const Eigen::MatrixXd& points, const Eigen::MatrixXd& colors);
  PointCloud(const PointCloud& other) = default;
  PointCloud(PointCloud&& other) = default;
  ~PointCloud() = default;

  PointCloud& operator=(const PointCloud& other) = default;
  PointCloud& operator=(PointCloud&& other) = default;

  void add_point(const Eigen::Vector3d& point);
  void add_point_and_color(const Eigen::Vector3d& point,
                           const Eigen::Vector3d& color);
  void add_points(const Eigen::MatrixXd& points);
  void add_points_and_colors(const Eigen::MatrixXd& points,
                             const Eigen::MatrixXd& colors);

  Eigen::Vector3d get_point(std::size_t idx) const;
  Eigen::Vector3d get_color(std::size_t idx) const;
  std::pair<Eigen::Vector3d, Eigen::Vector3d> get_point_and_color(
      std::size_t idx) const;
  Eigen::MatrixXd get_points() const;
  Eigen::MatrixXd get_colors() const;

  void clear();
  bool empty() const;

  std::size_t num_points() const;

  Eigen::Vector3d get_min_bound() const;
  Eigen::Vector3d get_max_bound() const;

  bool has_points() const;
  bool has_colors() const;

  // Estimate mean distance beetween point and its
  // first nearest neighbor for all points in the cloud
  double estimate_mean_distance() const;

 private:
  std::vector<Eigen::Vector3d> points_;
  std::vector<Eigen::Vector3d> colors_;
};

// Down scale given point cloud according to the "voxel hashing" procedure
std::shared_ptr<PointCloud> voxel_down_sample(const PointCloud& input,
                                              double voxel_size);
}  // namespace pcs
