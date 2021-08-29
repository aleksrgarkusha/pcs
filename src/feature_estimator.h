#pragma once
#include <string>
#include <vector>

#include "kd_tree.hpp"
#include "point_cloud.h"
#include "thread_pool.hpp"

namespace pcs {

class FeatureEstimator {
  static constexpr std::size_t features_per_scale = 15;
  static constexpr std::size_t color_features = 6;

 public:
  FeatureEstimator(std::shared_ptr<pcs::PointCloud> point_cloud,
                   double voxel_size = 0.05, int num_neighbors = 10,
                   int num_scales = 9, int batch_size = 1000);

  ~FeatureEstimator() = default;

  std::size_t feature_size() const;
  std::size_t num_points() const;
  std::size_t num_batches() const;
  std::size_t batch_size() const;

  Eigen::VectorXd get_features_for_point(std::size_t point_id) const;

  // All of this methods will calculate features in parall on thread pool
  Eigen::MatrixXd get_features_for_points(const std::vector<int> &idxs) const;
  Eigen::MatrixXd get_features_for_batch(std::size_t batch_id) const;
  Eigen::MatrixXd get_features_for_batch(std::size_t start_id,
                                         std::size_t end_id) const;

  // Refine segmentation results by averaging neighboring points class
  // probabilities.
  std::vector<unsigned int> soft_voting_smoothing(
      const Eigen::MatrixXd &probabilities,
      std::size_t num_neighbors = 10) const;

  // Refine segmentation results by averaging neighboring points labels
  // with majority voting scheme
  std::vector<unsigned int> hard_voting_smoothing(
      const std::vector<unsigned int> &labels, std::size_t num_neighbors) const;

 private:
  std::shared_ptr<PointCloud> point_cloud_;
  double voxel_size_;
  int num_neighbors_;
  int num_scales_;
  int batch_size_;
  std::shared_ptr<KDTree<3>> tree_;
  std::vector<std::shared_ptr<PointCloud>> point_clouds_;
  std::vector<std::shared_ptr<KDTree<3>>> trees_;
  std::size_t num_points_;
  bool has_colors_;
  mutable ThreadPool pool_;
};

}  // namespace pcs
