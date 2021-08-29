#include "feature_estimator.h"

#include <Eigen/Dense>

namespace {

int calculate_medoid_index(std::vector<Eigen::Vector3d> &points) {
  std::vector<double> distances(points.size(), 0);
  for (auto i = 0u; i < points.size(); ++i) {
    for (auto j = 0u; j < points.size(); ++j) {
      const double distance = (points[i] - points[j]).eval().norm();
      distances[i] += distance;
    }
  }

  int min_idx = 0;
  double min_distance = std::numeric_limits<double>::max();
  for (auto i = 0u; i < distances.size(); ++i) {
    if (distances[i] < min_distance) {
      min_idx = i;
      min_distance = distances[i];
    }
  }

  return min_idx;
}

}  // namespace

namespace pcs {

FeatureEstimator::FeatureEstimator(std::shared_ptr<pcs::PointCloud> point_cloud,
                                   double voxel_size, int num_neighbors,
                                   int num_scales, int batch_size)
    : point_cloud_(point_cloud),
      voxel_size_(voxel_size),
      num_neighbors_(num_neighbors),
      num_scales_(num_scales),
      batch_size_(batch_size),
      tree_(nullptr),
      num_points_(0),
      has_colors_(false),
      pool_(std::thread::hardware_concurrency()) {
  if (point_cloud_ and not point_cloud_->empty()) {
    num_points_ = point_cloud_->points_.size();
    has_colors_ = point_cloud_->has_colors();
    tree_ = std::shared_ptr<KDTree<3>>(new KDTree<3>(
        point_cloud_->points_, std::thread::hardware_concurrency()));

    // Build points pyramid
    point_clouds_.push_back(point_cloud_);
    trees_.push_back(tree_);
    for (auto s = 1u; s < num_scales_; ++s) {
      const auto voxel_scale = std::pow(2, s);
      auto down_scaled_cloud =
          voxel_down_sample(*(point_clouds_[s - 1]), voxel_size_ * voxel_scale);
      point_clouds_.push_back(down_scaled_cloud);

      auto tree = std::shared_ptr<KDTree<3>>(new KDTree<3>(
          point_clouds_[s]->points_, std::thread::hardware_concurrency()));
      trees_.push_back(tree);
    }
  }
}

Eigen::VectorXd FeatureEstimator::get_features_for_point(
    std::size_t point_id) const {
  if (point_id >= num_points_) {
    std::string error_string("given point index { ");
    error_string +=
        std::to_string(point_id) + " } is bigger then point cloud size { ";
    error_string += std::to_string(num_points_) + " }";
    throw std::out_of_range(error_string);
  }

  const std::size_t num_features = feature_size();
  Eigen::VectorXd features(num_features);
  Eigen::Vector3d point = point_cloud_->points_[point_id];
  Eigen::Vector3d neig_color(0., 0., 0.);

  for (auto s = 0u; s < point_clouds_.size(); ++s) {
    const std::size_t scale_offset = s * features_per_scale;
    auto neighbors = trees_[s]->find_nns(point, num_neighbors_);
    if (s == 0 and has_colors_) {
      for (auto k = 0u; k < neighbors.size(); ++k) {
        neig_color += (point_cloud_->colors_[neighbors[k].first]);
      }
    }
    std::vector<Eigen::Vector3d> n_points;
    for (auto j = 0u; j < neighbors.size(); ++j) {
      n_points.push_back(point_clouds_[s]->points_[neighbors[j].first]);
    }

    Eigen::Vector3d medoid = n_points[calculate_medoid_index(n_points)];
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(num_neighbors_, 3);
    for (auto j = 0u; j < num_neighbors_; ++j) {
      data.row(j) = n_points[j].transpose();
    }

    Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();
    cov = (centered.transpose() * centered) / (num_neighbors_ - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    Eigen::Matrix3d eigenvectors = eig.eigenvectors();
    Eigen::Vector3d eigenvalues = eig.eigenvalues();
    eigenvalues.normalize();

    // Covariance features
    const double omnivariance =
        std::pow(eigenvalues(0) * eigenvalues(1) * eigenvalues(2), 1.0 / 3.0);

    features[scale_offset + 0] = omnivariance;
    double eigenentropy = 0.0;
    for (int k = 0; k < 3; ++k) {
      eigenentropy -= (eigenvalues(k) * std::log(eigenvalues(k)));
    }

    features[scale_offset + 1] = eigenentropy;

    const double anisotropy =
        (eigenvalues(2) - eigenvalues(0)) / eigenvalues(2);
    features[scale_offset + 2] = anisotropy;

    const double planarity = (eigenvalues(1) - eigenvalues(0)) / eigenvalues(2);
    features[scale_offset + 3] = planarity;

    const double linearity = (eigenvalues(2) - eigenvalues(1)) / eigenvalues(2);
    features[scale_offset + 4] = linearity;

    const double surface_variation = eigenvalues(0);
    features[scale_offset + 5] = surface_variation;

    const double scatter = eigenvalues(0) / eigenvalues(2);
    features[scale_offset + 6] = scatter;

    const double verticality = 1. - std::abs(eigenvectors.col(0)[2]);
    features[scale_offset + 7] = verticality;

    // Moments features
    double first_order_first_axis = 0;
    double first_order_second_axis = 0;
    double second_order_first_axis = 0;
    double second_order_second_axis = 0;
    for (auto &p : n_points) {
      Eigen::Vector3d diff = p - medoid;
      first_order_first_axis += diff.dot(eigenvectors.col(2));
      first_order_second_axis += diff.dot(eigenvectors.col(1));
      second_order_first_axis +=
          first_order_first_axis * first_order_first_axis;
      second_order_second_axis +=
          first_order_second_axis * first_order_second_axis;
    }
    features[scale_offset + 8] = first_order_first_axis;
    features[scale_offset + 9] = first_order_second_axis;
    features[scale_offset + 10] = second_order_first_axis;
    features[scale_offset + 11] = second_order_second_axis;

    // Height features
    double z_min = std::numeric_limits<double>::max();
    double z_max = -std::numeric_limits<double>::max();
    for (auto &p : n_points) {
      const double z = p.z();
      if (z < z_min) {
        z_min = z;
      }
      if (z > z_max) {
        z_max = z;
      }
    }

    const double vertical_range = z_max - z_min;
    features[scale_offset + 12] = vertical_range;

    const double height_below = point.z() - z_min;
    features[scale_offset + 13] = height_below;

    const double height_above = z_max - point.z();
    features[scale_offset + 14] = height_above;
  }

  // Color features calculated if initial cloud has colors
  if (has_colors_) {
    const std::size_t geometric_features_offset =
        num_scales_ * features_per_scale;
    Eigen::Vector3d main_color = point_cloud_->colors_[point_id];
    features[geometric_features_offset + 0] = main_color(0);
    features[geometric_features_offset + 1] = main_color(1);
    features[geometric_features_offset + 2] = main_color(2);

    neig_color /= num_neighbors_;
    features[geometric_features_offset + 3] = neig_color(0);
    features[geometric_features_offset + 4] = neig_color(1);
    features[geometric_features_offset + 5] = neig_color(2);
  }
  features =
      features.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.0; });
  return features;
}

Eigen::MatrixXd FeatureEstimator::get_features_for_batch(
    std::size_t start_id, std::size_t end_id) const {
  if (end_id > point_cloud_->points_.size() or
      start_id >= point_cloud_->points_.size() or end_id <= start_id) {
    std::string error_string("invalid slice { ");
    error_string += std::to_string(start_id) + " : ";
    error_string += std::to_string(end_id) + " } for array with { ";
    error_string += std::to_string(point_cloud_->points_.size()) + " } points";
    throw std::out_of_range(error_string);
  }

  const std::size_t range = end_id - start_id;
  const std::size_t num_features = feature_size();
  Eigen::MatrixXd result(range, num_features);
  const double drange =
      static_cast<double>(range) / static_cast<double>(pool_.num_threads());
  const std::size_t num_points_per_thread =
      std::max<std::size_t>(1u, std::ceil(drange));
  const std::size_t max_threads = std::min(pool_.num_threads(), range);
  for (auto i = 0u; i < max_threads; ++i) {
    const auto s = start_id + i * num_points_per_thread;
    const auto e = std::min(end_id, s + num_points_per_thread);
    pool_.add_task([&result, s, e, start_id, this] {
      for (auto j = s; j < e; ++j) {
        result.row(j - start_id) = get_features_for_point(j);
      }
    });
  }
  pool_.wait();

  return result;
}

Eigen::MatrixXd FeatureEstimator::get_features_for_points(
    const std::vector<int> &idxs) const {
  const std::size_t range = idxs.size();
  const std::size_t num_features = feature_size();
  Eigen::MatrixXd result(range, num_features);
  const double drange =
      static_cast<double>(range) / static_cast<double>(pool_.num_threads());
  const std::size_t num_points_per_thread =
      std::max<std::size_t>(1u, std::ceil(drange));
  const std::size_t max_threads = std::min(pool_.num_threads(), range);
  std::vector<std::future<void>> futures;
  for (auto i = 0u; i < max_threads; ++i) {
    const auto s = i * num_points_per_thread;
    const auto e = std::min(range, s + num_points_per_thread);
    futures.push_back(pool_.add_task([&result, &idxs, s, e, this] {
      for (auto j = s; j < e; ++j) {
        result.row(j) = get_features_for_point(idxs[j]);
      }
    }));
  }
  for (auto &&x : futures) {
    x.get();
  }

  return result;
}

std::size_t FeatureEstimator::num_points() const { return num_points_; }

std::vector<unsigned int> FeatureEstimator::soft_voting_smoothing(
    const Eigen::MatrixXd &probabilities, std::size_t num_neighbors) const {
  std::vector<unsigned int> classes(probabilities.rows());

  if (probabilities.size() == 0) {
    return classes;
  }
  const std::size_t num_classes = probabilities.cols();

  const double drange = static_cast<double>(num_points_) /
                        static_cast<double>(pool_.num_threads());
  const std::size_t num_points_per_thread =
      std::max<std::size_t>(1u, std::ceil(drange));
  const std::size_t max_threads = std::min(pool_.num_threads(), num_points_);
  for (auto i = 0u; i < max_threads; ++i) {
    const auto s = i * num_points_per_thread;
    const auto e = std::min(num_points_, s + num_points_per_thread);
    pool_.add_task(
        [&classes, &probabilities, num_neighbors, num_classes, s, e, this] {
          for (auto point_id = s; point_id < e; ++point_id) {
            Eigen::Vector3d point = point_cloud_->points_[point_id];
            const Eigen::Vector3d main_color = point_cloud_->colors_[point_id];
            const auto neighbors = tree_->find_nns(point, num_neighbors);
            float weight_sum = 0.f;
            Eigen::VectorXd commite_result = Eigen::VectorXd::Zero(num_classes);
            for (const auto &neighbor : neighbors) {
              commite_result += probabilities.row(neighbor.first);
            }
            Eigen::VectorXd::Index index;
            commite_result.maxCoeff(&index);
            classes[point_id] = index;
          }
        });
  }
  pool_.wait();

  return classes;
}

std::size_t FeatureEstimator::num_batches() const {
  return static_cast<std::size_t>(
      std::ceil(static_cast<double>(num_points_) / batch_size_));
}

std::size_t FeatureEstimator::batch_size() const { return batch_size_; }

Eigen::MatrixXd FeatureEstimator::get_features_for_batch(
    std::size_t batch_id) const {
  const std::size_t start_id = batch_id * batch_size_;
  const std::size_t end_id =
      std::min<std::size_t>(start_id + batch_size_, num_points_);
  return get_features_for_batch(start_id, end_id);
}

std::size_t FeatureEstimator::feature_size() const {
  if (has_colors_) {
    return color_features + features_per_scale * num_scales_;
  } else {
    return features_per_scale * num_scales_;
  }
}

std::vector<unsigned int> FeatureEstimator::hard_voting_smoothing(
    const std::vector<unsigned int> &labels, std::size_t num_neighbors) const {
  std::vector<unsigned int> smooth_labels(labels.size(), 0);

  if (labels.empty()) {
    return smooth_labels;
  }

  const double drange = static_cast<double>(num_points_) /
                        static_cast<double>(pool_.num_threads());
  const std::size_t num_points_per_thread =
      std::max<std::size_t>(1u, std::ceil(drange));
  const std::size_t max_threads = std::min(pool_.num_threads(), num_points_);
  for (auto i = 0u; i < max_threads; ++i) {
    const auto s = i * num_points_per_thread;
    const auto e = std::min(num_points_, s + num_points_per_thread);
    pool_.add_task([this, s, e, num_neighbors, &smooth_labels, &labels] {
      for (auto point_id = s; point_id < e; ++point_id) {
        Eigen::Vector3d point = point_cloud_->points_[point_id];
        const auto neighbors = tree_->find_nns(point, num_neighbors);
        std::unordered_map<unsigned int, int> commite_result;
        for (const auto &neighbor : neighbors) {
          commite_result[labels[neighbor.first]]++;
        }
        unsigned int max_class = 0;
        int max_value = -1;
        for (auto &v : commite_result) {
          if (v.second > max_value) {
            max_class = v.first;
            max_value = v.second;
          }
        }
        smooth_labels[point_id] = max_class;
      }
    });
  }
  pool_.wait();

  return smooth_labels;
}

}  // namespace pcs
