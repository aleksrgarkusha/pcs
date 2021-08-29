#include <numeric>
#include <unordered_map>
#include <vector>

#include "point_cloud.h"

namespace hash_eigen {

template <typename T>
struct hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    size_t seed = 0;
    for (int i = 0; i < (int)matrix.size(); i++) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

}  // namespace hash_eigen

namespace pcs {

namespace {
class AccumulatedPoint {
 public:
  AccumulatedPoint()
      : num_of_points_(0), point_(0.0, 0.0, 0.0), color_(0.0, 0.0, 0.0) {}

  void add_point(const Eigen::Vector3d& point, const Eigen::Vector3d& color) {
    point_ += point;
    color_ += color;
    num_of_points_++;
  }

  void add_point(const Eigen::Vector3d& point) {
    point_ += point;
    num_of_points_++;
  }

  Eigen::Vector3d get_average_point() const {
    return point_ / double(num_of_points_);
  }

  Eigen::Vector3d get_average_color() const {
    return color_ / double(num_of_points_);
  }

 private:
  int num_of_points_;
  Eigen::Vector3d point_;
  Eigen::Vector3d color_;
};
}  // namespace

std::shared_ptr<PointCloud> voxel_down_sample(const PointCloud& input,
                                              double voxel_size) {
  auto output = std::make_shared<PointCloud>();
  if (voxel_size <= 0.0) {
    return output;
  }

  Eigen::Vector3d voxel_size3(voxel_size, voxel_size, voxel_size);
  Eigen::Vector3d voxel_min_bound = input.get_min_bound() - voxel_size3 * 0.5;
  Eigen::Vector3d voxel_max_bound = input.get_max_bound() + voxel_size3 * 0.5;
  if (voxel_size * std::numeric_limits<int>::max() <
      (voxel_max_bound - voxel_min_bound).maxCoeff()) {
    return output;
  }

  std::unordered_map<Eigen::Vector3i, AccumulatedPoint,
                     hash_eigen::hash<Eigen::Vector3i>>
      voxelindex_to_accpoint;

  const bool has_colors = input.has_colors();
  Eigen::Vector3d ref_coord;
  Eigen::Vector3i voxel_index;
  for (int i = 0; i < (int)input.points_.size(); i++) {
    ref_coord = (input.points_[i] - voxel_min_bound) / voxel_size;
    voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
        int(floor(ref_coord(2)));
    if (has_colors) {
      voxelindex_to_accpoint[voxel_index].add_point(input.points_[i],
                                                    input.colors_[i]);
    } else {
      voxelindex_to_accpoint[voxel_index].add_point(input.points_[i]);
    }
  }

  for (auto accpoint : voxelindex_to_accpoint) {
    output->points_.push_back(accpoint.second.get_average_point());
    if (has_colors) {
      output->colors_.push_back(accpoint.second.get_average_color());
    }
  }
  return output;
}

}  // namespace pcs