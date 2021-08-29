#include "catch.hpp"

#include <random>

#include <kd_tree.hpp>

TEST_CASE("KDtree tests" ) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);

  std::vector<Eigen::Vector3d> points;
  const std::size_t num_samples = 100u;
  for(auto j = 0u; j < num_samples; ++j) {
    Eigen::Vector3d point(dis(gen), dis(gen), dis(gen));
    points.push_back(point);
  }

  pcs::KDTree<3> tree(points);
  for(auto j = 0u; j < num_samples; ++j) {
    auto self = tree.find_nn(points[j]);
    CHECK(self.first == j);
  }
}
