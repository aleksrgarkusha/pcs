#include "catch.hpp"

#include <random>

#include <feature_estimator.h>

TEST_CASE("Feature estimator tests") {
  auto point_cloud = std::make_shared<pcs::PointCloud>();
  SECTION("Empty point cloud") {
    pcs::FeatureEstimator estimator(point_cloud, 0.05, 10, 2);
    CHECK_THROWS_AS(estimator.get_features_for_point(1), std::out_of_range);
  }

  SECTION("Random point cloud") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    std::uniform_int_distribution<> dis_u(0, 255);

    const std::size_t num_samples = 100u;
    for (auto j = 0u; j < num_samples; ++j) {
      Eigen::Vector3d point(dis(gen), dis(gen), dis(gen));
      Eigen::Vector3d color(dis_u(gen), dis_u(gen), dis_u(gen));
      point_cloud->add_point_and_color(point, color);
    }

    pcs::FeatureEstimator estimator(point_cloud, 0.05, 5, 2, 2);
    CHECK(estimator.batch_size() == 2);
    CHECK(estimator.num_batches() == 50);
    CHECK(estimator.feature_size() == 36);

    CHECK_NOTHROW(estimator.get_features_for_batch(0));
    CHECK_THROWS_AS(estimator.get_features_for_batch(51), std::out_of_range);

    CHECK_NOTHROW(estimator.get_features_for_point(1));
    CHECK_THROWS_AS(estimator.get_features_for_point(-1), std::out_of_range);

    CHECK_NOTHROW(estimator.get_features_for_batch(0, 10));
    CHECK_THROWS_AS(estimator.get_features_for_batch(10, 2), std::out_of_range);

    CHECK_NOTHROW(estimator.get_features_for_points({0, 1, 2}));
    CHECK_THROWS_AS(estimator.get_features_for_points({0, 1, 101}),
                    std::out_of_range);

    CHECK(estimator.num_points() == 100);

    Eigen::MatrixXd probs;
    const std::size_t num_classes = 5;
    probs.resize(num_samples, num_classes);
    for (auto j = 0u; j < probs.rows(); ++j) {
      for (auto i = 0u; i < num_classes; ++i) {
        probs(j, i) = static_cast<double>(dis(gen)) / 100.0f;
      }
    }
    CHECK(estimator.soft_voting_smoothing(probs, 3).size() == num_samples);
    CHECK(estimator.soft_voting_smoothing(Eigen::MatrixXd(), 3).size() == 0);

    CHECK(estimator.hard_voting_smoothing(std::vector<unsigned int>(), 3)
              .size() == 0);
    std::vector<unsigned int> labels;
    for (auto i = 0u; i < num_samples; ++i) {
      labels.push_back(0);
    }
    CHECK(estimator.hard_voting_smoothing(labels, 3).size() == num_samples);
  }
}
