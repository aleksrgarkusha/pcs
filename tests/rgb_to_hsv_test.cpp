#include "catch.hpp"

#include <rgb_to_hsv.hpp>

TEST_CASE("Rgb to hsv tests") {
  Eigen::Vector3d white(255.0, 255.0, 255.0);
  CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(white)) == white);
  Eigen::Vector3d black(0.0, 0.0, 0.0);
  CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(black)) == black);
  Eigen::Vector3d mean(127.0, 127.0, 127.0);
  CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(mean)) == mean);
  Eigen::Vector3d r(255.0, 0.0, 0.0);
  CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(r)) == r);
  Eigen::Vector3d g(0.0, 255.0, 0.0);
  CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(g)) == g);
  Eigen::Vector3d b(0.0, 0.0, 255.0);
  CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(b)) == b);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis_u(0, 255);
  const std::size_t num_samples = 100u;
  for (auto j = 0u; j < num_samples; ++j) {
    Eigen::Vector3d color(dis_u(gen), dis_u(gen), dis_u(gen));
    CHECK(pcs::hsv_to_rgb(pcs::rgb_to_hsv(color)) == color);
  }
}