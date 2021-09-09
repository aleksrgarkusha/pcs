#pragma once
#include <cmath>
#include <limits>

#include <Eigen/Dense>

namespace pcs {

template <typename Derived>
Eigen::Matrix<typename Eigen::DenseBase<Derived>::Scalar, 3, 1> hsv_to_rgb(
    const Eigen::DenseBase<Derived>& hsv) {
  using T = typename Eigen::DenseBase<Derived>::Scalar;
  T hh, p, q, t, ff;
  long i;
  T h = hsv.x();
  T s = hsv.y();
  T v = hsv.z();

  Eigen::Matrix<T, 3, 1> rgb;

  if (s == 0) {
    rgb.x() = std::round(v * 255);
    rgb.y() = std::round(v * 255);
    rgb.z() = std::round(v * 255);
    return rgb;
  }

  hh = h;
  if (hh >= 360.0) {
    hh = 0.f;
  }
  hh /= 60.0;
  i = static_cast<long>(hh);
  ff = hh - i;
  p = v * (1.0 - s);
  q = v * (1.0 - (s * ff));
  t = v * (1.0 - (s * (1.0 - ff)));

  switch (i) {
    case 0:
      rgb.x() = std::round(v * 255);
      rgb.y() = std::round(t * 255);
      rgb.z() = std::round(p * 255);
      break;
    case 1:
      rgb.x() = std::round(q * 255);
      rgb.y() = std::round(v * 255);
      rgb.z() = std::round(p * 255);
      break;
    case 2:
      rgb.x() = std::round(p * 255);
      rgb.y() = std::round(v * 255);
      rgb.z() = std::round(t * 255);
      break;
    case 3:
      rgb.x() = std::round(p * 255);
      rgb.y() = std::round(q * 255);
      rgb.z() = std::round(v * 255);
      break;
    case 4:
      rgb.x() = std::round(t * 255);
      rgb.y() = std::round(p * 255);
      rgb.z() = std::round(v * 255);
      break;
    case 5:
    default:
      rgb.x() = std::round(v * 255);
      rgb.y() = std::round(p * 255);
      rgb.z() = std::round(q * 255);
      break;
  }

  return rgb;
}

template <typename Derived>
Eigen::Matrix<typename Eigen::DenseBase<Derived>::Scalar, 3, 1> rgb_to_hsv(
    const Eigen::DenseBase<Derived>& rgb) {
  using T = typename Eigen::DenseBase<Derived>::Scalar;
  T r = rgb.x() / 255.0;
  T g = rgb.y() / 255.0;
  T b = rgb.z() / 255.0;
  Eigen::Matrix<T, 3, 1> hsv;
  T vmin, diff;
  hsv.z() = vmin = r;
  if (hsv.z() < g) hsv.z() = g;
  if (hsv.z() < b) hsv.z() = b;
  if (vmin > g) vmin = g;
  if (vmin > b) vmin = b;

  diff = hsv.z() - vmin;
  hsv.y() = diff / (std::abs(hsv.z()) + std::numeric_limits<T>::epsilon());
  diff = 60.0 / (diff + std::numeric_limits<T>::epsilon());
  if (hsv.z() == r) {
    hsv.x() = (g - b) * diff;
  } else if (hsv.z() == g) {
    hsv.x() = (b - r) * diff + 120.0;
  } else {
    hsv.x() = (r - g) * diff + 240.0;
  }

  if (hsv.x() < 0) {
    hsv.x() += 360.0;
  }

  return hsv;
}

}  // namespace pcs
