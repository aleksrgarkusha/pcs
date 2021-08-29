#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <feature_estimator.h>
#include <point_cloud.h>

namespace py = pybind11;

PYBIND11_MODULE(pypcs, m) {
  m.doc() = "Python bindings for pcs module";

  py::class_<pcs::PointCloud, std::shared_ptr<pcs::PointCloud>>(m, "PointCloud")
      .def(py::init())
      .def(py::init<const Eigen::MatrixXd>(), py::arg("points"))
      .def(py::init<const Eigen::MatrixXd, const Eigen::MatrixXd>(),
           py::arg("points"), py::arg("colors"))
      .def(py::init<const Eigen::MatrixXd>(), py::arg("points"))
      .def("add_point_and_color", &pcs::PointCloud::add_point_and_color,
           py::arg("point"), py::arg("color"))
      .def("add_point", &pcs::PointCloud::add_point)
      .def("get_point", &pcs::PointCloud::get_point)
      .def("get_color", &pcs::PointCloud::get_color)
      .def("get_points", &pcs::PointCloud::get_points)
      .def("get_colors", &pcs::PointCloud::get_colors)
      .def("get_point_and_color", &pcs::PointCloud::get_point_and_color)
      .def("__getitem__", [](const pcs::PointCloud& self,
                             int idx) { return self.get_point_and_color(idx); })
      .def("__len__",
           [](const pcs::FeatureEstimator& self) { return self.num_points(); })
      .def("num_points", &pcs::PointCloud::num_points)
      .def("clear", &pcs::PointCloud::clear)
      .def("empty", &pcs::PointCloud::empty)
      .def("get_min_bound", &pcs::PointCloud::get_min_bound)
      .def("get_max_bound", &pcs::PointCloud::get_max_bound)
      .def("has_points", &pcs::PointCloud::has_points)
      .def("has_colors", &pcs::PointCloud::has_colors)
      .def("estimate_mean_distance", &pcs::PointCloud::estimate_mean_distance);

  py::class_<pcs::FeatureEstimator>(m, "FeatureEstimator")
      .def(py::init<std::shared_ptr<pcs::PointCloud>, double, int, int, int>(),
           py::arg("cloud"), py::arg("voxel_size") = 0.05,
           py::arg("num_neighbors") = 10, py::arg("num_scales") = 9,
           py::arg("batch_size") = 1000)
      .def("num_points", &pcs::FeatureEstimator::num_points)
      .def("batch_size", &pcs::FeatureEstimator::batch_size)
      .def("feature_size", &pcs::FeatureEstimator::feature_size)
      .def("get_features_for_point",
           &pcs::FeatureEstimator::get_features_for_point)
      .def("get_features_for_points",
           &pcs::FeatureEstimator::get_features_for_points)
      .def("get_features_for_batch",
           (Eigen::MatrixXd(pcs::FeatureEstimator::*)(std::size_t) const) &
               pcs::FeatureEstimator::get_features_for_batch)
      .def("soft_voting_smoothing",
           &pcs::FeatureEstimator::soft_voting_smoothing, py::arg("probs"),
           py::arg("num_neighbors") = 10)
      .def("hard_voting_smoothing",
           &pcs::FeatureEstimator::hard_voting_smoothing, py::arg("labels"),
           py::arg("num_neighbors") = 10)
      .def("__getitem__",
           [](const pcs::FeatureEstimator& self, int index) {
             return self.get_features_for_batch(index);
           })
      .def("__len__", [](const pcs::FeatureEstimator& self) {
        return self.num_batches();
      });
}