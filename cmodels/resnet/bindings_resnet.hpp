#pragma once

#include "configuration_resnet.hpp"
#include "modeling_resnet.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using namespace infinidemo;

namespace infinidemo::models {
inline void bind_resnet_model(py::module_ &m) {
  py::class_<ResNetForImageClassification>(m, "ResNetForImageClassification")
      .def(py::init([](ResNetConfig config) {
             return ResNetForImageClassification(config);
           }),
           py::arg("config"))
      .def(
          "forward",
          [](ResNetForImageClassification &self, infinicore::Tensor &input)
              -> py::object { return py::cast(self.forward(input)); },
          py::arg("input"))
      .def(
          "load_state_dict",
          [](ResNetForImageClassification &self, py::dict _state_dict) -> void {
            std::unordered_map<std::string, infinicore::Tensor> state_dict;
            for (auto item : _state_dict) {
              std::string key = py::cast<std::string>(item.first);
              py::object value_obj =
                  py::reinterpret_borrow<py::object>(item.second);
              state_dict[key] =
                  py::cast<infinicore::Tensor>(value_obj.attr("_underlying"));
            }

            self.load_state_dict(state_dict);
          },
          py::arg("_state_dict"))
      .def("state_dict",
           [](const ResNetForImageClassification &self) -> py::dict {
             std::unordered_map<std::string, infinicore::nn::Parameter>
                 cpp_state_dict = self.state_dict();

             py::dict py_state_dict;
             for (const auto &pair : cpp_state_dict) {
               const std::string &key = pair.first;
               const infinicore::Tensor &value = pair.second;
               py_state_dict[key.c_str()] = py::cast(value);
             }

             return py_state_dict;
           })
      .def("__repr__",
           [](const ResNetForImageClassification &self) {
             return "<ResNetForImageClassification>";
           })
      .def(
          "to",
          [](ResNetForImageClassification &self, infinicore::Device device) {
            infinidemo::nn::modules::Module &ref = self;
            ref.to_device_(device);
            infinicore::context::syncDevice();
            return self;
          },
          py::arg("device"));
}

inline void bind_resnet_config(py::module_ &m) {
  py::class_<ResNetConfig>(m, "ResNetConfig")
      .def(py::init<>())
      .def_readwrite("architectures", &ResNetConfig::architectures)
      .def_readwrite("depths", &ResNetConfig::depths)
      .def_readwrite("downsample_in_first_stage",
                     &ResNetConfig::downsample_in_first_stage)
      .def_readwrite("downsample_in_bottleneck",
                     &ResNetConfig::downsample_in_bottleneck)
      .def_readwrite("embedding_size", &ResNetConfig::embedding_size)
      .def_readwrite("hidden_act", &ResNetConfig::hidden_act)
      .def_readwrite("hidden_sizes", &ResNetConfig::hidden_sizes)
      .def_readwrite("layer_type", &ResNetConfig::layer_type)
      .def_readwrite("model_type", &ResNetConfig::model_type)
      .def_readwrite("num_channels", &ResNetConfig::num_channels)
      .def_readwrite("torch_dtype", &ResNetConfig::torch_dtype)
      .def_readwrite("transformers_version",
                     &ResNetConfig::transformers_version)
      .def_readwrite("num_labels", &ResNetConfig::num_labels)
      .def("__repr__", [](const ResNetConfig &self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });
}
} // namespace infinidemo::models
