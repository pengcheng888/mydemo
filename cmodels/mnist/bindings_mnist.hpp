#pragma once

#include "modeling_mnist.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

namespace py = pybind11;
using namespace infinidemo;

namespace infinidemo::models {
// 绑定 MnistForImageClassification 类到 Python 模块
inline void bind_mnist(py::module_ &m) {
  // 注意：由于 infinidemo::nn::modules::Module 在另一个模块中定义，我们无法在 pybind11
  // 中直接声明继承关系 但 C++ 中的继承关系仍然存在，功能不受影响
  py::class_<MnistForImageClassification>(m, "MnistForImageClassification")
      // 构造函数重载1: 只有必需参数（使用默认的bias, dtype, device）
      .def(py::init([]() { return new MnistForImageClassification(); }),
           R"doc(
                MNIST model for image classification.

                Args:
                    in_features: Number of input features
                    out_features: Number of output features (number of classes)

                Example:
                    >>> import _infinidemo
                    >>> model = _infinidemo.MnistForImageClassification(512, 1000)
                )doc")
      .def(
          "forward",
          [](MnistForImageClassification &self,
             py::object input_obj) -> py::object {
            // 尝试将 Python Tensor 对象转换为 C++ Tensor
            // 支持两种方式：直接转换或通过 _underlying 属性
            infinicore::Tensor input;
            bool converted = false;

            // 方法1: 尝试通过 _underlying 属性访问（如果存在）
            if (py::hasattr(input_obj, "_underlying")) {
              try {
                py::object underlying = input_obj.attr("_underlying");
                input = py::cast<infinicore::Tensor>(underlying);
                converted = true;
              } catch (...) {
                // 如果 _underlying 转换失败，继续尝试其他方法
              }
            }

            // 方法2: 直接转换 Python Tensor 对象
            if (!converted) {
              try {
                input = py::cast<infinicore::Tensor>(input_obj);
                converted = true;
              } catch (const py::cast_error &) {
                // 转换失败
              }
            }

            if (!converted) {
              std::string error_msg =
                  "Cannot convert Python Tensor to C++ Tensor. ";
              error_msg +=
                  "Please ensure the input is an infinicore.Tensor object.";
              throw std::runtime_error(error_msg);
            }

            // forward 方法需要非const引用
            infinicore::Tensor &input_ref = input;
            infinicore::Tensor result = self.forward(input_ref);

            // 返回结果
            return py::cast(result);
          },
          py::arg("input"),
          R"doc(
                Forward pass through the MNIST model.

                Args:
                    input: Input tensor of shape (batch_size, in_features) from infinicore

                Returns:
                    Output tensor of shape (batch_size, out_features)
                )doc")
      .def(
          "load_state_dict",
          [](MnistForImageClassification &self, py::dict _state_dict) -> void {
            // Load state dictionary into the model
            // Convert Python dict to C++ unordered_map and load parameters
            std::unordered_map<std::string, infinicore::Tensor> state_dict;

            // Convert Python dict to C++ unordered_map
            for (auto item : _state_dict) {
              std::string key = py::cast<std::string>(item.first);
              py::object value_obj =
                  py::reinterpret_borrow<py::object>(item.second);

              // Convert Python Tensor to C++ Tensor
              infinicore::Tensor tensor_value;
              bool converted = false;

              // Method 1: Try through _underlying attribute
              if (py::hasattr(value_obj, "_underlying")) {
                try {
                  py::object underlying = value_obj.attr("_underlying");
                  tensor_value = py::cast<infinicore::Tensor>(underlying);
                  converted = true;
                } catch (...) {
                  // Continue to next method
                }
              }

              // Method 2: Direct cast
              if (!converted) {
                try {
                  tensor_value = py::cast<infinicore::Tensor>(value_obj);
                  converted = true;
                } catch (const py::cast_error &) {
                  // Conversion failed
                }
              }

              if (!converted) {
                throw std::runtime_error("Cannot convert value for key '" +
                                         key + "' to Tensor");
              }

              state_dict[key] = tensor_value;
            }

            // Now use the converted state_dict
            self.load_state_dict(state_dict);
          },
          py::arg("_state_dict"),
          R"doc(
                Load state dictionary into the model.

                Args:
                    _state_dict: Dictionary containing model parameters
                                Maps parameter names (string) to Tensor values from infinicore
                )doc")
      .def(
          "state_dict",
          [](const MnistForImageClassification &self) -> py::dict {
            // Get state dictionary from the model
            // Call the C++ state_dict method and convert to Python dict
            std::unordered_map<std::string, infinicore::nn::Parameter>
                cpp_state_dict = self.state_dict();

            // Convert C++ unordered_map to Python dict
            py::dict py_state_dict;
            for (const auto &pair : cpp_state_dict) {
              const std::string &key = pair.first;
              const infinicore::Tensor &value = pair.second;

              // Convert C++ Tensor to Python Tensor
              py_state_dict[key.c_str()] = py::cast(value);
            }

            return py_state_dict;
          },
          R"doc(
                Get state dictionary from the model.

                Returns:
                    Dictionary containing model parameters
                    Maps parameter names (string) to Tensor values from infinicore
                    
                Example:
                    >>> model = _infinidemo.MnistForImageClassification()
                    >>> state_dict = model.state_dict()
                    >>> print(state_dict)
                )doc")
      .def("__repr__", [](const MnistForImageClassification &self) {
        return "<MnistForImageClassification>";
      });
}
} // namespace infinidemo::models
