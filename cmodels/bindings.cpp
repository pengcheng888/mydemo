#include "mnist/bindings_mnist.hpp"
#include "resnet/bindings_resnet.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 统一的绑定模块
PYBIND11_MODULE(_infinidemo, m) {
  m.doc() = "InfiniDemo Python bindings - ResNet and MNIST models for image "
            "classification";

  // 调用各个模块的绑定函数
  infinidemo::models::bind_mnist(m);
  infinidemo::models::bind_resnet_model(m);
  infinidemo::models::bind_resnet_config(m);
}
