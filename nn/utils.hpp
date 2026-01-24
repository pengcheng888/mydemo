#pragma once

#include "functional/add_op.hpp"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define INFINICORE_CHECK_ERROR(call)                                           \
  do {                                                                         \
    infiniStatus_t ret = (call);                                               \
    if (ret != INFINI_STATUS_SUCCESS) {                                        \
      throw std::runtime_error(                                                \
          "`" #call "` failed with error: " + std::to_string(int(ret)) +       \
          " from " + std::string(__func__) + " at " + std::string(__FILE__) +  \
          ":" + std::to_string(__LINE__) + ".");                               \
    }                                                                          \
  } while (false)

namespace infinicore {

// [[maybe_unused]] static Tensor operator+(const Tensor &lhs, const Tensor
// &rhs) {
//   return lhs;
// }

// 类外部实现
[[maybe_unused]] static Tensor &operator+=(Tensor &lhs, const Tensor &rhs) {
  INFINICORE_CHECK_ERROR(
      infinidemo::nn::functional::performAdd(lhs, lhs, rhs, lhs->device()));
  return lhs;
}

}; // namespace infinicore

[[maybe_unused]] static void printShape(const std::vector<ptrdiff_t> &shape,
                                        const std::string name = "") {
  if (name != "") {
    std::cout << name << ": [";
  } else {
    std::cout << "[";
  }
  for (size_t i = 0; i < shape.size(); i++) {
    std::cout << shape[i];
    if (i < shape.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]";
  if (name != "") {
    std::cout << std::endl;
  }
}

[[maybe_unused]] static void printShape(const std::vector<size_t> &shape,
                                        const std::string name = "") {
  if (name != "") {
    std::cout << name << ": [";
  } else {
    std::cout << "[";
  }
  for (size_t i = 0; i < shape.size(); i++) {
    std::cout << shape[i];
    if (i < shape.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]";
  if (name != "") {
    std::cout << std::endl;
  }
}
