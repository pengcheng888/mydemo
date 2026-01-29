#pragma once

#include "../functional/avg_pool2d_op.hpp"
#include "../functional/max_pool2d_op.hpp"
#include "../utils.hpp"
#include "module.hpp"
#include <cmath>
#include <cstddef>
#include <infinicore/device.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>
#include <stdexcept>
#include <vector>

namespace infinidemo::nn::modules {
using namespace infinicore;

class AvgPool2d : public infinidemo::nn::modules::Module {
public:
  AvgPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0,
            bool ceil_mode = false, const DataType &dtype = DataType::F32)
      : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
        padding_(padding), ceil_mode_(ceil_mode), dtype_(dtype) {}

  inline Tensor forward(Tensor &input) const {

    Device original_device = input->device();
    if ((original_device.getType() == Device::Type::HYGON) ||
        (original_device.getType() == Device::Type::MOORE)) {
      input = input->to(Device::cpu());
    }

    int kernel_h = static_cast<int>(kernel_size_);
    int kernel_w = static_cast<int>(kernel_size_);
    int stride_h = static_cast<int>(stride_);
    int stride_w = static_cast<int>(stride_);
    int padding_h = static_cast<int>(padding_);
    int padding_w = static_cast<int>(padding_);
    int dilation_h = static_cast<int>(1);
    int dilation_w = static_cast<int>(1);

    std::vector<size_t> output_shape = computePool2dOutputShape(
        input->shape(), kernel_h, kernel_w, stride_h, stride_w, padding_h,
        padding_w, dilation_h, dilation_w, ceil_mode_);

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    INFINICORE_CHECK_ERROR(infinidemo::nn::functional::performAvgPool2d(
        input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h,
        padding_w, dilation_h, dilation_w, ceil_mode_, input->device()));

    if ((original_device.getType() == Device::Type::HYGON) ||
        (original_device.getType() == Device::Type::MOORE)) {
      output = output->to(original_device);
    }
    return output;
  }

private:
  void to_device_(const Device &device) override { device_ = device; }

protected:
  // 计算Pool2D输出形状
  // input_shape = [N, C, H, W]
  // 输出: [N, C, OH, OW]
  std::vector<size_t>
  computePool2dOutputShape(const std::vector<size_t> &input_shape, int kernel_h,
                           int kernel_w, int stride_h, int stride_w,
                           int padding_h, int padding_w, int dilation_h,
                           int dilation_w, bool ceil_mode) const {

    std::vector<size_t> output_shape;
    output_shape.push_back(input_shape[0]); // N
    output_shape.push_back(input_shape[1]); // C

    // 计算输出高度和宽度
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];

    // 计算有效kernel大小
    size_t effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
    size_t effective_kernel_w = dilation_w * (kernel_w - 1) + 1;

    // 计算padded输入大小
    size_t padded_h = input_h + 2 * padding_h;
    size_t padded_w = input_w + 2 * padding_w;

    // 计算输出大小
    size_t output_h, output_w;
    if (ceil_mode) {
      output_h = static_cast<size_t>(std::ceil(
          static_cast<double>(padded_h - effective_kernel_h) / stride_h + 1));
      output_w = static_cast<size_t>(std::ceil(
          static_cast<double>(padded_w - effective_kernel_w) / stride_w + 1));
    } else {
      output_h = (padded_h - effective_kernel_h) / stride_h + 1;
      output_w = (padded_w - effective_kernel_w) / stride_w + 1;
    }

    output_shape.push_back(output_h);
    output_shape.push_back(output_w);

    return output_shape;
  }

protected:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
  size_t dilation_;
  bool ceil_mode_;
  DataType dtype_;
  Device device_ = Device::cpu();
};

class MaxPool2d : public infinidemo::nn::modules::Module {
public:
  MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0,
            size_t dilation = 1, bool ceil_mode = false,
            const DataType &dtype = DataType::F32)
      : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
        padding_(padding), dilation_(dilation), ceil_mode_(ceil_mode),
        dtype_(dtype) {}

  inline Tensor forward(Tensor &input) const {
    Device original_device = input->device();
    if ((original_device.getType() == Device::Type::HYGON) ||
        (original_device.getType() == Device::Type::MOORE)) {
      input = input->to(Device::cpu());
    }

    int kernel_h = static_cast<int>(kernel_size_);
    int kernel_w = static_cast<int>(kernel_size_);
    int stride_h = static_cast<int>(stride_);
    int stride_w = static_cast<int>(stride_);
    int padding_h = static_cast<int>(padding_);
    int padding_w = static_cast<int>(padding_);
    int dilation_h = static_cast<int>(dilation_);
    int dilation_w = static_cast<int>(dilation_);

    std::vector<size_t> output_shape = computePool2dOutputShape(
        input->shape(), kernel_h, kernel_w, stride_h, stride_w, padding_h,
        padding_w, dilation_h, dilation_w, ceil_mode_);

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    INFINICORE_CHECK_ERROR(infinidemo::nn::functional::performMaxPool2d(
        input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h,
        padding_w, dilation_h, dilation_w, ceil_mode_, input->device()));

    if ((original_device.getType() == Device::Type::HYGON) ||
        (original_device.getType() == Device::Type::MOORE)) {
      output = output->to(original_device);
    }
    return output;
  }

private:
  void to_device_(const Device &device) override { device_ = device; }

protected:
  // 计算Pool2D输出形状
  // input_shape = [N, C, H, W]
  // 输出: [N, C, OH, OW]
  std::vector<size_t>
  computePool2dOutputShape(const std::vector<size_t> &input_shape, int kernel_h,
                           int kernel_w, int stride_h, int stride_w,
                           int padding_h, int padding_w, int dilation_h,
                           int dilation_w, bool ceil_mode) const {

    std::vector<size_t> output_shape;
    output_shape.push_back(input_shape[0]); // N
    output_shape.push_back(input_shape[1]); // C

    // 计算输出高度和宽度
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];

    // 计算有效kernel大小
    size_t effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
    size_t effective_kernel_w = dilation_w * (kernel_w - 1) + 1;

    // 计算padded输入大小
    size_t padded_h = input_h + 2 * padding_h;
    size_t padded_w = input_w + 2 * padding_w;

    // 计算输出大小
    size_t output_h, output_w;
    if (ceil_mode) {
      output_h = static_cast<size_t>(std::ceil(
          static_cast<double>(padded_h - effective_kernel_h) / stride_h + 1));
      output_w = static_cast<size_t>(std::ceil(
          static_cast<double>(padded_w - effective_kernel_w) / stride_w + 1));
    } else {
      output_h = (padded_h - effective_kernel_h) / stride_h + 1;
      output_w = (padded_w - effective_kernel_w) / stride_w + 1;
    }

    output_shape.push_back(output_h);
    output_shape.push_back(output_w);

    return output_shape;
  }

protected:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
  size_t dilation_;
  bool ceil_mode_;
  DataType dtype_;
  Device device_ = Device::cpu();
};
} // namespace infinidemo::nn::modules
