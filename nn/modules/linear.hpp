#pragma once

#include "../functional/gemm_op.hpp"
#include "../utils.hpp"
#include "module.hpp"
#include <infinicore/device.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <stdexcept>
#include <string>

namespace infinidemo::nn::modules {
using namespace infinicore;

class Identity : public infinidemo::nn::modules::Module {
public:
  Identity() = default;
  inline Tensor forward(const Tensor &input) const { return input; }

private:
  void to_device_(const Device &device) override {}
};

class Linear : public infinidemo::nn::modules::Module {
public:
  Linear(size_t in_features, size_t out_features, bool bias = true,
         const DataType &dtype = DataType::F32)
      : in_features_(in_features), out_features_(out_features), has_bias_(bias),
        dtype_(dtype) {
    INFINICORE_NN_PARAMETER_INIT(
        weight, ({out_features, in_features}, dtype_, device_));

    if (bias) {
      INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device_));
    } else {
      bias_ =
          infinicore::nn::Parameter(); // Default constructed empty parameter
    }
  }

  inline Tensor forward(Tensor &input) const {
    Size ndim = input->ndim();
    Size out_features = weight_->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    float alpha = 1.0f;
    float beta = 0.0f;
    if (has_bias_) {
      beta = 1.0f;
      auto new_bias = bias_->as_strided(output->shape(), {0, 1});
      output->copy_from(new_bias);
    }
    INFINICORE_CHECK_ERROR(infinidemo::nn::functional::performGemm(
        output, input, weight_->permute({1, 0}), alpha, beta, input->device()));

    return output;
  }

private:
  void to_device_(const Device &device) override {
    Tensor &weight_ref = weight_;
    weight_ = weight_ref->to(device);
    if (has_bias_) {
      Tensor &bias_ref = bias_;
      bias_ = bias_ref->to(device);
    }
    device_ = device;
  }

protected:
  INFINICORE_NN_PARAMETER(weight);
  INFINICORE_NN_PARAMETER(bias);

protected:
  size_t in_features_;
  size_t out_features_;
  bool has_bias_;
  DataType dtype_;
  Device device_ = Device::cpu();
};

} // namespace infinidemo::nn::modules
