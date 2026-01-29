#pragma once

#include "../functional/conv_op.hpp"
#include "../utils.hpp"
#include "module.hpp"
#include <cstddef>
#include <infinicore/device.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace infinidemo::nn::modules {
using namespace infinicore;

class Conv2d : public infinidemo::nn::modules::Module {
public:
    Conv2d(int in_channels, int out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0, size_t dilation1 = 1,
           int groups = 1, bool bias = true, const DataType &dtype = DataType::F32)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding),
          dilation_(dilation1), groups_(groups), has_bias_(bias), dtype_(dtype) {

        INFINICORE_NN_PARAMETER_INIT(weight, ({static_cast<size_t>(out_channels), static_cast<size_t>(in_channels), kernel_size, kernel_size}, dtype_, device_));
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({static_cast<size_t>(out_channels)}, dtype_, device_));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
    }

    inline Tensor forward(Tensor &input) const {
        std::vector<size_t> pads = {padding_, padding_};
        std::vector<ptrdiff_t> strides = {static_cast<ptrdiff_t>(stride_), static_cast<ptrdiff_t>(stride_)};
        std::vector<size_t> dilations = {dilation_, dilation_};
        std::vector<size_t> output_shape = computeConv2dOutputShape(input->shape(), weight_->shape(), pads, strides, dilations);

        auto output = Tensor::empty(output_shape, input->dtype(), input->device());
        INFINICORE_CHECK_ERROR(infinidemo::nn::functional::performConv2D(
            output, input, weight_, bias_, strides, pads, dilations, input->device()));

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
    // 计算2D卷积输出形状
    // x_shape = [N, C, H, W], w_shape = [OC, IC, KH, KW]
    // 输出: [N, OC, OH, OW]
    std::vector<size_t> computeConv2dOutputShape(
        const std::vector<size_t> &x_shape, const std::vector<size_t> &w_shape,
        const std::vector<size_t> &pads, const std::vector<ptrdiff_t> &strides,
        const std::vector<size_t> &dilations) const {

        size_t spatial_dims = pads.size();
        std::vector<size_t> output_shape;
        output_shape.push_back(x_shape[0]); // N
        output_shape.push_back(w_shape[0]); // OC

        // 计算每个空间维度的输出大小
        for (size_t i = 0; i < spatial_dims; i++) {
            size_t input_size = x_shape[i + 2];
            size_t kernel_size = w_shape[i + 2];
            size_t pad = pads[i];
            size_t stride = static_cast<size_t>(strides[i]); // Convert ptrdiff_t to size_t
            size_t dilation = dilations[i];

            size_t effective_kernel = dilation * (kernel_size - 1) + 1;
            size_t padded_input = input_size + 2 * pad;
            size_t output_size = (padded_input - effective_kernel) / stride + 1;
            output_shape.push_back(output_size);
        }
        return output_shape;
    }

protected:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
    int in_channels_;
    int out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t dilation_;
    int groups_;
    bool has_bias_;
    DataType dtype_;
    Device device_ = Device::cpu();
};

} // namespace infinidemo::nn::modules
