#pragma once

#include "../functional/relu_op.hpp"
#include "../utils.hpp"
#include "module.hpp"
#include <infinicore/device.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

namespace infinidemo::nn::modules {
using namespace infinicore;

class ReLU : public infinidemo::nn::modules::Module {
public:
    ReLU() = default;
    inline Tensor forward(const Tensor &input) const {
        auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
        INFINICORE_CHECK_ERROR(infinidemo::nn::functional::performRelu(output, input, input->device()));
        return output;
    }

private:
    void to_device_(const Device &device) override {}
};

} // namespace infinidemo::nn::modules
