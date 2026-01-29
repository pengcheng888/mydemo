#pragma once

#include <infinicore/device.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>
#include <string>

#include "../../nn/modules/conv.hpp"
#include "../../nn/modules/linear.hpp"
#include "../../nn/modules/module.hpp"
#include "../../nn/modules/relu.hpp"

using namespace infinicore;

namespace infinidemo::models {
class MnistForImageClassification : public infinidemo::nn::modules::Module {
public:
    MnistForImageClassification();
    Tensor forward(Tensor &input) const;

private:
    void to_device_(const Device &device) override {
        ;
    }

protected:
    INFINICORE_NN_MODULE(infinidemo::nn::modules::Linear, fc1);
    INFINICORE_NN_MODULE(infinidemo::nn::modules::Conv2d, conv1);
    infinidemo::nn::modules::ReLU relu_;

protected:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    DataType dtype_;
};

} // namespace infinidemo::models
