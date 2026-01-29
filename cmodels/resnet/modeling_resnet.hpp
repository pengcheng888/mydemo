#pragma once

#include "../../nn/modules/flatten.hpp"
#include "../../nn/modules/linear.hpp"
#include "../../nn/modules/module.hpp"
#include "configuration_resnet.hpp"
#include <infinicore/device.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>
#include <stdexcept>
#include <string>

namespace infinidemo::models {

using namespace infinicore;
class ResNetModel;

class ResNetForImageClassification : public infinidemo::nn::modules::Module {
public:
    ResNetForImageClassification(const ResNetConfig &config);
    Tensor forward(Tensor &pixel_values);

protected:
    void to_device_(const Device &device) override;

protected:
    INFINICORE_NN_MODULE(ResNetModel, resnet);
    INFINICORE_NN_MODULE_VEC(infinidemo::nn::modules::Linear, classifier);
    infinidemo::nn::modules::Flatten flatten_;
    ResNetConfig config_;
    int num_labels_;
};

} // namespace infinidemo::models
