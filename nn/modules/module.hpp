#pragma once
#include <infinicore/context/context.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/parameter.hpp>
#include <infinicore/tensor.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinidemo::nn::modules {
using namespace infinicore;
class Module : public infinicore::nn::Module {
public:
  virtual void to_device_(const Device &device) = 0;
};
} // namespace infinidemo::nn::modules