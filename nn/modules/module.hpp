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
    virtual ~Module() = default;
    virtual void to_device_(const Device &device) = 0;
    void to(const Device &device) {
        to_recursively(device);
        infinicore::context::syncDevice();
    }

public:
    void to_recursively(const Device &device) {
        if (parameters_.size() > 0) {
            to_device_(device);
        }
        for (const auto &[sub_name, submodule] : submodules_) {
            auto submodule_my = static_cast<Module *>(submodule.get());
            if (submodule_my) {
                submodule_my->to_recursively(device);
            }
        }
    }
};

} // namespace infinidemo::nn::modules