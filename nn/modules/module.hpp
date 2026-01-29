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

// class Module : public infinicore::nn::Module {
// public:
//   virtual ~Module() = default;
//   virtual void to_device_(const Device &device) = 0;

// public:
//   void to(const Device &device) {
//     to_recursively(device);
//     infinicore::context::syncDevice();
//   }

// private:
//   void to_recursively(const Device &device) {
//     if (parameters_.size() > 0) {
//       to_device_(device);
//     }
//     for (const auto &[sub_name, submodule] : submodules_) {
//       auto submodule_my = static_cast<Module *>(submodule.get());
//       if (submodule_my) {
//         submodule_my->to_recursively(device);
//       }
//     }
//   }
// };

template <typename Derived = void>
class Module : public infinicore::nn::Module {
public:
    void interface(const Device &device) {
        if constexpr (std::is_same_v<Derived, void>) {
            // Modules with Derived = void don't have to_device_ method, skip
            return;
        } else {
            static_cast<Derived *>(this)->to_device_(device);
        }
    }

public:
    void to(const Device &device) {
        to_recursively(device);
        infinicore::context::syncDevice();
    }

public:
    void to_recursively(const Device &device) {
        if (parameters_.size() > 0) {
            // for (const auto &[param_name, param] : parameters_) {
            //   printf("param_name: %s\n", param_name.c_str());
            // }
            interface(device);
        }
        for (const auto &[sub_name, submodule] : submodules_) {
            auto submodule_my = static_cast<Module<> *>(submodule.get());
            if (submodule_my) {
                submodule_my->to_recursively(device);
            }
        }
    }

    // void to_recursively(const std::string &prefix, const Device &device) {
    //   // Add direct parameters with the given prefix

    //   if (parameters_.size() > 0) {
    //     for (const auto &[param_name, param] : parameters_) {
    //       std::string full_name =
    //           prefix.empty() ? param_name : prefix + "." + param_name;
    //       std::cout << " to_recursively : " << full_name << std::endl;

    //       Tensor &param_ref = parameters_[param_name];
    //       parameters_[param_name] = param_ref->to(device);
    //     }
    //     interface(device);
    //   }
    //   // Recursively collect parameters from submodules with extended prefix
    //   for (const auto &[sub_name, submodule] : submodules_) {
    //     std::string sub_prefix =
    //         prefix.empty() ? sub_name : prefix + "." + sub_name;
    //     auto submodule_my = static_cast<Module<> *>(submodule.get());
    //     if (submodule_my) {
    //       submodule_my->to_recursively(sub_prefix, device);
    //     }
    //   }
    // }
};
}; // namespace infinidemo::nn::modules