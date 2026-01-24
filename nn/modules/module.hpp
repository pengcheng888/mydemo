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

// class Module : public infinicore::nn::Module {
//     public:
//       //   void to(const Device &device) {
//       //     // to_recursively("", device);
//       //     to_device_(device);
//       //     infinicore::context::syncDevice();
//       //   }

//       virtual void to_device_(const Device &device) = 0;

//     private:
//       void to_recursively(const std::string &prefix, const Device &device) {
//         // Add direct parameters with the given prefix
//         for (const auto &[param_name, param] : parameters_) {
//           std::string full_name =
//               prefix.empty() ? param_name : prefix + "." + param_name;
//           std::cout << " to_recursively : " << full_name << std::endl;

//           Tensor &param_ref = parameters_[param_name];
//           parameters_[param_name] = param_ref->to(device);
//         }

//         // Recursively collect parameters from submodules with extended
//         prefix for (const auto &[sub_name, submodule] : submodules_) {
//           std::string sub_prefix =
//               prefix.empty() ? sub_name : prefix + "." + sub_name;

//           // Cast to our Module type (all submodules should be
//           // infinidemo::nn::modules::Module)
//           auto submodule_my = static_cast<Module *>(submodule.get());
//           if (submodule_my) {
//             submodule_my->to_recursively(sub_prefix, device);
//           }
//         }
//       }
//     };