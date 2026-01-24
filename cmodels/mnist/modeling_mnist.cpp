#include "modeling_mnist.hpp"
#include <infinicore/context/context.hpp>
#include <string>
#include <unordered_map>

namespace infinidemo::models {
MnistForImageClassification::MnistForImageClassification() {
  size_t in_features = 4;
  size_t out_features = 1;

  // Initialize parameters using macro
  // INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features},
  // DataType::F32, Device::cpu()));

  // context::setDevice(infinicore::Device::cpu());

  in_features = 1936;
  out_features = 10;
  size_t in_channels = 1;
  size_t out_channels = 4;
  size_t kernel_size = 7;

  INFINICORE_NN_MODULE_INIT(fc1, in_features, out_features, true);
  INFINICORE_NN_MODULE_INIT(conv1, in_channels, out_channels, kernel_size,
                            true);
}

Tensor MnistForImageClassification::forward(Tensor &input) const {
  // std::cout << "forward input: " << input->shape()[0] << " " <<
  // input->shape()[1] << " " << input->shape()[2] << " " << input->shape()[3]
  // << std::endl;
  std::cout << " ------> MnistForImageClassification input:   \n"
            << input << std::endl;

  auto output = relu_.forward(conv1_->forward(input));

  // y = y.view(y.shape[0], -1)
  size_t temp = 1;
  for (size_t i = 1; i < output->shape().size(); i++) {
    temp *= output->shape()[i];
  }

  std::vector<Size> new_shape = {output->shape()[0], temp};

  output = output->view(new_shape);

  auto output2 = relu_.forward(fc1_->forward(output));
  return output2;
}

} // namespace infinidemo::models
