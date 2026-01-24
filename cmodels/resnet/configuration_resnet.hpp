#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace infinidemo::models {
struct ResNetConfig {
  std::vector<std::string> architectures;
  std::vector<int> depths;
  bool downsample_in_first_stage = false;
  int embedding_size = 64;
  std::string hidden_act = "relu";
  std::vector<int> hidden_sizes;
  std::string layer_type = "basic";
  std::string model_type = "resnet";
  int num_channels = 3;
  std::string torch_dtype = "float32";
  std::string transformers_version = "4.18.0.dev0";
  int num_labels = -1;                   // Additional parameters
  bool downsample_in_bottleneck = false; // Additional parameters
};

inline std::ostream &operator<<(std::ostream &os, const ResNetConfig &config) {
  const std::string indent = "  "; // 2 spaces indent
  os << "\n{\n";

  // architectures
  os << indent << "\"architectures\": [ ";
  for (size_t i = 0; i < config.architectures.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << "\"" << config.architectures[i] << "\"";
  }
  os << " ],\n";

  // depths
  os << indent << "\"depths\": [";
  for (size_t i = 0; i < config.depths.size(); ++i) {
    if (i == 0)
      os << " ";
    else
      os << ", ";
    os << config.depths[i];
  }
  os << " ],\n";

  // downsample_in_first_stage
  os << indent << "\"downsample_in_first_stage\": "
     << (config.downsample_in_first_stage ? "true" : "false") << ",\n";

  // downsample_in_bottleneck
  os << indent << "\"downsample_in_bottleneck\": "
     << (config.downsample_in_bottleneck ? "true" : "false") << ",\n";

  // embedding_size
  os << indent << "\"embedding_size\": " << config.embedding_size << ",\n";

  // hidden_act
  os << indent << "\"hidden_act\": \"" << config.hidden_act << "\",\n";

  // hidden_sizes
  os << indent << "\"hidden_sizes\": [ ";
  for (size_t i = 0; i < config.hidden_sizes.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << config.hidden_sizes[i];
  }
  os << " ],\n";

  // layer_type
  os << indent << "\"layer_type\": \"" << config.layer_type << "\",\n";

  // model_type
  os << indent << "\"model_type\": \"" << config.model_type << "\",\n";

  // num_channels
  os << indent << "\"num_channels\": " << config.num_channels << ",\n";

  // torch_dtype
  os << indent << "\"torch_dtype\": \"" << config.torch_dtype << "\",\n";

  // transformers_version
  os << indent << "\"transformers_version\": \"" << config.transformers_version
     << "\",\n";

  // num_labels (last item, no trailing comma)
  os << indent << "\"num_labels\": " << config.num_labels << "\n";

  os << "}";
  return os;
}

} // namespace infinidemo::models
