#include "modeling_resnet.hpp"
#include "../../nn/modules/conv.hpp"
#include "../../nn/modules/module.hpp"
#include "../../nn/modules/relu.hpp"
#include <stdexcept>
#include <string>

namespace {
using namespace infinicore;
using namespace infinidemo::models;

class ResNetShortCut : public infinidemo::nn::modules::Module {
public:
  ResNetShortCut(int in_channels, int out_channels, int stride = 2,
                 const DataType &dtype = DataType::F32) {
    INFINICORE_NN_MODULE_INIT(convolution, in_channels, out_channels, 1, stride,
                              0, 1, 1, false, dtype);
  }

  inline Tensor forward(Tensor &input) const {
    Tensor hidden_state = convolution_->forward(input);
    return hidden_state;
  }

private:
  void to_device_(const Device &device) override {
    static_cast<infinidemo::nn::modules::Module *>(convolution_.get())
        ->to_device_(device);
  }

protected:
  INFINICORE_NN_MODULE(infinidemo::nn::modules::Conv2d, convolution);
};

class ResNetConvLayer : public infinidemo::nn::modules::Module {
public:
  ResNetConvLayer(int in_channels, int out_channels, int kernel_size = 3,
                  int stride = 1, const std::string activation = "relu",
                  const DataType &dtype = DataType::F32)
      : in_channels_(in_channels), out_channels_(out_channels),
        kernel_size_(kernel_size), stride_(stride), activation_(activation) {
    INFINICORE_NN_MODULE_INIT(convolution, in_channels_, out_channels_,
                              kernel_size_, stride_, kernel_size_ / 2, 1, 1,
                              false, dtype);
  }

  inline Tensor forward(Tensor &input) const {
    Tensor hidden_state = convolution_->forward(input);
    if (activation_.empty()) {
      hidden_state = identity_.forward(hidden_state);
    } else {
      if (activation_ == "relu" || activation_ == "ReLU") {
        hidden_state = relu_.forward(hidden_state);
      } else {
        throw std::runtime_error("Invalid activation function: " + activation_);
      }
    }
    return hidden_state;
  }

private:
  void to_device_(const Device &device) override {
    static_cast<infinidemo::nn::modules::Module *>(convolution_.get())
        ->to_device_(device);
    static_cast<infinidemo::nn::modules::Module *>(&relu_)->to_device_(device);
    static_cast<infinidemo::nn::modules::Module *>(&identity_)
        ->to_device_(device);
  }

protected:
  INFINICORE_NN_MODULE(infinidemo::nn::modules::Conv2d, convolution);
  infinidemo::nn::modules::ReLU relu_;
  infinidemo::nn::modules::Identity identity_;
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  std::string activation_;
};

class ResNetBasicLayer : public infinidemo::nn::modules::Module {
public:
  ResNetBasicLayer(int in_channels, int out_channels, int stride = 1,
                   const std::string activation = "relu",
                   const DataType &dtype = DataType::F32)
      : activation_(activation) {
    should_apply_shortcut_ = (in_channels != out_channels) || (stride != 1);
    if (should_apply_shortcut_) {
      INFINICORE_NN_MODULE_INIT(shortcut, in_channels, out_channels, stride,
                                dtype);
    }

    layer_.reserve(2);
    layer_.push_back(this->register_module<ResNetConvLayer>(
        "layer." + std::to_string(0), in_channels, out_channels, 3, stride,
        "relu", dtype));
    layer_.push_back(this->register_module<ResNetConvLayer>(
        "layer." + std::to_string(1), out_channels, out_channels, 3, 1, "",
        dtype));
  }

  inline Tensor forward(Tensor &hidden_state) const {
    Tensor residual = Tensor::empty(
        hidden_state->shape(), hidden_state->dtype(), hidden_state->device());
    residual->copy_from(hidden_state);

    size_t num_layers = layer_.size();
    for (size_t i = 0; i < num_layers; ++i) {
      hidden_state = layer_[i]->forward(hidden_state);
    }

    if (should_apply_shortcut_) {
      residual = shortcut_->forward(residual);
    } else {
      residual = identity_.forward(residual);
    }

    hidden_state += residual;
    if (activation_ == "relu" || activation_ == "ReLU") {
      hidden_state = relu_.forward(hidden_state);
    } else {
      throw std::runtime_error("Invalid activation function: " + activation_);
    }
    return hidden_state;
  }

private:
  void to_device_(const Device &device) override {
    if (should_apply_shortcut_) {
      static_cast<infinidemo::nn::modules::Module *>(shortcut_.get())
          ->to_device_(device);
    }

    for (auto &layer_module : layer_) {
      static_cast<infinidemo::nn::modules::Module *>(layer_module.get())
          ->to_device_(device);
    }

    static_cast<infinidemo::nn::modules::Module *>(&relu_)->to_device_(device);
    static_cast<infinidemo::nn::modules::Module *>(&identity_)
        ->to_device_(device);
  }

protected:
  INFINICORE_NN_MODULE(ResNetShortCut, shortcut);
  INFINICORE_NN_MODULE_VEC(ResNetConvLayer, layer);
  infinidemo::nn::modules::ReLU relu_;
  infinidemo::nn::modules::Identity identity_;
  std::string activation_;
  bool should_apply_shortcut_;
};

class ResNetStage : public infinidemo::nn::modules::Module {
public:
  ResNetStage(const ResNetConfig &config, int in_channels, int out_channels,
              int stride = 2, int depth = 2,
              const DataType &dtype = DataType::F32) {
    if (config.layer_type == "bottleneck") {
      throw std::runtime_error("Bottleneck layer is not supported");
    }

    layers_.reserve(depth);
    layers_.push_back(this->register_module<ResNetBasicLayer>(
        "layers." + std::to_string(0), in_channels, out_channels, stride,
        config.hidden_act, dtype));

    for (int i = 1; i < depth; ++i) {
      layers_.push_back(this->register_module<ResNetBasicLayer>(
          "layers." + std::to_string(i), out_channels, out_channels, 1,
          config.hidden_act, dtype));
    }
  }

  inline Tensor forward(Tensor &input) const {
    Tensor hidden_state = input;
    size_t num_layers = layers_.size();
    for (size_t i = 0; i < num_layers; ++i) {
      hidden_state = layers_[i]->forward(hidden_state);
    }
    return hidden_state;
  }

private:
  void to_device_(const Device &device) override {
    for (auto &layer_module : layers_) {
      static_cast<infinidemo::nn::modules::Module *>(layer_module.get())
          ->to_device_(device);
    }
  }

protected:
  INFINICORE_NN_MODULE_VEC(ResNetBasicLayer, layers);
};

class ResNetEncoder : public infinidemo::nn::modules::Module {
public:
  ResNetEncoder(const ResNetConfig &config,
                const DataType &dtype = DataType::F32) {
    if (config.hidden_sizes.empty() || config.depths.empty()) {
      throw std::runtime_error(
          "ResNetConfig must have non-empty hidden_sizes and depths");
    }

    if (config.hidden_sizes.size() != config.depths.size()) {
      throw std::runtime_error(
          "ResNetConfig hidden_sizes and depths must have the same size");
    }

    size_t num_stages = config.hidden_sizes.size();
    stages_.reserve(num_stages);

    int first_stride = config.downsample_in_first_stage ? 2 : 1;
    stages_.push_back(this->register_module<ResNetStage>(
        "stages." + std::to_string(0), config, config.embedding_size,
        config.hidden_sizes[0], first_stride, config.depths[0], dtype));

    for (size_t i = 1; i < num_stages; ++i) {
      stages_.push_back(this->register_module<ResNetStage>(
          "stages." + std::to_string(i), config, config.hidden_sizes[i - 1],
          config.hidden_sizes[i], 2, config.depths[i], dtype));
    }
  }

  inline Tensor forward(Tensor &hidden_state) const {
    size_t num_stages = stages_.size();
    for (size_t i = 0; i < num_stages; ++i) {
      hidden_state = stages_[i]->forward(hidden_state);
    }
    return hidden_state;
  }

private:
  void to_device_(const Device &device) override {
    for (auto &stage_module : stages_) {
      static_cast<infinidemo::nn::modules::Module *>(stage_module.get())
          ->to_device_(device);
    }
  }

protected:
  INFINICORE_NN_MODULE_VEC(ResNetStage, stages);
};

class ResNetEmbeddings : public infinidemo::nn::modules::Module {
public:
  ResNetEmbeddings(const ResNetConfig &config,
                   const DataType &dtype = DataType::F32)
      : num_channels_(config.num_channels) {

    INFINICORE_NN_MODULE_INIT(embedder, config.num_channels,
                              config.embedding_size, 7, 2, config.hidden_act,
                              dtype);

    INFINICORE_NN_MODULE_INIT(pooler, config.embedding_size,
                              config.embedding_size, 3, 2, 1, 1, 1, false,
                              dtype);
  }

  inline Tensor forward(Tensor &pixel_values) const {

    int num_channels = static_cast<int>(pixel_values->shape()[1]);
    if (num_channels != num_channels_) {
      throw std::runtime_error("Channel dimension mismatch");
    }

    Tensor embedding = embedder_->forward(pixel_values);
    embedding = pooler_->forward(embedding);
    return embedding;
  }

private:
  void to_device_(const Device &device) override {
    static_cast<infinidemo::nn::modules::Module *>(embedder_.get())
        ->to_device_(device);
    static_cast<infinidemo::nn::modules::Module *>(pooler_.get())
        ->to_device_(device);
  }

protected:
  INFINICORE_NN_MODULE(ResNetConvLayer, embedder);
  INFINICORE_NN_MODULE(infinidemo::nn::modules::Conv2d, pooler);
  int num_channels_;
};

} // namespace

namespace infinidemo::models {
using namespace infinicore;

class ResNetModel : public infinidemo::nn::modules::Module {
public:
  ResNetModel(const ResNetConfig &config,
              const DataType &dtype = DataType::F32) {
    INFINICORE_NN_MODULE_INIT(embedder, config, dtype);
    INFINICORE_NN_MODULE_INIT(encoder, config, dtype);
    INFINICORE_NN_MODULE_INIT(pooler, 512, 512, 7, 1, 0, 1, 1, false,
                              dtype); // 512 需要修改
  }

  inline Tensor forward(Tensor &pixel_values) const {
    Tensor embedding_output = embedder_->forward(pixel_values);
    Tensor encoder_output = encoder_->forward(embedding_output);
    Tensor pooled_output = pooler_->forward(encoder_output);
    return pooled_output;
  }

private:
  void to_device_(const Device &device) override {
    static_cast<infinidemo::nn::modules::Module *>(embedder_.get())
        ->to_device_(device);
    static_cast<infinidemo::nn::modules::Module *>(encoder_.get())
        ->to_device_(device);
    static_cast<infinidemo::nn::modules::Module *>(pooler_.get())
        ->to_device_(device);
  }

protected:
  INFINICORE_NN_MODULE(::ResNetEmbeddings, embedder);
  INFINICORE_NN_MODULE(::ResNetEncoder, encoder);
  INFINICORE_NN_MODULE(infinidemo::nn::modules::Conv2d, pooler);
};

ResNetForImageClassification::ResNetForImageClassification(
    const ResNetConfig &config)
    : config_(config), num_labels_(config.num_labels) {
  if (config.num_labels <= 0) {
    throw std::runtime_error("ResNetConfig num_labels must be greater than 0");
  }
  DataType dtype = DataType::F32;
  if (config.torch_dtype != "float32") {
    throw std::runtime_error("Invalid data dtype: " + config.torch_dtype);
  }

  INFINICORE_NN_MODULE_INIT(resnet, config, dtype);

  int in_features = config.hidden_sizes.back();
  int out_features = config.num_labels;
  classifier_.reserve(1);
  classifier_.push_back(this->register_module<infinidemo::nn::modules::Linear>(
      "classifier.1", static_cast<size_t>(in_features),
      static_cast<size_t>(out_features), true, dtype));
}

void ResNetForImageClassification::to_device_(const Device &device) {

  static_cast<infinidemo::nn::modules::Module *>(resnet_.get())
      ->to_device_(device);
  for (auto &classifier_module : classifier_) {
    static_cast<infinidemo::nn::modules::Module *>(classifier_module.get())
        ->to_device_(device);
  }
  static_cast<infinidemo::nn::modules::Module *>(&flatten_)->to_device_(device);
  device_ = device;
}

Tensor ResNetForImageClassification::forward(Tensor &pixel_values) {
  if (pixel_values->device() != device_) {
    throw std::runtime_error("Device mismatch");
  }

  Tensor outputs = resnet_->forward(pixel_values);
  Tensor pooled_output = flatten_.forward(outputs);
  Tensor logits = classifier_[0]->forward(pooled_output);
  return logits;
}

} // namespace infinidemo::models
