#pragma once

#include <cstddef>
#include <infinicore/context/context.hpp>
#include <infinicore/device.hpp>
#include <infinicore/tensor.hpp>
#include <infiniop.h>
#include <infinirt.h>
#include <iostream>
#include <memory>
#include <vector>

namespace infinidemo::nn::functional {
using namespace infinicore;

// Performs 2D Convolution operation
inline infiniStatus_t performConv2D(Tensor &output, const Tensor &input,
                                    const Tensor &weight, const Tensor &bias,
                                    std::vector<ptrdiff_t> strides,
                                    std::vector<size_t> pads,
                                    std::vector<size_t> dilations,
                                    Device device) {

  // Create InfiniOP handle
  infiniopHandle_t handle = context::getInfiniopHandle(device);

  // 创建Conv descriptor
  infiniopConvDescriptor_t conv_desc = nullptr;
  infiniStatus_t status = infiniopCreateConvDescriptor(
      handle, &conv_desc, output->desc(), input->desc(), weight->desc(),
      bias ? bias->desc() : nullptr,
      const_cast<void *>(static_cast<const void *>(pads.data())),
      const_cast<void *>(static_cast<const void *>(strides.data())),
      const_cast<void *>(static_cast<const void *>(dilations.data())),
      pads.size());

  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to create Conv descriptor: " << status << std::endl;
    return status;
  }

  // 获取workspace大小
  size_t workspace_size = 0;
  status = infiniopGetConvWorkspaceSize(conv_desc, &workspace_size);
  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to get workspace size: " << status << std::endl;
    infiniopDestroyConvDescriptor(conv_desc);
    return status;
  }

  // 分配workspace
  void *workspace = nullptr;
  std::shared_ptr<Memory> workspace_memory = nullptr;
  if (workspace_size > 0) {
    workspace_memory = context::allocateMemory(workspace_size);
    workspace = workspace_memory->data();
  }

  // 执行Conv
  status = infiniopConv(conv_desc, workspace, workspace_size, output->data(),
                        input->data(), weight->data(),
                        bias ? bias->data() : nullptr, context::getStream());
  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to execute Conv: " << status << std::endl;
    infiniopDestroyConvDescriptor(conv_desc);
    return status;
  }

  // 同步设备
  context::syncDevice();

  // 清理资源
  infiniopDestroyConvDescriptor(conv_desc);

  return INFINI_STATUS_SUCCESS;
}

} // namespace infinidemo::nn::functional
