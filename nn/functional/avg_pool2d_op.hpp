#pragma once

#include <infinicore/context/context.hpp>
#include <infinicore/device.hpp>
#include <infinicore/tensor.hpp>
#include <infiniop.h>
#include <infinirt.h>
#include <iostream>
#include <memory>

namespace infinidemo::nn::functional {
using namespace infinicore;

// 执行AvgPool2D操作: output = avg_pool2d(input)
inline infiniStatus_t performAvgPool2d(const Tensor &tensor_input,
                                       Tensor &tensor_output, int kernel_h,
                                       int kernel_w, int stride_h, int stride_w,
                                       int padding_h, int padding_w,
                                       int dilation_h, int dilation_w,
                                       bool ceil_mode, Device device) {
  // Create InfiniOP handle
  infiniopHandle_t handle = context::getInfiniopHandle(device);
  // 创建AvgPool2D descriptor
  infiniopAvgPool2dDescriptor_t pool_desc = nullptr;
  infiniStatus_t status = infiniopCreateAvgPool2dDescriptor(
      handle, &pool_desc, tensor_output->desc(), tensor_input->desc(), kernel_h,
      kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h,
      dilation_w, ceil_mode ? 1 : 0);

  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to create AvgPool2D descriptor: " << status
              << std::endl;
    return status;
  }

  // 获取workspace大小
  size_t workspace_size = 0;
  status = infiniopGetAvgPool2dWorkspaceSize(pool_desc, &workspace_size);
  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to get workspace size: " << status << std::endl;
    infiniopDestroyAvgPool2dDescriptor(pool_desc);
    return status;
  }

  // 分配workspace
  void *workspace = nullptr;
  std::shared_ptr<Memory> workspace_memory = nullptr;
  if (workspace_size > 0) {
    workspace_memory = context::allocateMemory(workspace_size);
    workspace = workspace_memory->data();
  }

  // 执行AvgPool2D
  status = infiniopAvgPool2d(pool_desc, workspace, workspace_size,
                             tensor_output->data(), tensor_input->data(),
                             context::getStream());
  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to execute AvgPool2D: " << status << std::endl;
    infiniopDestroyAvgPool2dDescriptor(pool_desc);
    return status;
  }

  // 同步设备
  context::syncDevice();

  // 清理资源
  infiniopDestroyAvgPool2dDescriptor(pool_desc);

  return INFINI_STATUS_SUCCESS;
}

} // namespace infinidemo::nn::functional
