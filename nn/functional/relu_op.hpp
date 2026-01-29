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

// Performs ReLU activation operation: y = max(0, x)
inline infiniStatus_t performRelu(Tensor &output, const Tensor &input,
                                  Device device) {
    // Create InfiniOP handle
    infiniopHandle_t handle = context::getInfiniopHandle(device);

    // 创建ReLU descriptor
    infiniopReluDescriptor_t relu_desc = nullptr;
    infiniStatus_t status = infiniopCreateReluDescriptor(
        handle, &relu_desc, output->desc(), input->desc());

    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to create ReLU descriptor: " << status << std::endl;
        return status;
    }

    // 获取workspace大小
    size_t workspace_size = 0;
    status = infiniopGetReluWorkspaceSize(relu_desc, &workspace_size);
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get workspace size: " << status << std::endl;
        infiniopDestroyReluDescriptor(relu_desc);
        return status;
    }

    // 分配workspace
    void *workspace = nullptr;
    std::shared_ptr<Memory> workspace_memory = nullptr;
    if (workspace_size > 0) {
        workspace_memory = context::allocateMemory(workspace_size);
        workspace = workspace_memory->data();
    }

    // 执行ReLU
    status = infiniopRelu(relu_desc, workspace, workspace_size, output->data(),
                          input->data(), context::getStream());
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to execute ReLU: " << status << std::endl;
        infiniopDestroyReluDescriptor(relu_desc);
        return status;
    }

    // 同步设备
    context::syncDevice();

    // 清理资源
    infiniopDestroyReluDescriptor(relu_desc);

    return INFINI_STATUS_SUCCESS;
}

} // namespace infinidemo::nn::functional
