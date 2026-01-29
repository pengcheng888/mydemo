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

// Performs Add operation: C = A + B
inline infiniStatus_t performAdd(Tensor &out, const Tensor &input,
                                 const Tensor &other, Device device) {
    // Create InfiniOP handle
    infiniopHandle_t handle = context::getInfiniopHandle(device);

    // Create Add descriptor
    infiniopAddDescriptor_t add_desc = nullptr;
    infiniStatus_t status = infiniopCreateAddDescriptor(
        handle, &add_desc, out->desc(), input->desc(), other->desc());

    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to create Add descriptor: " << status << std::endl;
        return status;
    }

    // Get workspace size
    size_t workspace_size = 0;
    status = infiniopGetAddWorkspaceSize(add_desc, &workspace_size);
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get workspace size: " << status << std::endl;
        infiniopDestroyAddDescriptor(add_desc);
        return status;
    }

    // Allocate workspace
    void *workspace = nullptr;
    std::shared_ptr<Memory> workspace_memory = nullptr;
    if (workspace_size > 0) {
        workspace_memory = context::allocateMemory(workspace_size);
        workspace = workspace_memory->data();
    }

    // Execute Add operator
    status = infiniopAdd(add_desc, workspace, workspace_size, out->data(),
                         input->data(), other->data(), context::getStream());
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to execute Add: " << status << std::endl;
        infiniopDestroyAddDescriptor(add_desc);
        return status;
    }

    // Clean up resources
    infiniopDestroyAddDescriptor(add_desc);

    return INFINI_STATUS_SUCCESS;
}

} // namespace infinidemo::nn::functional
