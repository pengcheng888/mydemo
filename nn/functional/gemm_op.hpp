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

// Performs GEMM operation: C = alpha * A * B + beta * C
inline infiniStatus_t performGemm(Tensor &tensor_C, const Tensor &tensor_A,
                                  const Tensor &tensor_B, float alpha,
                                  float beta, Device device) {
    // Create InfiniOP handle
    infiniopHandle_t handle = context::getInfiniopHandle(device);

    // Create GEMM descriptor
    infiniopGemmDescriptor_t gemm_desc = nullptr;
    infiniStatus_t status = infiniopCreateGemmDescriptor(
        handle, &gemm_desc, tensor_C->desc(), tensor_A->desc(), tensor_B->desc());
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to create GEMM descriptor: " << status << std::endl;
        return status;
    }

    // Get workspace
    size_t workspace_size = 0;
    status = infiniopGetGemmWorkspaceSize(gemm_desc, &workspace_size);
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get workspace size: " << status << std::endl;
        infiniopDestroyGemmDescriptor(gemm_desc);
        return status;
    }

    void *workspace = nullptr;
    std::shared_ptr<Memory> workspace_memory = nullptr;
    if (workspace_size > 0) {
        workspace_memory = context::allocateMemory(workspace_size);
        workspace = workspace_memory->data();
    }

    // Execute GEMM operator
    status = infiniopGemm(gemm_desc, workspace, workspace_size, tensor_C->data(),
                          tensor_A->data(), tensor_B->data(), alpha, beta,
                          context::getStream());

    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to execute GEMM: " << status << std::endl;
        infiniopDestroyGemmDescriptor(gemm_desc);
        return status;
    }

    // Synchronize device
    context::syncDevice();

    // Clean up resources
    infiniopDestroyGemmDescriptor(gemm_desc);

    return INFINI_STATUS_SUCCESS;
}

} // namespace infinidemo::nn::functional
