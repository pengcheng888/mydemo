#include <CLI/CLI.hpp>
#include <cmath>
#include <infinicore/context/context.hpp>
#include <infinicore/tensor.hpp>
#include <infiniop.h>
#include <infinirt.h>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace infinicore;

// Parses command line arguments and selects the device.
// CPU, NVIDIA, MOORE, METAX, ILUVATAR, HYGON, ASCEND, CAMBRICON
Device selectDevice(int argc, char *argv[]) {
  CLI::App app{"InfiniOP GEMM Test - Test GEMM operation using InfiniOP API"};

  Device device = Device::cpu();

  // Platform configuration: (flag_name, device_type, help_message)
  using PlatformConfig = std::tuple<const char *, Device::Type, const char *>;
  const std::vector<PlatformConfig> platforms = {
      {"--cpu,-c", Device::Type::CPU, "Use CPU device (default)"},
      {"--nvidia", Device::Type::NVIDIA, "Use NVIDIA GPU device"},
      {"--moore", Device::Type::MOORE, "Use MOORE device"},
      {"--metax", Device::Type::METAX, "Use METAX device"},
      {"--iluvatar", Device::Type::ILUVATAR, "Use ILUVATAR device"},
      {"--hygon", Device::Type::HYGON, "Use HYGON device"},
      {"--ascend", Device::Type::ASCEND, "Use ASCEND device"},
      {"--cambricon", Device::Type::CAMBRICON, "Use CAMBRICON device"},
  };

  for (const auto &[flag_name, device_type, help_message] : platforms) {
    app.add_flag(
        flag_name,
        [&device, device_type](bool) { device = Device(device_type); },
        help_message);
  }

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    app.exit(e);
    throw std::runtime_error("Failed to parse command line arguments");
  }

  size_t device_count = context::getDeviceCount(device.getType());
  if (device_count == 0) {
    throw std::runtime_error("No " + device.toString() + " device available");
  }

  return device;
}

// Performs GEMM operation: C = alpha * A * B + beta * C
infiniStatus_t performGemm(const Tensor &tensor_A, const Tensor &tensor_B,
                           Tensor &tensor_C, float alpha, float beta,
                           Device device) {
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

void test_gemm(int argc, char *argv[]) {

  // Select device
  Device device = selectDevice(argc, argv);
  context::setDevice(device);

  std::cout << "current device: " << device.toString() << std::endl;

  // Infini runtime
  infiniStatus_t status = infinirtInit();
  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "Failed to initialize InfiniRT: " << status << std::endl;
    return;
  }

  // Define matrix dimensions:
  // A: M x K,
  // B: K x N, C: M x N
  // C = alpha * A * B + beta * C
  const size_t M = 2;
  const size_t K = 3;
  const size_t N = 4;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  std::cout << "==========================================" << std::endl;
  std::cout << "InfiniOP GEMM Test (using Tensor class)" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Device: " << device.toString() << std::endl;
  std::cout << "Matrix A: " << M << " x " << K << std::endl;
  std::cout << "Matrix B: " << K << " x " << N << std::endl;
  std::cout << "Matrix C: " << M << " x " << N << std::endl;
  std::cout << "alpha: " << alpha << ", beta: " << beta << std::endl;

  // Initialize input data (host memory)
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  for (size_t i = 0; i < M * K; i++) {
    h_A[i] = static_cast<float>(i / 10.0f);
  }
  for (size_t i = 0; i < K * N; i++) {
    h_B[i] = static_cast<float>(i / 10.0f);
  }

  // Create Tensor objects
  // First create temporary tensors on CPU referencing host data
  Device cpu_device = Device::cpu();
  Tensor temp_A =
      Tensor::from_blob(h_A.data(), {M, K}, DataType::F32, cpu_device);
  Tensor temp_B =
      Tensor::from_blob(h_B.data(), {K, N}, DataType::F32, cpu_device);

  // Create tensors on target device and copy data
  Tensor tensor_A = Tensor::empty({M, K}, DataType::F32, device);
  Tensor tensor_B = Tensor::empty({K, N}, DataType::F32, device);
  Tensor tensor_C = Tensor::zeros({M, N}, DataType::F32, device);

  tensor_A->copy_from(temp_A);
  tensor_B->copy_from(temp_B);

  // Execute GEMM operation
  std::cout << "\nExecuting GEMM..." << std::endl;
  status = performGemm(tensor_A, tensor_B, tensor_C, alpha, beta, device);
  if (status != INFINI_STATUS_SUCCESS) {
    std::cerr << "GEMM operation failed: " << status << std::endl;
    return;
  }
  std::cout << "\nExecuting GEMM over !\n" << std::endl;

  // Print input data
  std::cout << "Input Matrix A :" << std::endl;
  std::cout << tensor_A << std::endl;

  std::cout << "Input Matrix B :" << std::endl;
  std::cout << tensor_B << std::endl;

  std::cout << "Output Matrix C :" << std::endl;
  std::cout << tensor_C << std::endl;

  std::cout << "\nTest completed successfully!" << std::endl;
}

int main(int argc, char *argv[]) {

  test_gemm(argc, argv);

  return 0;
}
