import numpy as np
import infinicore
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import os
from pymodels.modeling_utils import infini_to_numpy
from pymodels import ResNetForImageClassification
import cv2

def selectDevice():
    import argparse
    import sys

    platform_to_device = {
        "cpu": "cpu",
        "nvidia": "cuda",
        "metax": "cuda",
        "moore": "musa",
        "iluvatar": "cuda",
        "hygon": "cuda",
        "ascend": "npu",
        "cambricon": "mlu",
    }

    parser = argparse.ArgumentParser(description="run Llama args")
    for platform, device_str in platform_to_device.items():
        help_msg = (
            f"Use {platform.upper()} device"
            if platform != "cpu"
            else "Use CPU device (default)"
        )
        parser.add_argument(
            f"--{platform}",
            action="store_true",
            help=help_msg,
        )

    args = parser.parse_args()

    device_str = platform_to_device["cpu"]  # 默认值
    for platform in platform_to_device.keys():
        if getattr(args, platform, False):
            device_str = platform_to_device[platform]
            break

    return infinicore.device(device_str, 0)



def func1():
    device = selectDevice()
    print("current device: ", device)

    # 创建模型实例
    from pymodels import ResNetForImageClassification
    model = ResNetForImageClassification.from_pretrained("/home/ubuntu/models/resnet/resnet-18-mnist/", device)
    print(f"模型创建成功: {model}")   
    if False:
        state_dict = model.state_dict()
        keys = sorted(state_dict.keys())
        for key in keys:
            print(f"{key}: {state_dict[key].shape}", state_dict[key].dtype, state_dict[key].device)
            # print(f"{state_dict[key]}")
        exit(0)

    # 创建输入tensor
    input_tensor = infinicore.from_numpy(
        np.ones((1, 3,224,224)).astype(np.float32),
        device=infinicore.device("cpu", 0),
    )
    # input_tensor = infinicore.empty((1, 3,224,224), device=infinicore.device("cpu", 0),dtype=infinicore.float32)
    print(f"\n输入tensor形状: {input_tensor.shape}")

    # 推理
    output = model.forward(input_tensor.to(device))
    print(f"输出tensor: {output}")


if __name__ == "__main__":
    device = selectDevice()
    print("current device: ", device)

    # 创建模型实例

    model_path = "/home/ubuntu/models/resnet/resnet-18-mnist/"
    model = ResNetForImageClassification.from_pretrained(model_path)
    print(f"模型创建成功: {model}")   
    if True:
        state_dict = model.state_dict()
        keys = sorted(state_dict.keys())
        for key in keys:
            print(f"{key}: {state_dict[key].shape}", state_dict[key].dtype, state_dict[key].device)
            # print(f"{state_dict[key]}")
  

    # 创建输入tensor
    input_tensor = infinicore.from_numpy(
        np.ones((1, 3,224,224)).astype(np.float32),
        device=infinicore.device("cpu", 0),
    )
    # input_tensor = infinicore.empty((1, 3,224,224), device=infinicore.device("cpu", 0),dtype=infinicore.float32)
    print(f"\n输入tensor形状: {input_tensor.shape}")

    test_dataset = mnist.MNIST(root= os.path.join(model_path, "test"), train=False, transform=ToTensor())
    
    model.to(device=device)
    for i in range(10):
        test_image, test_label = test_dataset[i]
        test_image = test_image.unsqueeze(0)  # Add batch dimension
        if True:
            test_image = F.interpolate(
                test_image, size=(224, 224), mode="bilinear", align_corners=False
            ).contiguous()
            test_image = test_image.repeat(1, 3, 1, 1).contiguous()


        # 前向传播
        input_tensor = infinicore.from_torch(test_image)
        predict = model.forward(input_tensor.to(device))
        # print(f"输出tensor: {predict}")
        
        predict_np = infini_to_numpy(predict)
        predict_class = np.argmax(predict_np, axis=-1)
        print(f"预测类别: {predict_class} , 真实类别: {test_label}")

        if True:
            image = (test_image[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            cv2.imshow("test_image", image)
            cv2.waitKey()



