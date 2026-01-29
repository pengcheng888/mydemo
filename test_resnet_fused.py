import numpy as np
import infinicore
import argparse
import sys
from PIL import Image
from pymodels.modeling_utils import infini_to_numpy
from pymodels import ResNetForImageClassification
from transformers import AutoImageProcessor


def selectDevice():
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


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


if __name__ == "__main__":
    device = selectDevice()
    print("current device: ", device)

    # 创建模型实例
    model_path = "../resnet-18-fused/"
    model = ResNetForImageClassification.from_pretrained(model_path)
    print(f"模型创建成功: {model}")
    if False:
        state_dict = model.state_dict()
        keys = sorted(state_dict.keys())
        for key in keys:
            print(
                f"{key}: {state_dict[key].shape}",
                state_dict[key].dtype,
                state_dict[key].device,
            )

    # 创建输入图片
    image_path = "../resnet-18-fused/src/cats_image.jpeg"
    image_path = "../resnet-18-fused/src/dog.jpg"
    image = Image.open(image_path).convert("RGB")
    feature_extractor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    inputs = feature_extractor(image, return_tensors="pt")["pixel_values"]

    model.to(device=device)
    for i in range(1):
        input_tensor = infinicore.from_torch(inputs)
        predict = model.forward(input_tensor.to(device))

        predict_np = infini_to_numpy(predict)
        predict_np = softmax(predict_np)
        predict_class = np.argmax(predict_np, axis=-1)
        predict_probs = np.max(predict_np, axis=-1)

        predict_class = predict_class.item()
        predict_name = model.config.id2label[f"{predict_class}"]
        print(f" 类别: {predict_name}  概率: {round(predict_probs.item(), 3)}\n")
