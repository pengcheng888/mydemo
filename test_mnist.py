#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MnistForImageClassification 的 Python 接口
"""

import sys
import numpy as np

# 导入 infinicore
import infinicore

from pymodels import MnistForImageClassification


def inference(model):
    import torch
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision.datasets import mnist
    from torchvision.transforms import ToTensor
    from safetensors.torch import load_file


    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = mnist.MNIST(root="/home/ubuntu/pr666/demo/LeNet5-MNIST-PyTorch/test", train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1)



    for i in range(5):
        test_image, test_label = test_dataset[i]
        test_image = test_image.unsqueeze(0)  # Add batch dimension

        image_tensor = test_image.float()

        infini_tensor = infinicore.from_torch(image_tensor)
        print(f"infini_tensor: {infini_tensor}")
        predict_y = model(infini_tensor)
        print("predict_y: ",predict_y)
        predict_class = torch.argmax(predict_y, dim=-1)

        print(
            f"Image {i}: True label = {test_label}, Predicted = {predict_class}, "
            f"Match = {predict_class == test_label}"
        )

if __name__ == "__main__":
    print("=" * 50)
    print("测试 MnistForImageClassification")
    print("=" * 50)

    # 创建模型实例（使用 Python 接口）
    # model = MnistForImageClassification()
    # print(f"模型创建成功: {model}")

    model = MnistForImageClassification.from_pretrained("/home/ubuntu/pr666/demo/mydemo/")
    print(f"模型创建成功: {model}")   


    # 获取模型参数字典
    print("\n模型参数字典:")
    model_state_dict = model.state_dict()
    # print(model_state_dict)
    for key, value in model_state_dict.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")


    # 创建输入tensor（使用多种方式测试）


    print(f"\n测试前向传播...")
    
    # 方式1: 使用 infinicore.from_numpy
    input_tensor = infinicore.from_numpy(
        np.ones((1,1, 28,28),dtype=np.float32), #  np.random.randn(1,1, 28,28), (1,1936)
        device=infinicore.device("cpu", 0),
        dtype=infinicore.float32,
    )
    print(f"输入tensor形状: {input_tensor.shape}")
    

    inference(model)

