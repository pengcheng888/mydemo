#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Python ResNetConfig 类
"""

import sys
import json
from pymodels.resnet.configuration_resnet import ResNetConfig

# 测试 ResNetConfig 类
if __name__ == "__main__":
    print("=" * 50)
    print("测试 Python ResNetConfig 类")
    print("=" * 50)



    # 方式2: 从 JSON 文件加载
    print("\n2. 从 config.json 文件加载配置...")
    config = ResNetConfig.from_pretrained("config.json")
    print(f"配置加载成功: {config}")

    exit(-1)

    # 方式1: 使用默认值创建
    print("\n1. 使用默认值创建配置...")
    try:
        config_default = ResNetConfig()
        print(f"默认配置: {config_default}")
        print(f"  model_type: {config_default.model_type}")
        print(f"  num_channels: {config_default.num_channels}")
        print(f"  embedding_size: {config_default.embedding_size}")
        
        # 方式2: 从 JSON 文件加载
        print("\n2. 从 config.json 文件加载配置...")
        config = ResNetConfig.from_pretrained("config.json")
        print(f"配置加载成功: {config}")
        
        # 访问成员变量
        print(f"\n配置信息:")
        print(f"  model_type: {config.model_type}")
        print(f"  architectures: {config.architectures}")
        print(f"  num_channels: {config.num_channels}")
        print(f"  embedding_size: {config.embedding_size}")
        print(f"  hidden_sizes: {config.hidden_sizes}")
        print(f"  depths: {config.depths}")
        print(f"  hidden_act: {config.hidden_act}")
        print(f"  layer_type: {config.layer_type}")
        print(f"  downsample_in_first_stage: {config.downsample_in_first_stage}")
        print(f"  torch_dtype: {config.torch_dtype}")
        print(f"  transformers_version: {config.transformers_version}")
        
        # 方式3: 使用位置参数创建
        print(f"\n3. 使用位置参数创建配置...")
        config2 = ResNetConfig(
            architectures=["ResNetForImageClassification"],
            depths=[2, 2, 2, 2],
            downsample_in_first_stage=False,
            embedding_size=128,
            hidden_act="gelu",
            hidden_sizes=[64, 128, 256, 512],
            layer_type="basic",
            model_type="resnet",
            num_channels=3,
            torch_dtype="float32",
            transformers_version="4.18.0.dev0"
        )
        print(f"  使用位置参数创建: {config2}")
        print(f"  model_type: {config2.model_type}")
        print(f"  embedding_size: {config2.embedding_size}")
        print(f"  hidden_act: {config2.hidden_act}")
        
        # 方式4: 使用关键字参数创建（部分参数）
        print(f"\n4. 使用关键字参数创建配置（部分参数）...")
        config3 = ResNetConfig(
            model_type="custom_resnet",
            num_channels=1,
            embedding_size=256,
            hidden_act="relu"
        )
        print(f"  使用关键字参数创建: {config3}")
        print(f"  model_type: {config3.model_type}")
        print(f"  num_channels: {config3.num_channels}")
        print(f"  embedding_size: {config3.embedding_size}")
        print(f"  hidden_act: {config3.hidden_act}")
        print(f"  architectures (默认): {config3.architectures}")
        print(f"  depths (默认): {config3.depths}")
        
        # 方式5: 测试修改配置
        print(f"\n5. 测试修改配置...")
        config.embedding_size = 128
        config.hidden_act = "gelu"
        print(f"  修改后 embedding_size: {config.embedding_size}")
        print(f"  修改后 hidden_act: {config.hidden_act}")
        print(f"  修改后的配置: {config}")
        
        # 方式6: 测试可选列表参数
        print(f"\n6. 测试可选列表参数...")
        config4 = ResNetConfig(
            model_type="test",
            architectures=["TestArch"],  # 设置 architectures
            # depths 不设置，应该保持默认值
        )
        print(f"  配置 (设置 architectures): {config4}")
        print(f"  architectures: {config4.architectures}")
        print(f"  depths: {config4.depths}")
        
        # 方式7: 测试 __repr__ 方法（调用 C++ 的 operator<<）
        print(f"\n7. 测试 __repr__ 方法（调用 C++ 的 operator<<）...")
        print(f"  config.__repr__(): {config.__repr__()}")
        print(f"  str(config): {str(config)}")
        print(f"  print(config): ", end="")
        print(config)
        
        print("\n" + "=" * 50)
        print("测试成功！Python ResNetConfig 类工作正常。")
        print("=" * 50)
        
    except FileNotFoundError:
        print("错误: 找不到 config.json 文件")
        print("提示: 测试将继续，但跳过从文件加载的测试")
        
        # 即使没有 config.json，也可以测试其他功能
        print("\n使用默认值创建配置...")
        config = ResNetConfig()
        print(f"默认配置: {config}")
        
        print("\n使用位置参数创建配置...")
        config2 = ResNetConfig(
            model_type="test",
            num_channels=1,
            embedding_size=32
        )
        print(f"自定义配置: {config2}")
        
        print("\n" + "=" * 50)
        print("测试完成（部分测试因缺少 config.json 而跳过）")
        print("=" * 50)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
