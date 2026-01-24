import sys
from pymodels.module_loader import load_module

try:
    # 加载add_module模块
    add_module = load_module("add_module")
except (FileNotFoundError, ImportError) as e:
    print(e)
    sys.exit(1)


# 测试add函数
result = add_module.add(3, 5)
print(f"add(3, 5) = {result}")

result2 = add_module.add(10, 20)
print(f"add(10, 20) = {result2}")

print("测试成功！add函数工作正常。")


import torch

torch.nn.Conv2d
import numpy as np
from pymodels.modeling_utils import load_state_dict

state_dict = load_state_dict(
    "/home/ubuntu/pr666/demo/mydemo/model-mnist.safetensors", dtype=np.float32
)

input = np.ones((1, 1936), dtype=np.float32)

print(input)
fc1_weight = state_dict["fc1.weight"]
fc1_bias = state_dict["fc1.bias"]
print(fc1_weight.shape)
print(fc1_bias.shape)
output = np.matmul(input, fc1_weight.T) + fc1_bias
print(output.shape)
print(output)
"""
[[-29.415178  -14.304864    3.5264838 -14.448445   -3.5334241  -3.1217756
  -22.760998   -1.1969457  -8.300251   -8.323998 ]]
"""
