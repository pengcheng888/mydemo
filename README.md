# 类CUDA任务的多平台演示

本仓库展示了类CUDA的"一套代码所有平台"的演示，包括 C三段式接口 和 Python一段式接口的示例。

- **一套代码**：同一份代码可以在多个硬件平台上运行，具备的跨平台兼容性
- **多平台支持**：支持 CPU、NVIDIA、METAX、 ILUVATAR、 HYGON
- **C++ 三段式接口**： InfiniOP API 模式（创建描述符 → 获取工作空间 → 执行）
- **Python 接口**： 对齐pytorch的 InfiniCore API

## 项目结构

```
mydemo/
├── example.cpp      # C 三段式 InfiniOP API 示例
├── example.py       # Python一段式 InfiniCore API 示例
├── xmake.lua
└── README.md       
```

## 使用方式

<!-- pip install opencv-python==4.9.0.80 -->
#### 一、编译并安装 `InfiniCore`
编译并安装 `InfiniCore`， 详情见 InfiniCore的 [`README`](https://github.com/InfiniTensor/InfiniCore) :

- 注意根据提示设置好 `INFINI_ROOT` 环境变量（默认为 `$HOME/.infini`）
- 根据硬件平台，选择 xmake 构建配置
- 编译安装InfiniCore
- 安装 C++ 库
- 安装 Python 包

#### 二、克隆`InfiniDemo`仓库

```shell
git clone  https://github.com/InfiniTensor/InfiniDemo.git
```

#### 三、 运行 C++ 示例
执行 GEMM 算子计算：`C = alpha * A * B + beta * C`

```bash
xmake run examples [--cpu | --nvidia | --metax | --moore | --iluvatar]
```
- 例如：
```bash
xmake run examples  --nvidia
```

![image](https://github.com/pengcheng888/mydemo/blob/main/resources/c_cpu.png)

![image](https://github.com/pengcheng888/mydemo/blob/main/resources/c_nvidia.png)



#### 四、 运行 Python 示例
执行 MLP推理计算
```bash
python examples.py [--cpu | --nvidia | --metax | --moore | --iluvatar]
```
- 例如：
```bash
python examples.py --nvidia
```
![image](https://github.com/pengcheng888/mydemo/blob/main/resources/py_cpu.png)

![image](https://github.com/pengcheng888/mydemo/blob/main/resources/py_nvidia.png)

## 各平台测试情况
有7个pr需要合并:

(1)【祝悦】 修复hygon的cuda_fp8文件找不到

https://github.com/InfiniTensor/InfiniCore/pull/865

(2)【】海光平台添加silu和mul算子

 暂无pr

(3)【朱爽】修复天数 TG-V200平台算子精度问题 (但不确定150还能不能跑) 

https://github.com/InfiniTensor/InfiniCore/pull/633

(4)【王鹏程】沐曦平台的出现循环import，文件和类名歧义

https://github.com/InfiniTensor/InfiniCore/pull/944

(5)【王鹏程】 添加infinicore.tensor函数

 https://github.com/InfiniTensor/InfiniCore/pull/894

(6)【王鹏程】 为nn.module添加to函数

https://github.com/InfiniTensor/InfiniCore/pull/891

(7)【王鹏程】 为c++和python中的tensor添加打印函数

https://github.com/InfiniTensor/InfiniCore/pull/930





合并后的测试：

- NVIDIA : 符合预期
- METAX ： 符合预期
- HYGON ： 符合预期
- ILUVATAR ： 符合预期
- MOORE ： 结果对，但出了结果会卡住一会命令行才结束，然后报段错误之类的




## 该仓库的任务需求
1. **三段式接口演示**：一个简单的 C++ 程序，调用 InfiniOP 矩阵乘法接口并打印输入/输出
2. **Python 接口演示**：一个 PyTorch 风格的 MLP 模块，文件顶部有可注释的 `import infinicore as torch`