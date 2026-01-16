# 类CUDA任务的多平台演示

本仓库展示了类CUDA的"一套代码所有平台"的演示，包括 C三段式接口 和 Python一段式接口的示例。

- **一套代码**：同一份代码可以在多个硬件平台上运行，具备的跨平台兼容性。
- **多平台支持**：支持 CPU、NVIDIA、METAX 和 ILUVATAR
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


#### 一、编译并安装 `InfiniCore`
编译并安装 `InfiniCore`， 详情见 InfiniCore的 [`README`](https://github.com/InfiniTensor/InfiniCore) :

- 注意根据提示设置好 `INFINI_ROOT` 环境变量（默认为 `$HOME/.infini`）
- 根据硬件平台，选择 xmake 构建配置
- 编译安装InfiniCore
- 安装 C++ 库
- 安装 Python 包

克隆本仓库 `InfiniDemo`
```shell
git clone  https://github.com/InfiniTensor/InfiniDemo.git
```

#### 二、 运行 C++ 示例
执行 GEMM 计算：`C = alpha * A * B + beta * C`

```bash
xmake run examples [--cpu | --nvidia | --metax | --moore | --iluvatar]
```
- 例如：
```bash
xmake run examples  --nvidia
```

#### 三、 运行 Python++ 示例
执行 MLP推理计算
```bash
python examples.py [--cpu | --nvidia | --metax | --moore | --iluvatar]
```
- 例如：
```bash
python examples.py --nvidia
```

## 该仓库的任务需求
1. **三段式接口演示**：一个简单的 C++ 程序，调用 InfiniOP 矩阵乘法接口并打印输入/输出
2. **Python 接口演示**：一个 PyTorch 风格的 MLP 模块，文件顶部有可注释的 `import infinicore as torch`