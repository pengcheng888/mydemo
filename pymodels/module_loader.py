import sys
import os
import importlib.util


def load_module(module_name, build_dir=None):
    """加载xmake构建的Python扩展模块"""
    # 确定构建目录
    if build_dir is None:
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
    
    # 检查模块是否已加载
    if module_name in sys.modules:
        return sys.modules[module_name]
    
    # 构建模块路径
    module_path = os.path.join(build_dir, f"{module_name}.so")
    
    # 检查文件是否存在
    if not os.path.exists(module_path):
        files = [f for f in os.listdir(build_dir) if f.endswith(".so")] if os.path.exists(build_dir) else []
        msg = f"错误: 找不到{module_name}模块文件\n构建目录: {build_dir}\n"
        if files:
            msg += "构建目录中的文件:\n" + "\n".join(f"  - {f}" for f in files) + "\n"
        raise FileNotFoundError(msg + "\n请先运行 'xmake' 构建项目")
    
    # 加载模块
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法创建模块规范: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise ImportError(f"加载模块失败: {module_name}\n错误: {e}")


try:
    _infinidemo = load_module("_infinidemo")
except (FileNotFoundError, ImportError) as e:
    raise ImportError(f"Failed to load _infinidemo module: {e}\nPlease run 'xmake' to build the project.")
