#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块加载器 - 用于加载xmake构建的Python扩展模块
"""

import sys
import os
import importlib.util

# 模块缓存，避免重复加载
_loaded_modules = {}


def load_module(module_name, build_dir=None, project_dir=None):
    """
    加载xmake构建的Python扩展模块
    
    Args:
        module_name: 模块名称（例如 'add_module'）
        build_dir: 构建目录路径，如果为None则自动查找
        project_dir: 项目根目录，如果为None则使用当前文件所在目录的父目录
    
    Returns:
        加载的模块对象
    
    Raises:
        FileNotFoundError: 如果找不到构建目录或模块文件
        ImportError: 如果模块加载失败
    """
    # 确定项目目录
    if project_dir is None:
        project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定构建目录
    if build_dir is None:
        build_dir = os.path.join(project_dir, "..", "build")
    
    if not os.path.exists(build_dir):
        raise FileNotFoundError(
            f"错误: 构建目录不存在: {build_dir}\n"
            "请先运行 'xmake' 构建项目"
        )
    
    # 添加构建目录到Python路径
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    
    # 尝试查找模块文件（可能是module_name.so或libmodule_name.so）
    module_files = [
        os.path.join(build_dir, f"{module_name}.so"),
        os.path.join(build_dir, f"lib{module_name}.so"),
        os.path.join(build_dir, f"{module_name}.pyd"),
        os.path.join(build_dir, f"lib{module_name}.pyd"),
    ]
    
    module_path = None
    for path in module_files:
        if os.path.exists(path):
            module_path = path
            break
    
    if module_path is None:
        # 列出构建目录中的文件以便调试
        available_files = [
            f for f in os.listdir(build_dir)
            if f.endswith(('.so', '.pyd'))
        ]
        error_msg = (
            f"错误: 找不到{module_name}模块文件\n"
            f"构建目录: {build_dir}\n"
        )
        if available_files:
            error_msg += "构建目录中的文件:\n"
            for f in available_files:
                error_msg += f"  - {f}\n"
        error_msg += "\n请先运行 'xmake' 构建项目"
        raise FileNotFoundError(error_msg)
    
    # 检查模块是否已经加载（避免重复加载导致pybind11类型重复注册）
    # 首先检查 sys.modules，这是Python的标准模块缓存
    if module_name in sys.modules:
        module = sys.modules[module_name]
        # 同时更新我们的缓存
        _loaded_modules[module_name] = module
        return module
    
    # 检查我们的缓存
    if module_name in _loaded_modules:
        module = _loaded_modules[module_name]
        # 确保也在 sys.modules 中
        sys.modules[module_name] = module
        return module
    
    # 使用importlib动态加载模块
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法创建模块规范: {module_name}")
        
        module = importlib.util.module_from_spec(spec)
        # 先注册到 sys.modules，避免循环导入问题
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        # 缓存模块（基于模块名）
        _loaded_modules[module_name] = module
        return module
    except Exception as e:
        # 如果加载失败，从 sys.modules 中移除
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise ImportError(f"加载模块失败: {module_name}\n错误: {e}")
