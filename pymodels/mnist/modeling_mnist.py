import sys
import os
import infinicore


# 导入 C++ 绑定模块
try:
    from ..module_loader import load_module
    _infinidemo = load_module("_infinidemo")
except (FileNotFoundError, ImportError) as e:
    raise ImportError(f"Failed to load _infinidemo module: {e}\nPlease run 'xmake' to build the project.")


class MnistForImageClassification(_infinidemo.MnistForImageClassification):
    """
    MNIST模型用于图像分类
    
    这是一个继承自C++实现的Python接口类，提供了更友好的Python API。
    继承自 _infinidemo.MnistForImageClassification，可以直接使用所有C++绑定的方法。
    
    Example:
        >>> import infinicore
        >>> from models.mnist.modeling_mnist import MnistForImageClassification
        >>> 
        >>> # 创建模型
        >>> model = MnistForImageClassification()
        >>> 
        >>> # 加载参数
        >>> state_dict = {
        ...     "weight": infinicore.randn(10, 1936)
        ... }
        >>> model.load_state_dict(state_dict)
        >>> 
        >>> # 前向传播
        >>> input = infinicore.randn(1, 1936)
        >>> output = model(input)
        >>> print(output.shape)
    """
    
    def __init__(self):
        super().__init__() # 调用父类（C++绑定）的构造函数
    
    def forward(self, input):
        """
        前向传播（重写以支持自动类型转换）
        
        Args:
            input: 输入tensor，可以是 infinicore.Tensor
            
        Returns:
            output: 输出tensor
            
        Example:
            >>> import infinicore
            >>> model = MnistForImageClassification()
            >>> input = infinicore.randn(1, 1936)
            >>> output = model.forward(input)
        """
        # 自动处理输入类型转换
        if isinstance(input, infinicore.Tensor):
            # 如果是 infinicore.Tensor，尝试使用 _underlying 属性（如果存在）
            if hasattr(input, '_underlying'):
                return super().forward(input._underlying)
            else:
                return super().forward(input)
        
        raise ValueError(f"Unsupported input type: {type(input)}. Expected infinicore.Tensor")
    
    def __call__(self, input):
        """
        使模型可调用，支持 model(input) 语法
        
        Args:
            input: 输入tensor
            
        Returns:
            output: 输出tensor
        """
        return self.forward(input)
    
    def load_state_dict(self, state_dict, strict=None):
        """
        加载模型参数字典（重写以支持类型转换）
        
        Args:
            state_dict: 参数字典，键为参数名（str），值为tensor
            
        Example:
            >>> import infinicore
            >>> model = MnistForImageClassification()
            >>> state_dict = {
            ...     "weight": infinicore.randn(10, 1936),
            ...     "fc1.weight": infinicore.randn(10, 1936),
            ...     "fc1.bias": infinicore.randn(10),
            ... }
            >>> model.load_state_dict(state_dict)
        """
        # 确保所有值都是 infinicore.Tensor
        processed_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, infinicore.Tensor):
                processed_state_dict[key] = value
            else:
                # 尝试转换为tensor
                import numpy as np
                if isinstance(value, np.ndarray):
                    if hasattr(infinicore, 'from_numpy'):
                        processed_state_dict[key] = infinicore.from_numpy(value)
                    else:
                        processed_state_dict[key] = infinicore.tensor(value)
                else:
                    processed_state_dict[key] = infinicore.tensor(value)
        
        # 调用父类方法
        super().load_state_dict(processed_state_dict)
    
    def __repr__(self):
        """
        返回模型的字符串表示
        """
        return f"MnistForImageClassification()"
    
    @classmethod
    def from_pretrained(cls, model_path):
        """
        从预训练模型加载模型参数
        """
        from ..modeling_utils import load_model_state_dict_by_file

        model = MnistForImageClassification()
        load_model_state_dict_by_file(model, model_path, dtype=infinicore.float32)

        return model
