import json
import os
from typing import Dict, List, Optional, Union

# 导入 C++ 绑定模块
try:
    from ..module_loader import load_module

    _infinidemo = load_module("_infinidemo")
except (FileNotFoundError, ImportError) as e:
    raise ImportError(
        f"Failed to load _infinidemo module: {e}\nPlease run 'xmake' to build the project."
    )


class ResNetConfig(_infinidemo.ResNetConfig):
    """
    Attributes:
        architectures: List of architecture names
        depths: List of depths for each stage
        downsample_in_first_stage: Whether to downsample in the first stage
        embedding_size: Embedding size
        hidden_act: Hidden activation function name
        hidden_sizes: List of hidden sizes for each stage
        layer_type: Layer type
        model_type: Model type
        num_channels: Number of input channels
        torch_dtype: PyTorch data type
        transformers_version: Transformers library version
    """

    def __init__(
        self,
        architectures: Optional[List[str]] = None,
        depths: Optional[List[int]] = None,
        downsample_in_first_stage: bool = False,
        embedding_size: int = 64,
        hidden_act: str = "relu",
        hidden_sizes: Optional[List[int]] = None,
        layer_type: str = "basic",
        model_type: str = "resnet",
        num_channels: int = 3,
        torch_dtype: str = "float32",
        transformers_version: str = "4.18.0.dev0",
        **kwargs,
    ):
        super().__init__()
        
        if architectures is not None:
            self.architectures = architectures
        if depths is not None:
            self.depths = depths
        self.downsample_in_first_stage = downsample_in_first_stage
        self.embedding_size = embedding_size
        self.hidden_act = hidden_act
        if hidden_sizes is not None:
            self.hidden_sizes = hidden_sizes
        self.layer_type = layer_type
        self.model_type = model_type
        self.num_channels = num_channels
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version

        # 设置额外的关键字参数（用于向后兼容）
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
            if key=="id2label":
                self.num_labels = len(value)

    @classmethod
    def from_pretrained(cls, json_path: Union[str, os.PathLike]) -> "ResNetConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __repr__(self) -> str:
        return super().__repr__()
