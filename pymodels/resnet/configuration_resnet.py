import json
import os
from typing import Dict, List, Optional, Union

# 导入 C++ 绑定模块
from ..module_loader import _infinidemo


class ResNetConfig(_infinidemo.ResNetConfig):
    def __init__(
        self,
        architectures: Optional[List[str]] = None,
        depths: Optional[List[int]] = None,
        downsample_in_first_stage: bool = False,
        downsample_in_bottleneck: bool = False,
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
        self.downsample_in_bottleneck = downsample_in_bottleneck
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

        self.id2label = kwargs.get("id2label", None)
        self.label2id = kwargs.get("label2id", None)
        if self.id2label is not None:
            self.num_labels = len(self.id2label)
        else:
            raise ValueError("id2label is not set")

    @classmethod
    def from_pretrained(cls, json_path: Union[str, os.PathLike]) -> "ResNetConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __repr__(self) -> str:
        return super().__repr__()
