import sys
import os
import infinicore

from ..module_loader import _infinidemo


class ResNetForImageClassification(_infinidemo.ResNetForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.num_labels = config.num_labels

    def forward(self, input: infinicore.Tensor):
        return infinicore.Tensor(super().forward(input._underlying))

    def __call__(self, input: infinicore.Tensor):
        return self.forward(input)

    def load_state_dict(self, state_dict, strict=None):
        super().load_state_dict(state_dict)

    def to(self, *, device: infinicore.device):
        super().to(device._underlying)
        return self

    def __repr__(self):
        return super().__repr__()

    @classmethod
    def from_pretrained(cls, model_path):
        from .configuration_resnet import ResNetConfig
        from ..modeling_utils import load_model_state_dict_by_file

        config_path = os.path.join(model_path, "config.json")
        config = ResNetConfig.from_pretrained(config_path)
        model = ResNetForImageClassification(config)
        load_model_state_dict_by_file(model, model_path, dtype=infinicore.float32)
        return model
