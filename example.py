if False:
    import torch
    import torch.nn.functional as F

else:
    import infinicore as torch
    import infinicore.nn.functional as F


def selectDevice():
    import argparse
    import sys

    platform_to_device = {
        "cpu": "cpu",
        "nvidia": "cuda",
        "metax": "cuda",
        "moore": "musa",
        "iluvatar": "cuda",
        "hygon": "cuda",
        "ascend": "npu",
        "cambricon": "mlu",
    }

    parser = argparse.ArgumentParser(description="run Llama args")
    for platform, device_str in platform_to_device.items():
        help_msg = (
            f"Use {platform.upper()} device"
            if platform != "cpu"
            else "Use CPU device (default)"
        )
        parser.add_argument(
            f"--{platform}",
            action="store_true",
            help=help_msg,
        )

    args = parser.parse_args()

    device_str = platform_to_device["cpu"]  # 默认值
    for platform in platform_to_device.keys():
        if getattr(args, platform, False):
            device_str = platform_to_device[platform]
            break

    return torch.device(device_str, 0)


params_dict = {
    "gate_proj.weight": torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float16
    ),
    "gate_proj.bias": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16),
    "up_proj.weight": torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float16
    ),
    "up_proj.bias": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16),
    "down_proj.weight": torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float16
    ),
}


class Network(torch.nn.Module):
    def __init__(self, hidden_size=2, intermediate_size=3):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=True, dtype=torch.float16
        )
        self.up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=True, dtype=torch.float16
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=torch.float16
        )
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


if __name__ == "__main__":
    device = selectDevice()
    print("current device: ", device)

    model = Network()
    model.load_state_dict(params_dict)
    model.to(device)

    input = torch.tensor([[0.1, 0.2]], dtype=torch.float16)
    input = input.to(device)

    output = model(input)

    print(output)
