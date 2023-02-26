import torch

SUPPORTED_MODULES = [
    torch.nn.modules.linear.Linear,
    torch.nn.modules.Sequential, ##used to skip layers
    torch.nn.modules.BatchNorm1d,
    torch.nn.modules.Conv2d,
    torch.nn.modules.BatchNorm2d,
    torch.nn.modules.Flatten,
    torch.nn.modules.AvgPool2d,
    torch.nn.modules.Dropout
]

SUPPORTED_ACT = [
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.LeakyReLU,
]