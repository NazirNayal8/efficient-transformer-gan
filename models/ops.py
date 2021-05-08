import torch.nn as nn
import torch
from torch import Tensor
from typing import Any


class MatMul(nn.Module):
    """
    Class for computing matrix multiplication. Used for calculating flops.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x1: Tensor, x2: Tensor) -> Tensor:
        x = x1 @ x2
        return x


def count_matmul(m: nn.Module, x: Any, y: Any) -> None:
    """
    This function is used as part of thop library profiling in order to account for flops caused by
    matrix multiplication operations.
    Params:
    - m: torch module
    - x: a tuple of inputs to module m
    - y: a tuple of outputs of module m
    """
    num_mul = x[0].numel() * x[1].size(-1)
    m.total_ops += torch.DoubleTensor([int(num_mul)])
