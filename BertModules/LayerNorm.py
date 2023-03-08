from torch import nn
import torch as t
from typing import Union

class LayerNorm(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self, normalized_shape: Union[int, tuple, t.Size], eps=1e-06, elementwise_affine=True, device=None, dtype=None
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(t.empty(normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(t.empty(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    
    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """x and the output should both have shape (batch, *)."""
        x = (x - x.mean(dim=-1, keepdim=True)) / t.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x