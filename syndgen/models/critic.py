"""
MIT License

Copyright (c) 2024 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2024-11-17
Last updated: 2024-11-17
"""

import torch
import torch.nn as nn

from typing import (
    Iterable,
    Union,
)


class Critic(nn.Module):
    """Discriminator neural network architecture for Wasserstein GAN with packing."""

    def __init__(
        self,
        in_dim: int,
        layer_dims: Iterable[int],
        *,
        dropout: float = 0.3,
        negative_slope: float = 0.1,
        packing_size: int = 10,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """"""
        super(Critic, self).__init__()

        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)

        self._in_dim = in_dim
        self._layer_dims = layer_dims
        self._dropout = dropout
        self._negative_slope = negative_slope
        self._packing_size = packing_size
        self._device = device

        sequence = []
        dims = [in_dim * packing_size] + layer_dims
        if layer_dims != []:
            for in_, out in zip(dims[:-1], dims[1:]):
                sequence += [
                    nn.Linear(in_, out),
                    nn.Dropout(dropout),
                    nn.LeakyRELU(negative_slope),
                ]
        
        sequence += [nn.Linear(dims[-1], 1)]
        encoder = nn.Sequential(*sequence)

        self._encoder = encoder
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        assert not x.size()[0] % self._packing_size, "batch size must be divisible by packing size"
        return self._encoder(x.view(-1, self._in_dim * self._packing_size))
