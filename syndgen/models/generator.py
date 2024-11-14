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

File created: 2024-11-14
Last updated: 2024-11-14
"""

import torch
import torch.nn as nn

from typing import (
    Optional,
    Iterable,
    Union,
    Tuple,
)
from .residual import Residual
from ._errors import NetworkArchitectureError


class Generator(nn.Module):
    """Decoder neural network architecture."""

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        *,
        layer_dims: Iterable[int] = (),
        residual_layer_dims: Iterable[Iterable[int]] = (()),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """ """
        super(Generator, self).__init__()

        if len(layer_dims) != len(residual_layer_dims) and residual_layer_dims != ():
            raise NetworkArchitectureError("Number of layer dims must match the number of residual layers.")

        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        
        if not isinstance(residual_layer_dims, list):
            residual_layer_dims = list(residual_layer_dims)

        self._emb_dim = emb_dim
        self._out_dim = out_dim
        self._layer_dims = layer_dims
        self._residual_layer_dims = residual_layer_dims
        self._device = device

        sequence = []
        dims = [emb_dim] + layer_dims
        if layer_dims != []:
            for i, (in_, out) in enumerate(zip(dims[:-1], dims[1:])):
                sequence += [Residual(in_, out, layer_dims=residual_layer_dims[i])]

        sequence += [nn.Linear(dims[-1], out_dim)]
        decoder = nn.Sequential(*sequence)

        self._decoder = decoder
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """...
        
        Parameters
        ----------
        x : torch.Tensor
            ...

        Returns
        -------
        torch.Tensor
            The decoded output tensor.
        
        """
        return self._decoder(x)
