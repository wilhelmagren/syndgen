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
Last updated: 2024-11-16
"""

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Union,
    Iterable,
)
from ._warnings import NetworkArchitectureWarning


class Residual(nn.Module):
    """One-dimensional residual block for a neural network [1].
    
    [1] Kaiming He et al., 2015
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        layer_dims: Iterable[int] = (),
        project_skip_connection: bool = False,
        device: Union[str, torch.tensor] = "cpu",
    ) -> None:
        """"""
        super(Residual, self).__init__()

        if in_dim != out_dim and not project_skip_connection:
            warnings.warn(
                "input dim does not match output dim, need to project the skip connection to match the output dim",
                NetworkArchitectureWarning,
            )
            project_skip_connection = True

        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._layer_dims = layer_dims
        self._project_skip_connection = project_skip_connection
        self._device = device

        sequence = []
        dims = [in_dim] + layer_dims
        if layer_dims != []:
            for in_, out in zip(dims[:-1], dims[1:]):
                sequence += [
                    nn.Linear(in_, out, bias=False),
                    nn.BatchNorm1d(out),
                    nn.ReLU(),
                ]

        sequence += [
            nn.Linear(dims[-1], out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        ]

        self._block = nn.Sequential(*sequence)
        self._skip_connection = nn.Linear(in_dim, out_dim, bias=False) if project_skip_connection else nn.Identity()

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward call of the residual block.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data to pass through the residual block.

        Returns
        -------
        torch.Tensor
            The computed output tensor.
        
        """
        out = self._block(x) + self._skip_connection(x)
        return F.relu(out)
