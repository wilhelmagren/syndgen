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

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from syndgen.datasets import Dataset
from .critic import Critic
from .generator import Generator
from typing import (
    Any,
    Tuple,
    Union,
)


class TGAN(nn.Module):
    """Tabular Wasserstein Generative Adversarial Network with packing and gradient penalty.
    
    [1] Zinan Lin et al., November 2nd 2018, "PacGAN: The power of two samples in generative 
        adversarial networks", https://arxiv.org/abs/1712.04086.

    [2] Martin Arjovsky et al., December 6th 2017, "Wasserstein GAN", https://arxiv.org/abs/1701.07875.
    
    """

    _activation_fn = {
        "tanh": F.tanh,
        "softmax": F.gumbel_softmax,
    }

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        generator_dims: Tuple[int, ...] = (256, 256),
        critic_dims: Tuple[int, ...] = (256, 256),
        generator_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        generator_decay: float = 1e-6,
        critic_decay: float = 1e-6,
        generator_betas: Tuple[float, float] = (0.5, 0.9),
        critic_betas: Tuple[float, float] = (0.5, 0.9),
        batch_size: int = 500,
        packing_size: int = 10,
        n_critic_train_steps: int = 2,
        device: Union[str, torch.device] = "cpu",
        **kwargs: dict[Any, Any],
    ) -> None:
        """"""
        super(TGAN, self).__init__(**kwargs)

        assert not batch_size % packing_size, "batch size must be divisible by packing size"

        self._emb_dim = embedding_dim
        self._generator_dims = generator_dims
        self._critic_dims = critic_dims
        self._generator_lr = generator_lr
        self._critic_lr = critic_lr
        self._generator_decay = generator_decay
        self._critic_decay = critic_decay
        self._generator_betas = generator_betas
        self._critic_betas = critic_betas
        self._batch_size = batch_size
        self._packing_size = packing_size
        self._n_critic_train_steps = n_critic_train_steps
        self._device = device

        self.to(device)

    def fit(
        self,
        dataset: Dataset,
        *,
        discard_critic: bool = False,
        n_epochs: int = 100,
    ) -> TGAN:
        """"""

        if not dataset.is_fitted():
            raise ValueError

        data_dim = dataset.n_output_dims

        generator = Generator(
            emb_dim=self._emb_dim,
            out_dim=data_dim,
            layer_dims=self._generator_dims,
        ).to(self._device)

        critic = Critic(
            in_dim=data_dim,
            layer_dims=self._critic_dims,
            packing_size=self._packing_size,
        ).to(self._device)

        optimizer_G = optim.AdamW(
            generator.parameters(),
            lr=self._generator_lr,
            betas=self._generator_betas,
            weight_decay=self._generator_decay,
        )

        optimizer_C = optim.AdamW(
            critic.parameters(),
            lr=self._critic_lr,
            betas=self._critic_betas,
            weight_decay=self._critic_decay,
        )

        n_steps_per_epoch = max(len(dataset) // self._batch_size, 1)
        for epoch in range(1, n_epochs + 1):
            for step in range(1, n_steps_per_epoch + 1):
                for _ in range(self._n_critic_train_steps):
                    pass
