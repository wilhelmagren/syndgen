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

File created: 2024-11-13
Last updated: 2024-11-17
"""

from syndgen import datasets  # noqa
from syndgen import datautil  # noqa
from syndgen import metadata  # noqa
from syndgen import metrics  # noqa
from syndgen import models  # noqa
from syndgen import processing  # noqa
from syndgen import sampling  # noqa
from .__version__ import __version__  # noqa

# Local imports. 
import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Set the seed to use for all of the random number generators.

    Parameters
    ----------
    seed : int
        The desired seed.
    
    """

    np.random.seed(seed)
    torch.manual_seed(seed)