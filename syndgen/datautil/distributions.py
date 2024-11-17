import numpy as np

from scipy import stats
from typing import Union


class Distribution:
    """"""

    def sample_from_ids(self, ids: list[int]) -> list[int]:
        """"""
        raise NotImplementedError


class Constant(Distribution):
    """"""

    def __init__(self, c: Union[int, float]) -> None:
        """"""
        self._c = c

    def sample_from_ids(self, ids: list[int]) -> list[int]:
        """"""
        samples = [[id_] * self._c for id_ in ids]
        return [i for s in samples for i in s]  # flatten the nested list


class Gamma(Distribution):
    """"""

    def __init__(self, freq: np.ndarray) -> None:
        """"""

        mean = np.mean(freq)
        std = np.std(freq)

        if np.allclose(std, 0, atol=1e-6):
            raise ValueError("std is close to zero.")

        shape = (mean / std) ** 2
        rate = mean / (std ** 2)
        pdf = stats.gamma.pdf(freq, shape, rate)
        probs = pdf / pdf.sum()

        self._freq = freq
        self._probs = probs
    
    def sample_from_ids(self, ids: list[int]) -> list[int]:
        """"""
        amounts = np.random.choice(
            self._freq,
            size=(len(ids), ),
            p=self._probs,
            replace=True,
        )

        samples = [[i] * a for i, a in zip(ids, amounts)]
        return [i for s in samples for i in s]  # flatten nested list
